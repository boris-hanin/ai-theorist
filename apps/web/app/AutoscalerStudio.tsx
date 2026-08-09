"use client";

import { useEffect, useMemo, useState } from "react";

type Optimizer = "sgd" | "adam" | "muon";
type Activation = "relu" | "gelu" | "tanh";
type BlockType = "pre_norm_mlp" | "pre_norm_moe" | "chizat_mlp";
type DatasetKind = "linear" | "tanh_teacher" | "sinusoid_quadratic";
type Scale = { name: string; width: number; repeats: number; expert_width?: number; particle_width?: number };

type StudySpec = {
  schema_version: 2;
  name: string;
  architecture: {
    block_type: BlockType;
    activation: Activation;
    input_dim: number;
    output_dim: number;
    residual_multiplier: number;
    num_experts?: number;
    active_experts?: number;
    router_balance_rate?: number;
  };
  optimizer: { name: Optimizer };
  dataset: { kind: DatasetKind; n_train: number; n_validation: number; noise_std: number; seed: number; generator_version: 1 };
  horizon: { steps: number; batch_size: number; microbatch_size: null };
  scales: Scale[];
  tuning: { normalized_learning_rates: number[]; max_expansion_rounds: number; expansion_factor: number };
  validation: {
    transfer_probe_decades: number;
    run_negative_control: boolean;
    bootstrap_samples: number;
    routing_load_tolerance: number;
  };
  seeds: number[];
  holdout_count: number;
};

type Progress = { phase: string; completed: number; total: number; message: string };
type ScaleResult = {
  scale: string;
  role: "fit" | "holdout";
  estimated_training_compute: number;
  mean_final_validation_loss: number;
  sem_final_validation_loss: number;
};
type Calibration = {
  scale: string;
  predicted_final_validation_loss: number;
  observed_final_validation_loss: number;
  relative_error: number;
  accepted: boolean;
};
type StudyResult = {
  forecastable: boolean;
  refusal_reasons: string[];
  warnings: string[];
  scale_results: ScaleResult[];
  holdout_calibration: Calibration[];
  tuning: { selected_normalized_learning_rate: number; optimum_is_interior: boolean };
  scaling_law: { loss_floor: number; exponent: number; r_squared: number; forecastable: boolean };
  next_scale_forecast: null | {
    mode: "asymptotic_floor_power_law" | "heldout_calibrated_one_step";
    predicted_final_validation_loss: number;
    prediction_interval_95: [number, number];
    estimated_training_compute: number;
  };
};
type StudyJob = {
  id: string;
  status: "queued" | "running" | "completed" | "failed";
  progress: Progress;
  result: StudyResult | null;
  error: string | null;
};

const API_BASE = process.env.NEXT_PUBLIC_AUTOSCALER_API ?? "http://127.0.0.1:8787";
const INITIAL_SCALES: Scale[] = [
  { name: "S1", width: 16, repeats: 1 },
  { name: "S2", width: 24, repeats: 2 },
  { name: "S3", width: 32, repeats: 3 },
  { name: "S4", width: 48, repeats: 4 },
  { name: "S5", width: 64, repeats: 6 },
];
const MOE_SCALES: Scale[] = [
  { name: "S1", width: 8, repeats: 2, expert_width: 16 },
  { name: "S2", width: 18, repeats: 3, expert_width: 24 },
  { name: "S3", width: 32, repeats: 4, expert_width: 32 },
  { name: "S4", width: 50, repeats: 5, expert_width: 40 },
  { name: "S5", width: 72, repeats: 6, expert_width: 48 },
];
const CHIZAT_SCALES: Scale[] = [
  { name: "S1", width: 8, repeats: 2, particle_width: 16 },
  { name: "S2", width: 18, repeats: 3, particle_width: 24 },
  { name: "S3", width: 32, repeats: 4, particle_width: 32 },
  { name: "S4", width: 50, repeats: 5, particle_width: 40 },
  { name: "S5", width: 72, repeats: 6, particle_width: 48 },
];

function parameterCount(scale: Scale, blockType: BlockType, numExperts: number) {
  const width = scale.width;
  if (blockType === "chizat_mlp") {
    return 16 * width + 2 * scale.repeats * width * (scale.particle_width ?? 1) + width;
  }
  const block = blockType === "pre_norm_mlp"
    ? 2 * width * width + 4 * width
    : 2 * width
      + width * numExperts + numExperts
      + numExperts * (2 * width * (scale.expert_width ?? 1) + (scale.expert_width ?? 1) + width);
  return 16 * width + width + scale.repeats * block + 2 * width + width + 1;
}

function formatNumber(value: number) {
  return new Intl.NumberFormat("en-US", { maximumFractionDigits: 2, notation: "compact" }).format(value);
}

function formatLoss(value: number) {
  return value < 0.01 ? value.toExponential(2) : value.toFixed(4);
}

function LossChart({ rows }: { rows: ScaleResult[] }) {
  if (rows.length < 2) return null;
  const width = 720;
  const height = 220;
  const pad = 32;
  const logs = rows.map((row) => Math.log10(row.estimated_training_compute));
  const losses = rows.map((row) => row.mean_final_validation_loss);
  const xMin = Math.min(...logs);
  const xMax = Math.max(...logs);
  const yMin = Math.min(...losses) * 0.96;
  const yMax = Math.max(...losses) * 1.04;
  const x = (value: number) => pad + ((value - xMin) / Math.max(1e-9, xMax - xMin)) * (width - 2 * pad);
  const y = (value: number) => height - pad - ((value - yMin) / Math.max(1e-9, yMax - yMin)) * (height - 2 * pad);
  const points = rows.map((row, index) => `${x(logs[index])},${y(row.mean_final_validation_loss)}`).join(" ");

  return (
    <div className="chart-wrap" aria-label="Validation loss by estimated training compute">
      <svg viewBox={`0 0 ${width} ${height}`} role="img">
        <line x1={pad} y1={height - pad} x2={width - pad} y2={height - pad} className="axis" />
        <line x1={pad} y1={pad} x2={pad} y2={height - pad} className="axis" />
        <polyline points={points} className="loss-line" />
        {rows.map((row, index) => (
          <g key={row.scale}>
            <circle
              cx={x(logs[index])}
              cy={y(row.mean_final_validation_loss)}
              r={row.role === "holdout" ? 7 : 5}
              className={row.role === "holdout" ? "dot holdout-dot" : "dot"}
            />
            <text x={x(logs[index])} y={height - 10} textAnchor="middle">{row.scale}</text>
          </g>
        ))}
        <text x={pad + 4} y={20}>{formatLoss(yMax)}</text>
        <text x={pad + 4} y={height - pad - 8}>{formatLoss(yMin)}</text>
      </svg>
      <div className="chart-legend"><span><i /> Fit scales</span><span><i className="held" /> Held out</span></div>
    </div>
  );
}

export function AutoscalerStudio() {
  const [blockType, setBlockType] = useState<BlockType>("pre_norm_mlp");
  const [activation, setActivation] = useState<Activation>("gelu");
  const [numExperts, setNumExperts] = useState(4);
  const [activeExperts, setActiveExperts] = useState(1);
  const [optimizer, setOptimizer] = useState<Optimizer>("adam");
  const [scales, setScales] = useState<Scale[]>(INITIAL_SCALES);
  const [steps, setSteps] = useState(40);
  const [datasetSize, setDatasetSize] = useState(512);
  const [datasetKind, setDatasetKind] = useState<DatasetKind>("sinusoid_quadratic");
  const [selectedNode, setSelectedNode] = useState<"embed" | "residual" | "unembed">("residual");
  const [apiOnline, setApiOnline] = useState<boolean | null>(null);
  const [job, setJob] = useState<StudyJob | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [draggingBlock, setDraggingBlock] = useState<BlockType | null>(null);

  const learningRates = useMemo(
    () => blockType === "chizat_mlp"
      ? optimizer === "muon"
        ? [0.03, 0.05, 0.08, 0.1, 0.14]
        : optimizer === "sgd"
          ? [0.0003, 0.001, 0.003, 0.01, 0.03]
          : [0.003, 0.01, 0.03, 0.1, 0.3]
      : optimizer === "adam"
        ? blockType === "pre_norm_moe" ? [0.03, 0.1, 0.3, 1, 3] : [0.0001, 0.0003, 0.001, 0.003, 0.01]
        : [0.02, 0.06, 0.2, 0.6, 2.0],
    [blockType, optimizer],
  );

  const spec = useMemo<StudySpec>(() => ({
    schema_version: 2,
    name: `${optimizer}-${blockType === "pre_norm_moe" ? "moe" : blockType === "chizat_mlp" ? "chizat" : "mlp"}-fixed-horizon`,
    architecture: {
      block_type: blockType,
      activation,
      input_dim: 16,
      output_dim: 1,
      residual_multiplier: 1,
      ...(blockType === "pre_norm_moe" ? {
        num_experts: numExperts,
        active_experts: activeExperts,
        router_balance_rate: 0.1,
      } : {}),
    },
    optimizer: { name: optimizer },
    dataset: { kind: datasetKind, n_train: datasetSize, n_validation: 256, noise_std: 0.03, seed: 1729, generator_version: 1 },
    horizon: { steps, batch_size: 64, microbatch_size: null },
    scales,
    tuning: {
      normalized_learning_rates: learningRates,
      max_expansion_rounds: 1,
      expansion_factor: 3,
    },
    validation: { transfer_probe_decades: 0.3, run_negative_control: true, bootstrap_samples: 200, routing_load_tolerance: 0.25 },
    seeds: [11, 29],
    holdout_count: 1,
  }), [activation, activeExperts, blockType, numExperts, optimizer, datasetKind, datasetSize, steps, scales, learningRates]);

  // Reference tuning and the center holdout probe are reused from cache.
  const estimatedTrials = learningRates.length * 2 + (scales.length - 1) * 2 + 2 * 2 + 2;
  const totalParameters = parameterCount(scales[scales.length - 1], blockType, numExperts);
  const lmOverD = scales.map((scale) => scale.repeats * (scale.expert_width ?? scale.particle_width ?? 0) / scale.width);
  const jointInvariant = blockType === "pre_norm_mlp"
    || Math.max(...lmOverD) - Math.min(...lmOverD) < 1e-9;
  const planValid = scales.length >= 5 && scales.every((scale, index) => (
    scale.width >= 4
    && scale.repeats >= 1
    && (blockType !== "pre_norm_moe" || (scale.expert_width ?? 0) >= 2)
    && (blockType !== "chizat_mlp" || (scale.particle_width ?? 0) >= 2)
    && (index === 0 || parameterCount(scale, blockType, numExperts) > parameterCount(scales[index - 1], blockType, numExperts))
  )) && activeExperts >= 1 && activeExperts <= numExperts;
  const studyLocked = job?.status === "queued" || job?.status === "running";

  useEffect(() => {
    const controller = new AbortController();
    fetch(`${API_BASE}/api/health`, { signal: controller.signal })
      .then((response) => setApiOnline(response.ok))
      .catch(() => setApiOnline(false));
    return () => controller.abort();
  }, []);

  useEffect(() => {
    if (!job || !["queued", "running"].includes(job.status)) return;
    const timer = window.setTimeout(async () => {
      try {
        const response = await fetch(`${API_BASE}/api/studies/${job.id}`);
        if (!response.ok) throw new Error("The study monitor lost contact with the compute service.");
        setJob(await response.json() as StudyJob);
      } catch (caught) {
        setError(caught instanceof Error ? caught.message : "Could not refresh this study.");
      }
    }, 1200);
    return () => window.clearTimeout(timer);
  }, [job]);

  function updateScale(index: number, field: "width" | "repeats" | "expert_width" | "particle_width", value: number) {
    if (studyLocked) return;
    setScales((current) => current.map((scale, scaleIndex) => (
      scaleIndex === index ? { ...scale, [field]: Math.max(field === "width" ? 4 : field === "repeats" ? 1 : 2, value || 1) } : scale
    )));
    setJob(null);
  }

  function chooseBlock(next: BlockType) {
    if (studyLocked) return;
    if (next !== blockType) setScales(next === "pre_norm_moe" ? MOE_SCALES : next === "chizat_mlp" ? CHIZAT_SCALES : INITIAL_SCALES);
    setBlockType(next);
    if (next === "pre_norm_moe" || (next !== "chizat_mlp" && optimizer === "muon")) setOptimizer("adam");
    setActivation(next === "chizat_mlp" ? "tanh" : activation === "tanh" ? "gelu" : activation);
    setSelectedNode("residual");
    setJob(null);
  }

  function chooseActivation(next: Activation) {
    if (studyLocked) return;
    setActivation(next);
    setJob(null);
  }

  async function startStudy() {
    if (!planValid || studyLocked) return;
    setError(null);
    try {
      const response = await fetch(`${API_BASE}/api/studies`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ spec, device: "cpu" }),
      });
      const payload = await response.json() as StudyJob & { error?: string };
      if (!response.ok) throw new Error(payload.error ?? "The study could not start.");
      setApiOnline(true);
      setJob(payload);
      document.getElementById("run")?.scrollIntoView({ behavior: "smooth", block: "start" });
    } catch (caught) {
      setApiOnline(false);
      setError(caught instanceof Error ? caught.message : "The local compute service is unavailable.");
    }
  }

  const progressPercent = job?.progress.total
    ? Math.min(100, Math.round((job.progress.completed / job.progress.total) * 100))
    : 0;
  const result = job?.result;

  return (
    <main>
      <header className="topbar">
        <a className="brand" href="#top" aria-label="AI Theorist Autoscaler home">
          <span className="brand-mark"><b /><b /><b /></span>
          <span><strong>AI Theorist</strong><small>Autoscaler</small></span>
        </a>
        <nav aria-label="Workspace sections">
          <a href="#architecture">Architecture</a>
          <a href="#study">Study</a>
          <a href="#run">Results</a>
        </nav>
        <div className={`service-pill ${apiOnline ? "online" : ""}`}>
          <span /> {apiOnline === null ? "Checking compute" : apiOnline ? "Compute ready" : "Compute offline"}
        </div>
      </header>

      <section className="hero" id="top">
        <div>
          <p className="eyebrow">Fixed-horizon scaling workbench</p>
          <h1>Build the model. <span>Measure the law.</span></h1>
          <p className="hero-copy">Compose a residual network, tune one scale-normalized learning rate, and earn a largest-model forecast through held-out calibration.</p>
        </div>
        <div className="method-strip" aria-label="Study method">
          <div><strong>01</strong><span>Compose</span></div><i />
          <div><strong>02</strong><span>Tune & transfer</span></div><i />
          <div><strong>03</strong><span>Calibrate</span></div><i />
          <div><strong>04</strong><span>Forecast</span></div>
        </div>
      </section>

      <section className="workspace section-shell" id="architecture">
        <div className="section-heading">
          <div><p className="eyebrow">Architecture</p><h2>A deliberately typed canvas</h2></div>
          <p>Drag a residual block into the stack. Embed and unembed are always trained under explicit optimizer roles.</p>
        </div>

        <div className="builder-grid">
          <aside className="panel palette-panel">
            <div className="panel-title"><span>Component library</span><small>3 blocks</small></div>
            <p className="panel-note">Drag or click to replace the residual stack.</p>
            {(["pre_norm_mlp", "chizat_mlp", "pre_norm_moe"] as BlockType[]).map((item) => (
              <button
                className={`palette-card ${blockType === item ? "active" : ""}`}
                key={item}
                draggable={!studyLocked}
                disabled={studyLocked}
                onDragStart={(event) => event.dataTransfer.setData("application/x-autoscaler-block", item)}
                onDragEnd={() => setDraggingBlock(null)}
                onPointerDown={() => {
                  setDraggingBlock(item);
                  chooseBlock(item);
                }}
                onPointerUp={() => setDraggingBlock(null)}
                onClick={() => chooseBlock(item)}
              >
                <span className="drag-grip">⠿</span>
                <span className={`block-glyph ${item === "pre_norm_moe" ? "moe" : "mlp"}`}><i /><i /></span>
                <span>
                  <strong>{item === "pre_norm_moe" ? "Sparse MoE" : item === "chizat_mlp" ? "Chizat particles" : "Pre-norm MLP"}</strong>
                  <small>{item === "pre_norm_moe" ? "Top-k expert routing" : item === "chizat_mlp" ? "Mean-field residual cell" : "Dense residual cell"}</small>
                </span>
                <b>{blockType === item ? "In use" : "Add"}</b>
              </button>
            ))}
            <div className="coming-soon"><span>Later</span><p>Attention, convolution, AdamW, and MoE with SGD remain outside this validated slice.</p></div>
          </aside>

          <div className="panel canvas-panel">
            <div className="panel-title"><span>Model canvas</span><small>Typed linear graph</small></div>
            <div className="canvas-grid" />
            <div className="model-graph">
              <button className={`model-node compact ${selectedNode === "embed" ? "selected" : ""}`} onClick={() => setSelectedNode("embed")}>
                <span className="node-index">01</span><span><small>Input adapter</small><strong>Embed</strong><em>16 → D</em></span><b>Trained</b>
              </button>
              <div className="connector"><span /></div>
              <button
                className={`model-node residual-node ${selectedNode === "residual" ? "selected" : ""}`}
                onClick={() => setSelectedNode("residual")}
                onDragOver={(event) => event.preventDefault()}
                onPointerEnter={(event) => {
                  if (draggingBlock && event.buttons === 1) chooseBlock(draggingBlock);
                }}
                onPointerUp={() => setDraggingBlock(null)}
                onDrop={(event) => {
                  event.preventDefault();
                  const item = event.dataTransfer.getData("application/x-autoscaler-block");
                  if (item === "pre_norm_mlp" || item === "pre_norm_moe" || item === "chizat_mlp") chooseBlock(item);
                }}
              >
                <span className="node-index">02</span>
                <span><small>Repeatable cell</small><strong>Residual stack</strong><em>{blockType === "pre_norm_moe" ? `Sparse MoE · ${numExperts} experts · top ${activeExperts}` : blockType === "chizat_mlp" ? "Chizat mean-field particles" : "Pre-norm MLP"} · {activation.toUpperCase()}</em></span>
                <div className="repeat-badge">× R</div>
              </button>
              <div className="connector"><span /></div>
              <button className={`model-node compact ${selectedNode === "unembed" ? "selected" : ""}`} onClick={() => setSelectedNode("unembed")}>
                <span className="node-index">03</span><span><small>Output adapter</small><strong>Unembed</strong><em>D → 1</em></span><b>Trained</b>
              </button>
            </div>
            <div className="canvas-caption"><span /> Shape-safe by construction</div>
          </div>

          <aside className="panel inspector-panel">
            <div className="panel-title"><span>Inspector</span><small>{selectedNode}</small></div>
            {selectedNode === "residual" ? (
              <>
                <div className="field-label">Activation</div>
                {blockType === "chizat_mlp" ? (
                  <div className="readonly-field"><span>Validated nonlinearity</span><strong>Tanh</strong></div>
                ) : (
                  <div className="segmented">
                    <button disabled={studyLocked} className={activation === "gelu" ? "active" : ""} onClick={() => chooseActivation("gelu")}>GELU</button>
                    <button disabled={studyLocked} className={activation === "relu" ? "active" : ""} onClick={() => chooseActivation("relu")}>ReLU</button>
                  </div>
                )}
                <div className="field-label">Residual parameterization</div>
                {blockType === "pre_norm_moe" ? (
                  <>
                    <div className="paired-inputs inspector-inputs">
                      <label><span>Stored experts E</span><input disabled={studyLocked} type="number" min="1" value={numExperts} onChange={(event) => { const value = Math.max(1, Number(event.target.value)); setNumExperts(value); setActiveExperts((current) => Math.min(current, value)); setJob(null); }} /></label>
                      <label><span>Active experts a</span><input disabled={studyLocked} type="number" min="1" max={numExperts} value={activeExperts} onChange={(event) => { setActiveExperts(Math.min(numExperts, Math.max(1, Number(event.target.value)))); setJob(null); }} /></label>
                    </div>
                    <div className="readonly-field"><span>Active fraction κ</span><strong>{activeExperts}/{numExperts}</strong></div>
                    <div className="readonly-field"><span>Branch multiplier</span><strong>1 / L</strong></div>
                    <div className="formula-card"><small>Transfer contract</small><code>keep LM/D fixed · router/up η/D · down η/M</code></div>
                  </>
                ) : blockType === "chizat_mlp" ? (
                  <>
                    <div className="readonly-field"><span>Branch multiplier</span><strong>1 / (L × M)</strong></div>
                    <div className="formula-card"><small>Transfer contract</small><code>trained E,R · semantic rates in D,L,M</code></div>
                  </>
                ) : (
                  <>
                    <div className="readonly-field"><span>Branch multiplier</span><strong>1 / √R</strong></div>
                    <div className="formula-card"><small>Cell definition</small><code>x ← x + f(LN(x)) / √R</code></div>
                  </>
                )}
                <div className="readonly-field"><span>Normalization</span><strong>{blockType === "chizat_mlp" ? "None" : "Pre-norm"}</strong></div>
              </>
            ) : (
              <div className="locked-inspector"><span>Trained</span><h3>{selectedNode === "embed" ? "Dataset embedding" : "Validation head"}</h3><p>This adapter is initialized by the architecture contract and trained at every scale with its own declared learning-rate role.</p></div>
            )}
          </aside>
        </div>
      </section>

      <section className="study-section section-shell" id="study">
        <div className="section-heading">
          <div><p className="eyebrow">Study design</p><h2>One horizon, five scales</h2></div>
          <p>{blockType === "pre_norm_moe" ? "Depth L, expert width M, and embedding D follow the validated LM/D path. Expert sparsity stays fixed." : blockType === "chizat_mlp" ? "Depth L, particle width M, and representation width D scale together under semantic optimizer rates." : "Only width and repeat count grow."} The dataset and training horizon remain fixed.</p>
        </div>

        <div className="study-grid">
          <div className="panel scale-panel">
            <div className="panel-title"><span>Scale ladder</span><small>Largest level held out</small></div>
            <div className="scale-table" role="table" aria-label="Model scale ladder">
              <div className={`scale-row table-head ${blockType !== "pre_norm_mlp" ? "moe-scale-row" : ""}`} role="row"><span>Level</span><span>Width D</span><span>{blockType !== "pre_norm_mlp" ? "Depth L" : "Repeats R"}</span>{blockType !== "pre_norm_mlp" && <span>{blockType === "pre_norm_moe" ? "Expert M" : "Particle M"}</span>}<span>Parameters</span><span>Role</span></div>
              {scales.map((scale, index) => (
                <div className={`scale-row ${blockType !== "pre_norm_mlp" ? "moe-scale-row" : ""}`} role="row" key={scale.name}>
                  <strong>{scale.name}</strong>
                  <input disabled={studyLocked} aria-label={`${scale.name} width`} type="number" min="4" step="4" value={scale.width} onChange={(event) => updateScale(index, "width", Number(event.target.value))} />
                  <input disabled={studyLocked} aria-label={`${scale.name} repeats`} type="number" min="1" value={scale.repeats} onChange={(event) => updateScale(index, "repeats", Number(event.target.value))} />
                  {blockType === "pre_norm_moe" && <input disabled={studyLocked} aria-label={`${scale.name} expert width`} type="number" min="2" value={scale.expert_width} onChange={(event) => updateScale(index, "expert_width", Number(event.target.value))} />}
                  {blockType === "chizat_mlp" && <input disabled={studyLocked} aria-label={`${scale.name} particle width`} type="number" min="2" value={scale.particle_width} onChange={(event) => updateScale(index, "particle_width", Number(event.target.value))} />}
                  <span>{formatNumber(parameterCount(scale, blockType, numExperts))}</span>
                  <span className={index === scales.length - 1 ? "role holdout" : "role"}>{index === scales.length - 1 ? "Holdout" : "Fit"}</span>
                </div>
              ))}
            </div>
            {!planValid && <p className="validation-error">Each level must have strictly more parameters than the previous one.</p>}
            {blockType !== "pre_norm_mlp" && !jointInvariant && <p className="validation-warning">For the strongest transfer, keep L × M / D constant across the ladder.</p>}
          </div>

          <aside className="panel protocol-panel">
            <div className="panel-title"><span>Training protocol</span><small>Immutable after launch</small></div>
            <div className="field-label">Optimizer</div>
            <div className="optimizer-choice">
              <button disabled={studyLocked || blockType === "pre_norm_moe"} title={blockType === "pre_norm_moe" ? "MoE + SGD is not certified yet" : undefined} className={optimizer === "sgd" ? "active" : ""} onClick={() => { if (!studyLocked && blockType !== "pre_norm_moe") { setOptimizer("sgd"); setJob(null); } }}><strong>SGD</strong><small>{blockType === "pre_norm_moe" ? "Not certified for MoE" : "Momentum 0"}</small></button>
              <button disabled={studyLocked} className={optimizer === "adam" ? "active" : ""} onClick={() => { if (!studyLocked) { setOptimizer("adam"); setJob(null); } }}><strong>Adam</strong><small>β₁ .9 · β₂ .999</small></button>
              <button disabled={studyLocked || blockType !== "chizat_mlp"} title={blockType !== "chizat_mlp" ? "Muon is validated only for Chizat particles" : undefined} className={optimizer === "muon" ? "active" : ""} onClick={() => { if (!studyLocked && blockType === "chizat_mlp") { setOptimizer("muon"); setJob(null); } }}><strong>Muon</strong><small>U/W Muon · E/R Adam</small></button>
            </div>
            <label className="dataset-select"><span>Dataset task</span><select disabled={studyLocked} value={datasetKind} onChange={(event) => { setDatasetKind(event.target.value as DatasetKind); setJob(null); }}><option value="sinusoid_quadratic">Sinusoid + quadratic</option><option value="tanh_teacher">Tanh teacher</option><option value="linear">Linear control</option></select></label>
            <div className="paired-inputs">
              <label><span>Training points</span><input disabled={studyLocked} type="number" min="128" step="128" value={datasetSize} onChange={(event) => { if (!studyLocked) { setDatasetSize(Math.max(128, Number(event.target.value))); setJob(null); } }} /></label>
              <label><span>Update steps</span><input disabled={studyLocked} type="number" min="10" step="10" value={steps} onChange={(event) => { if (!studyLocked) { setSteps(Math.max(10, Number(event.target.value))); setJob(null); } }} /></label>
            </div>
            <div className="fixed-callout"><b>Fixed across every scale</b><span>{formatNumber(datasetSize)} points · {formatNumber(steps * 64)} sample presentations</span></div>
            <div className="seed-row"><span>Common random seeds</span><strong>11</strong><strong>29</strong></div>
          </aside>
        </div>

        <div className="review-bar">
          <div><span className="review-check">✓</span><p><strong>Schema valid</strong><small>Embed → {blockType === "pre_norm_moe" ? "MoE" : "residual"} → unembed</small></p></div>
          <div><span className="review-check">✓</span><p><strong>Paired design</strong><small>Common seeds at every probe</small></p></div>
          <div><span className="review-check">✓</span><p><strong>Honest holdout</strong><small>{scales[scales.length - 1].name} excluded from fit</small></p></div>
          <div className="budget"><small>Planned trials</small><strong>{estimatedTrials}</strong><span>+ edge expansion if needed</span></div>
          <button className="run-button" disabled={!planValid || studyLocked} onClick={startStudy}>
            {job?.status === "running" ? "Study running" : "Lock plan & run"}<span>→</span>
          </button>
        </div>
      </section>

      <section className="results-section section-shell" id="run">
        <div className="section-heading results-heading">
          <div><p className="eyebrow">Calibration report</p><h2>Forecasts must earn the right to appear</h2></div>
          {job && <span className={`status-badge ${job.status}`}>{job.status}</span>}
        </div>

        {error && <div className="error-banner"><strong>Compute service unavailable</strong><span>{error}</span><button onClick={() => setError(null)}>Dismiss</button></div>}

        {!job && (
          <div className="empty-results">
            <div className="empty-plot"><span /><span /><span /><span /><i /></div>
            <div><p className="eyebrow">No study yet</p><h3>Your held-out result lands here.</h3><p>Launch the immutable plan above. The system will tune normalized η on the reference model, hold it fixed across scale, fit only the smaller levels, then reveal the largest model.</p></div>
          </div>
        )}

        {job && ["queued", "running"].includes(job.status) && (
          <div className="running-card">
            <div className="run-orbit"><span>{progressPercent}%</span><i /></div>
            <div className="run-copy"><p className="eyebrow">{job.progress.phase.replaceAll("-", " ")}</p><h3>{job.progress.message}</h3><div className="progress-track"><span style={{ width: `${progressPercent}%` }} /></div><p>{job.progress.completed} of about {job.progress.total || estimatedTrials} trials · final validation loss only</p></div>
            <div className="run-id"><small>Study</small><code>{job.id}</code></div>
          </div>
        )}

        {job?.status === "failed" && <div className="failed-card"><strong>Study stopped safely</strong><p>{job.error}</p><button onClick={startStudy}>Run a fresh study</button></div>}

        {result && (
          <div className="result-grid">
            <div className="panel result-main">
              <div className="result-title"><div><span className={`verdict-mark ${result.forecastable ? "pass" : "refuse"}`}>{result.forecastable ? "✓" : "!"}</span><div><small>Forecast verdict</small><h3>{result.forecastable ? "Calibrated to forecast" : "Forecast withheld"}</h3></div></div><span>Final validation loss</span></div>
              <LossChart rows={result.scale_results} />
              <div className="metric-row">
                <div><small>Selected normalized η</small><strong>{result.tuning.selected_normalized_learning_rate.toExponential(1)}</strong><span>{result.tuning.optimum_is_interior ? "Interior optimum" : "Boundary warning"}</span></div>
                <div><small>Scaling exponent α</small><strong>{result.scaling_law.exponent.toFixed(3)}</strong><span>R² {result.scaling_law.r_squared.toFixed(3)}</span></div>
                <div><small>Estimated loss floor</small><strong>{formatLoss(result.scaling_law.loss_floor)}</strong><span>Fixed horizon</span></div>
              </div>
            </div>
            <aside className="panel evidence-panel">
              <div className="panel-title"><span>Held-out evidence</span><small>{result.holdout_calibration.length} check</small></div>
              {result.holdout_calibration.map((calibration) => (
                <div className="calibration-card" key={calibration.scale}>
                  <div><strong>{calibration.scale}</strong><span className={calibration.accepted ? "accepted" : "rejected"}>{calibration.accepted ? "Pass" : "Miss"}</span></div>
                  <p><span>Predicted</span><strong>{formatLoss(calibration.predicted_final_validation_loss)}</strong></p>
                  <p><span>Observed</span><strong>{formatLoss(calibration.observed_final_validation_loss)}</strong></p>
                  <p><span>Relative error</span><strong>{(calibration.relative_error * 100).toFixed(1)}%</strong></p>
                </div>
              ))}
              {result.next_scale_forecast ? (
                <div className="forecast-card"><small>{result.next_scale_forecast.mode === "heldout_calibrated_one_step" ? "Calibrated one-step prediction" : "Next-scale prediction"}</small><strong>{formatLoss(result.next_scale_forecast.predicted_final_validation_loss)}</strong><span>95% interval {formatLoss(result.next_scale_forecast.prediction_interval_95[0])}–{formatLoss(result.next_scale_forecast.prediction_interval_95[1])}</span>{result.warnings.map((warning) => <em key={warning}>{warning}</em>)}</div>
              ) : (
                <div className="refusal-card"><strong>No extrapolation issued</strong><ul>{result.refusal_reasons.map((reason) => <li key={reason}>{reason}</li>)}</ul></div>
              )}
            </aside>
          </div>
        )}
      </section>

      <footer><span>AI Theorist · Autoscaler v0.1</span><p>Typed architectures. Paired evidence. Refusal before false precision.</p><span>Largest plan level: {formatNumber(totalParameters)} parameters</span></footer>
    </main>
  );
}
