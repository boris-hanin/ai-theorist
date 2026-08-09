"use client";

import { useEffect, useMemo, useState } from "react";

type Optimizer = "sgd" | "adam";
type Activation = "relu" | "gelu";
type Scale = { name: string; width: number; repeats: number };

type StudySpec = {
  schema_version: 1;
  name: string;
  architecture: {
    block_type: "pre_norm_mlp";
    activation: Activation;
    input_dim: number;
    output_dim: number;
    residual_multiplier: number;
  };
  optimizer: { name: Optimizer };
  dataset: { n_train: number; n_validation: number; noise_std: number; seed: number };
  horizon: { steps: number; batch_size: number; microbatch_size: null };
  scales: Scale[];
  tuning: { learning_rates: number[]; max_expansion_rounds: number; expansion_factor: number };
  validation: {
    transfer_probe_decades: number;
    run_negative_control: boolean;
    bootstrap_samples: number;
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
  tuning: { selected_learning_rate: number; optimum_is_interior: boolean };
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

function parameterCount(scale: Scale) {
  const width = scale.width;
  return 16 * width + width + scale.repeats * (2 * width * width + 4 * width) + 2 * width + width + 1;
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
  const [activation, setActivation] = useState<Activation>("gelu");
  const [optimizer, setOptimizer] = useState<Optimizer>("adam");
  const [scales, setScales] = useState<Scale[]>(INITIAL_SCALES);
  const [steps, setSteps] = useState(40);
  const [datasetSize, setDatasetSize] = useState(512);
  const [selectedNode, setSelectedNode] = useState<"embed" | "residual" | "unembed">("residual");
  const [apiOnline, setApiOnline] = useState<boolean | null>(null);
  const [job, setJob] = useState<StudyJob | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [draggingActivation, setDraggingActivation] = useState<Activation | null>(null);

  const learningRates = useMemo(
    () => optimizer === "adam"
      ? [0.0001, 0.0003, 0.001, 0.003, 0.01]
      : [0.003, 0.01, 0.03, 0.1, 0.3],
    [optimizer],
  );

  const spec = useMemo<StudySpec>(() => ({
    schema_version: 1,
    name: `${optimizer}-mlp-fixed-horizon`,
    architecture: {
      block_type: "pre_norm_mlp",
      activation,
      input_dim: 16,
      output_dim: 1,
      residual_multiplier: 1,
    },
    optimizer: { name: optimizer },
    dataset: { n_train: datasetSize, n_validation: 256, noise_std: 0.03, seed: 1729 },
    horizon: { steps, batch_size: 64, microbatch_size: null },
    scales,
    tuning: { learning_rates: learningRates, max_expansion_rounds: 1, expansion_factor: 3 },
    validation: { transfer_probe_decades: 0.3, run_negative_control: true, bootstrap_samples: 200 },
    seeds: [11, 29],
    holdout_count: 1,
  }), [activation, optimizer, datasetSize, steps, scales, learningRates]);

  // Reference tuning and the center holdout probe are reused from cache.
  const estimatedTrials = learningRates.length * 2 + (scales.length - 1) * 2 + 2 * 2 + 2;
  const totalParameters = parameterCount(scales[scales.length - 1]);
  const planValid = scales.length >= 5 && scales.every((scale, index) => (
    scale.width >= 4 && scale.repeats >= 1 && (index === 0 || parameterCount(scale) > parameterCount(scales[index - 1]))
  ));
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

  function updateScale(index: number, field: "width" | "repeats", value: number) {
    if (studyLocked) return;
    setScales((current) => current.map((scale, scaleIndex) => (
      scaleIndex === index ? { ...scale, [field]: Math.max(field === "width" ? 4 : 1, value || 1) } : scale
    )));
    setJob(null);
  }

  function chooseBlock(next: Activation) {
    if (studyLocked) return;
    setActivation(next);
    setSelectedNode("residual");
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
          <p className="hero-copy">Compose a residual network, tune one global learning rate, and earn a largest-model forecast through held-out calibration.</p>
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
          <p>Drag a residual block into the stack. Embed and unembed stay fixed so every study is comparable.</p>
        </div>

        <div className="builder-grid">
          <aside className="panel palette-panel">
            <div className="panel-title"><span>Component library</span><small>2 blocks</small></div>
            <p className="panel-note">Drag or click to replace the residual stack.</p>
            {(["gelu", "relu"] as Activation[]).map((item) => (
              <button
                className={`palette-card ${activation === item ? "active" : ""}`}
                key={item}
                draggable={!studyLocked}
                disabled={studyLocked}
                onDragStart={(event) => event.dataTransfer.setData("application/x-autoscaler-block", item)}
                onDragEnd={() => setDraggingActivation(null)}
                onPointerDown={() => {
                  setDraggingActivation(item);
                  chooseBlock(item);
                }}
                onPointerUp={() => setDraggingActivation(null)}
                onClick={() => chooseBlock(item)}
              >
                <span className="drag-grip">⠿</span>
                <span className={`block-glyph ${item}`}><i /><i /></span>
                <span><strong>Pre-norm MLP</strong><small>{item.toUpperCase()} activation</small></span>
                <b>{activation === item ? "In use" : "Add"}</b>
              </button>
            ))}
            <div className="coming-soon"><span>Later</span><p>Attention, convolution, MoE, AdamW, and Muon remain outside this validated slice.</p></div>
          </aside>

          <div className="panel canvas-panel">
            <div className="panel-title"><span>Model canvas</span><small>Typed linear graph</small></div>
            <div className="canvas-grid" />
            <div className="model-graph">
              <button className={`model-node compact ${selectedNode === "embed" ? "selected" : ""}`} onClick={() => setSelectedNode("embed")}>
                <span className="node-index">01</span><span><small>Input adapter</small><strong>Embed</strong><em>16 → D</em></span><b>Fixed</b>
              </button>
              <div className="connector"><span /></div>
              <button
                className={`model-node residual-node ${selectedNode === "residual" ? "selected" : ""}`}
                onClick={() => setSelectedNode("residual")}
                onDragOver={(event) => event.preventDefault()}
                onPointerEnter={(event) => {
                  if (draggingActivation && event.buttons === 1) chooseBlock(draggingActivation);
                }}
                onPointerUp={() => setDraggingActivation(null)}
                onDrop={(event) => {
                  event.preventDefault();
                  const item = event.dataTransfer.getData("application/x-autoscaler-block");
                  if (item === "relu" || item === "gelu") chooseBlock(item);
                }}
              >
                <span className="node-index">02</span>
                <span><small>Repeatable cell</small><strong>Residual stack</strong><em>Pre-norm MLP · {activation.toUpperCase()}</em></span>
                <div className="repeat-badge">× R</div>
              </button>
              <div className="connector"><span /></div>
              <button className={`model-node compact ${selectedNode === "unembed" ? "selected" : ""}`} onClick={() => setSelectedNode("unembed")}>
                <span className="node-index">03</span><span><small>Output adapter</small><strong>Unembed</strong><em>D → 1</em></span><b>Fixed</b>
              </button>
            </div>
            <div className="canvas-caption"><span /> Shape-safe by construction</div>
          </div>

          <aside className="panel inspector-panel">
            <div className="panel-title"><span>Inspector</span><small>{selectedNode}</small></div>
            {selectedNode === "residual" ? (
              <>
                <div className="field-label">Activation</div>
                <div className="segmented">
                  <button disabled={studyLocked} className={activation === "gelu" ? "active" : ""} onClick={() => chooseBlock("gelu")}>GELU</button>
                  <button disabled={studyLocked} className={activation === "relu" ? "active" : ""} onClick={() => chooseBlock("relu")}>ReLU</button>
                </div>
                <div className="field-label">Residual parameterization</div>
                <div className="readonly-field"><span>Branch multiplier</span><strong>1 / √R</strong></div>
                <div className="readonly-field"><span>Normalization</span><strong>Pre-norm</strong></div>
                <div className="formula-card"><small>Cell definition</small><code>x ← x + f(LN(x)) / √R</code></div>
              </>
            ) : (
              <div className="locked-inspector"><span>Locked</span><h3>{selectedNode === "embed" ? "Dataset embedding" : "Validation head"}</h3><p>This adapter is inferred from the task and held constant across scale levels.</p></div>
            )}
          </aside>
        </div>
      </section>

      <section className="study-section section-shell" id="study">
        <div className="section-heading">
          <div><p className="eyebrow">Study design</p><h2>One horizon, five scales</h2></div>
          <p>Only width and repeat count grow. The dataset and training horizon remain fixed in this product slice.</p>
        </div>

        <div className="study-grid">
          <div className="panel scale-panel">
            <div className="panel-title"><span>Scale ladder</span><small>Largest level held out</small></div>
            <div className="scale-table" role="table" aria-label="Model scale ladder">
              <div className="scale-row table-head" role="row"><span>Level</span><span>Width D</span><span>Repeats R</span><span>Parameters</span><span>Role</span></div>
              {scales.map((scale, index) => (
                <div className="scale-row" role="row" key={scale.name}>
                  <strong>{scale.name}</strong>
                  <input disabled={studyLocked} aria-label={`${scale.name} width`} type="number" min="4" step="4" value={scale.width} onChange={(event) => updateScale(index, "width", Number(event.target.value))} />
                  <input disabled={studyLocked} aria-label={`${scale.name} repeats`} type="number" min="1" value={scale.repeats} onChange={(event) => updateScale(index, "repeats", Number(event.target.value))} />
                  <span>{formatNumber(parameterCount(scale))}</span>
                  <span className={index === scales.length - 1 ? "role holdout" : "role"}>{index === scales.length - 1 ? "Holdout" : "Fit"}</span>
                </div>
              ))}
            </div>
            {!planValid && <p className="validation-error">Each level must have strictly more parameters than the previous one.</p>}
          </div>

          <aside className="panel protocol-panel">
            <div className="panel-title"><span>Training protocol</span><small>Immutable after launch</small></div>
            <div className="field-label">Optimizer</div>
            <div className="optimizer-choice">
              <button disabled={studyLocked} className={optimizer === "sgd" ? "active" : ""} onClick={() => { if (!studyLocked) { setOptimizer("sgd"); setJob(null); } }}><strong>SGD</strong><small>Momentum 0</small></button>
              <button disabled={studyLocked} className={optimizer === "adam" ? "active" : ""} onClick={() => { if (!studyLocked) { setOptimizer("adam"); setJob(null); } }}><strong>Adam</strong><small>β₁ .9 · β₂ .999</small></button>
            </div>
            <div className="paired-inputs">
              <label><span>Training points</span><input disabled={studyLocked} type="number" min="128" step="128" value={datasetSize} onChange={(event) => { if (!studyLocked) { setDatasetSize(Math.max(128, Number(event.target.value))); setJob(null); } }} /></label>
              <label><span>Update steps</span><input disabled={studyLocked} type="number" min="10" step="10" value={steps} onChange={(event) => { if (!studyLocked) { setSteps(Math.max(10, Number(event.target.value))); setJob(null); } }} /></label>
            </div>
            <div className="fixed-callout"><b>Fixed across every scale</b><span>{formatNumber(datasetSize)} points · {formatNumber(steps * 64)} sample presentations</span></div>
            <div className="seed-row"><span>Common random seeds</span><strong>11</strong><strong>29</strong></div>
          </aside>
        </div>

        <div className="review-bar">
          <div><span className="review-check">✓</span><p><strong>Schema valid</strong><small>Embed → residual → unembed</small></p></div>
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
            <div><p className="eyebrow">No study yet</p><h3>Your held-out result lands here.</h3><p>Launch the immutable plan above. The system will tune the reference model, transfer its learning rate, fit only the smaller levels, then reveal the largest model.</p></div>
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
                <div><small>Selected learning rate</small><strong>{result.tuning.selected_learning_rate.toExponential(1)}</strong><span>{result.tuning.optimum_is_interior ? "Interior optimum" : "Boundary warning"}</span></div>
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
