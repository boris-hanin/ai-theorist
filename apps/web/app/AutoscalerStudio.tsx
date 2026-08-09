"use client";

import { useEffect, useMemo, useState } from "react";

type Optimizer = "sgd" | "adam";
type Activation = "relu" | "gelu" | "silu";
type BlockType = "pre_norm_mlp" | "pre_norm_moe" | "normalized_transformer";
type RunProfile = "smoke" | "pilot" | "a100" | "custom";
type Difficulty = "easy" | "scaling" | "stress" | "custom";
type DatasetTask = "nonlinear_regression" | "synthetic_markov";
type ScalePath = "width" | "depth" | "joint" | "moe_lmd";
type DataScalingMode = "fixed" | "geometric";
type HorizonPolicy = "fixed_updates" | "constant_epochs";
type Scale = { name: string; width: number; repeats: number; expert_width?: number };

type StudySpec = {
  schema_version: 2;
  name: string;
  run_profile: RunProfile;
  architecture: {
    block_type: BlockType;
    activation: Activation;
    input_dim: number;
    output_dim: number;
    residual_multiplier: number;
    num_experts?: number;
    active_experts?: number;
    router_balance_rate?: number;
    vocab_size?: number;
    context_length?: number;
    head_dimension?: number;
    mlp_multiplier?: number;
    reference_width?: number;
    reference_depth?: number;
  };
  optimizer: { name: Optimizer; beta1?: number; beta2?: number; epsilon?: number };
  dataset: {
    task_type: DatasetTask;
    difficulty: Difficulty;
    n_train: number;
    n_validation: number;
    noise_std: number;
    seed: number;
    teacher_width: number;
    teacher_depth: number;
    markov_order: number;
    markov_states: number;
  };
  horizon: { steps: number; batch_size: number; microbatch_size: number | null };
  data_scaling: { mode: DataScalingMode; growth_factor: number; horizon_policy: HorizonPolicy };
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
  n_train: number;
  steps: number;
  token_horizon: number;
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
  normalization_quality: {
    applicable: boolean;
    accepted: boolean;
    maximum_norm_error_tolerance: number;
    scales: { scale: string; maximum_matrix_norm_error: number; maximum_hidden_norm_error: number; mean_attention_entropy: number }[];
  };
  pilot_readiness: {
    ready: boolean;
    dynamic_range_to_noise: number;
    monotone_transition_fraction: number;
    parameter_span_ratio: number;
    compute_span_ratio: number;
    reasons: string[];
    recommendations: string[];
    suggested_next_scale: { width: number; repeats: number; expert_width?: number };
    suggested_next_training_points: number;
  };
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
function generateScaleLadder({
  blockType,
  path,
  count,
  startWidth,
  startDepth,
  startExpertWidth,
  widthGrowth,
  depthGrowth,
  headDimension,
}: {
  blockType: BlockType;
  path: ScalePath;
  count: number;
  startWidth: number;
  startDepth: number;
  startExpertWidth: number;
  widthGrowth: number;
  depthGrowth: number;
  headDimension: number;
}): Scale[] {
  const widthMultiple = blockType === "normalized_transformer" ? headDimension : 2;
  const lmdInvariant = startDepth * startExpertWidth / startWidth;
  const levels: Scale[] = [];
  for (let index = 0; index < count; index += 1) {
    const growsWidth = path !== "depth";
    const growsDepth = path !== "width";
    const rawWidth = startWidth * (growsWidth ? widthGrowth ** index : 1);
    const rawDepth = startDepth * (growsDepth ? depthGrowth ** index : 1);
    let width = Math.max(4, Math.round(rawWidth / widthMultiple) * widthMultiple);
    let repeats = Math.max(1, Math.round(rawDepth));
    const previous = levels.at(-1);
    if (previous && growsWidth && width <= previous.width) width = previous.width + widthMultiple;
    if (previous && growsDepth && repeats <= previous.repeats) repeats = previous.repeats + 1;
    const level: Scale = { name: `S${index + 1}`, width, repeats };
    if (blockType === "pre_norm_moe") {
      level.expert_width = path === "moe_lmd"
        ? Math.max(2, Math.round(lmdInvariant * width / repeats))
        : Math.max(2, Math.round(startExpertWidth * widthGrowth ** index));
    }
    levels.push(level);
  }
  return levels;
}

function parameterCount(scale: Scale, blockType: BlockType, numExperts: number, vocabSize: number, mlpMultiplier: number) {
  const width = scale.width;
  if (blockType === "normalized_transformer") {
    const mlpWidth = mlpMultiplier * width;
    return 2 * vocabSize * width
      + scale.repeats * (4 * width * width + 3 * width * mlpWidth + 3 * width + 2 * mlpWidth)
      + vocabSize;
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
  const [runProfile, setRunProfile] = useState<RunProfile>("smoke");
  const [blockType, setBlockType] = useState<BlockType>("pre_norm_mlp");
  const [activation, setActivation] = useState<Activation>("gelu");
  const [numExperts, setNumExperts] = useState(4);
  const [activeExperts, setActiveExperts] = useState(1);
  const [inputDimension, setInputDimension] = useState(16);
  const [vocabSize, setVocabSize] = useState(32);
  const [contextLength, setContextLength] = useState(16);
  const [headDimension, setHeadDimension] = useState(8);
  const [mlpMultiplier, setMlpMultiplier] = useState(4);
  const [optimizer, setOptimizer] = useState<Optimizer>("adam");
  const [scalePath, setScalePath] = useState<ScalePath>("joint");
  const [scaleCount, setScaleCount] = useState(5);
  const [startWidth, setStartWidth] = useState(16);
  const [startDepth, setStartDepth] = useState(1);
  const [startExpertWidth, setStartExpertWidth] = useState(16);
  const [widthGrowth, setWidthGrowth] = useState(1.5);
  const [depthGrowth, setDepthGrowth] = useState(1.4);
  const [manualScales, setManualScales] = useState<Scale[] | null>(null);
  const [steps, setSteps] = useState(40);
  const [datasetSize, setDatasetSize] = useState(512);
  const [validationSize, setValidationSize] = useState(256);
  const [difficulty, setDifficulty] = useState<Difficulty>("easy");
  const [noiseStd, setNoiseStd] = useState(0.03);
  const [teacherWidth, setTeacherWidth] = useState(32);
  const [teacherDepth, setTeacherDepth] = useState(2);
  const [markovOrder, setMarkovOrder] = useState(2);
  const [markovStates, setMarkovStates] = useState(4);
  const [batchSize, setBatchSize] = useState(64);
  const [microbatchSize, setMicrobatchSize] = useState<number | null>(null);
  const [dataScalingMode, setDataScalingMode] = useState<DataScalingMode>("fixed");
  const [dataGrowthFactor, setDataGrowthFactor] = useState(2);
  const [horizonPolicy, setHorizonPolicy] = useState<HorizonPolicy>("fixed_updates");
  const [targetDevice, setTargetDevice] = useState<"cpu" | "cuda">("cpu");
  const [selectedNode, setSelectedNode] = useState<"embed" | "residual" | "unembed">("residual");
  const [apiOnline, setApiOnline] = useState<boolean | null>(null);
  const [job, setJob] = useState<StudyJob | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [draggingBlock, setDraggingBlock] = useState<BlockType | null>(null);

  const generatedScales = useMemo(() => generateScaleLadder({
    blockType,
    path: blockType === "pre_norm_moe" ? "moe_lmd" : scalePath,
    count: scaleCount,
    startWidth,
    startDepth,
    startExpertWidth,
    widthGrowth,
    depthGrowth,
    headDimension,
  }), [blockType, scalePath, scaleCount, startWidth, startDepth, startExpertWidth, widthGrowth, depthGrowth, headDimension]);
  const scales = manualScales ?? generatedScales;
  const datasetTask: DatasetTask = blockType === "normalized_transformer"
    ? "synthetic_markov"
    : "nonlinear_regression";
  const presentationsPerStep = batchSize * (
    blockType === "normalized_transformer" ? contextLength : 1
  );
  const tokenBudget = steps * presentationsPerStep;

  const effectiveProtocol = (index: number) => {
    const growth = dataScalingMode === "geometric" ? dataGrowthFactor ** index : 1;
    const nTrain = Math.max(8, Math.round(datasetSize * growth));
    const levelSteps = horizonPolicy === "constant_epochs"
      ? Math.max(1, Math.ceil(steps * nTrain / datasetSize))
      : steps;
    return { nTrain, steps: levelSteps, tokenBudget: levelSteps * presentationsPerStep };
  };

  const learningRates = useMemo(
    () => optimizer === "adam"
      ? blockType === "pre_norm_moe" ? [0.03, 0.1, 0.3, 1, 3]
        : blockType === "normalized_transformer" ? [0.0003, 0.001, 0.003, 0.01, 0.03]
        : [0.0001, 0.0003, 0.001, 0.003, 0.01]
      : [0.02, 0.06, 0.2, 0.6, 2.0],
    [blockType, optimizer],
  );

  const spec = useMemo<StudySpec>(() => ({
    schema_version: 2,
    name: `${optimizer}-${blockType === "normalized_transformer" ? "nugpt" : blockType === "pre_norm_moe" ? "moe" : "mlp"}-${runProfile}`,
    run_profile: runProfile,
    architecture: {
      block_type: blockType,
      activation,
      input_dim: inputDimension,
      output_dim: 1,
      residual_multiplier: 1,
      ...(blockType === "pre_norm_moe" ? {
        num_experts: numExperts,
        active_experts: activeExperts,
        router_balance_rate: 0.1,
      } : blockType === "normalized_transformer" ? {
        vocab_size: vocabSize,
        context_length: contextLength,
        head_dimension: headDimension,
        mlp_multiplier: mlpMultiplier,
        reference_width: scales[Math.floor((scales.length - 1) / 2)].width,
        reference_depth: scales[Math.floor((scales.length - 1) / 2)].repeats,
      } : {}),
    },
    optimizer: blockType === "normalized_transformer"
      ? { name: "adam", beta1: 0.9, beta2: 0.95, epsilon: 1e-16 }
      : { name: optimizer },
    dataset: {
      task_type: datasetTask,
      difficulty,
      n_train: datasetSize,
      n_validation: validationSize,
      noise_std: noiseStd,
      seed: 1729,
      teacher_width: teacherWidth,
      teacher_depth: teacherDepth,
      markov_order: markovOrder,
      markov_states: markovStates,
    },
    horizon: { steps, batch_size: batchSize, microbatch_size: microbatchSize },
    data_scaling: {
      mode: dataScalingMode,
      growth_factor: dataScalingMode === "fixed" ? 1 : dataGrowthFactor,
      horizon_policy: horizonPolicy,
    },
    scales,
    tuning: {
      normalized_learning_rates: learningRates,
      max_expansion_rounds: 1,
      expansion_factor: 3,
    },
    validation: { transfer_probe_decades: 0.3, run_negative_control: true, bootstrap_samples: 200, routing_load_tolerance: 0.25 },
    seeds: [11, 29],
    holdout_count: 1,
  }), [activation, activeExperts, batchSize, blockType, contextLength, dataGrowthFactor, dataScalingMode, datasetSize, datasetTask, difficulty, headDimension, horizonPolicy, inputDimension, learningRates, markovOrder, markovStates, microbatchSize, mlpMultiplier, noiseStd, numExperts, optimizer, runProfile, scales, steps, teacherDepth, teacherWidth, validationSize, vocabSize]);

  // Reference tuning and the center holdout probe are reused from cache.
  const estimatedTrials = learningRates.length * 2 + (scales.length - 1) * 2 + 2 * 2 + 2;
  const totalParameters = parameterCount(scales[scales.length - 1], blockType, numExperts, vocabSize, mlpMultiplier);
  const lmOverD = scales.map((scale) => scale.repeats * (scale.expert_width ?? 0) / scale.width);
  const moeInvariant = blockType !== "pre_norm_moe"
    || Math.max(...lmOverD) - Math.min(...lmOverD) < 1e-9;
  const planValid = scales.length >= 5 && scales.every((scale, index) => (
    scale.width >= 4
    && scale.repeats >= 1
    && (blockType !== "pre_norm_moe" || (scale.expert_width ?? 0) >= 2)
    && (blockType !== "normalized_transformer" || (headDimension % 2 === 0 && scale.width % headDimension === 0))
    && (index === 0 || parameterCount(scale, blockType, numExperts, vocabSize, mlpMultiplier) > parameterCount(scales[index - 1], blockType, numExperts, vocabSize, mlpMultiplier))
  ))
    && activeExperts >= 1
    && activeExperts <= numExperts
    && batchSize <= datasetSize
    && (!microbatchSize || (microbatchSize <= batchSize && batchSize % microbatchSize === 0))
    && (blockType !== "normalized_transformer" || markovOrder < contextLength);
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

  function markProfileEdited() {
    setRunProfile("custom");
    setJob(null);
  }

  function updateScale(index: number, field: "width" | "repeats" | "expert_width", value: number) {
    if (studyLocked) return;
    setManualScales((current) => (current ?? generatedScales).map((scale, scaleIndex) => (
      scaleIndex === index ? { ...scale, [field]: Math.max(field === "width" ? 4 : field === "expert_width" ? 2 : 1, value || 1) } : scale
    )));
    setRunProfile("custom");
    setJob(null);
  }

  function applyDifficulty(next: Difficulty, preserveProfile = false) {
    if (studyLocked) return;
    setDifficulty(next);
    if (next === "easy") {
      setNoiseStd(0.03);
      setTeacherWidth(32);
      setTeacherDepth(2);
      setMarkovOrder(2);
      setMarkovStates(4);
    } else if (next === "scaling") {
      setNoiseStd(0.01);
      setTeacherWidth(128);
      setTeacherDepth(4);
      setMarkovOrder(3);
      setMarkovStates(16);
      setContextLength((value) => Math.max(value, 32));
    } else if (next === "stress") {
      setNoiseStd(0);
      setTeacherWidth(256);
      setTeacherDepth(6);
      setMarkovOrder(5);
      setMarkovStates(64);
      setContextLength((value) => Math.max(value, 64));
    }
    if (preserveProfile) setJob(null);
    else markProfileEdited();
  }

  function applyRunProfile(next: RunProfile, architecture: BlockType = blockType) {
    if (studyLocked) return;
    setRunProfile(next);
    setManualScales(null);
    if (next === "custom") {
      setJob(null);
      return;
    }
    const isTransformer = architecture === "normalized_transformer";
    const isMoe = architecture === "pre_norm_moe";
    setScalePath(isMoe ? "moe_lmd" : "joint");
    setDataScalingMode("fixed");
    setDataGrowthFactor(2);
    setHorizonPolicy("fixed_updates");
    if (next === "smoke") {
      setScaleCount(5);
      setStartWidth(isTransformer ? 16 : isMoe ? 8 : 16);
      setStartDepth(isMoe ? 2 : 1);
      setStartExpertWidth(16);
      setWidthGrowth(isTransformer ? 1.35 : 1.5);
      setDepthGrowth(1.4);
      setDatasetSize(isTransformer ? 256 : 512);
      setValidationSize(isTransformer ? 128 : 256);
      setSteps(isTransformer ? 30 : 40);
      setBatchSize(isTransformer ? 16 : 64);
      setMicrobatchSize(null);
      setTargetDevice("cpu");
      applyDifficulty("easy", true);
      if (isTransformer) {
        setVocabSize(32);
        setContextLength(16);
        setHeadDimension(8);
      }
    } else if (next === "pilot") {
      setScaleCount(6);
      setStartWidth(isTransformer ? 64 : isMoe ? 18 : 32);
      setStartDepth(isTransformer ? 2 : isMoe ? 3 : 2);
      setStartExpertWidth(isMoe ? 24 : 32);
      setWidthGrowth(isTransformer ? 1.3 : 1.5);
      setDepthGrowth(1.35);
      setDatasetSize(4096);
      setValidationSize(1024);
      setSteps(400);
      setBatchSize(isTransformer ? 64 : 128);
      setMicrobatchSize(isTransformer ? 16 : 32);
      setTargetDevice("cpu");
      applyDifficulty("scaling", true);
      if (isTransformer) {
        setVocabSize(128);
        setContextLength(64);
        setHeadDimension(16);
      }
    } else {
      setScaleCount(6);
      setStartWidth(isTransformer ? 128 : isMoe ? 32 : 96);
      setStartDepth(isTransformer ? 4 : isMoe ? 3 : 2);
      setStartExpertWidth(isMoe ? 48 : 64);
      setWidthGrowth(isTransformer ? 1.25 : 1.5);
      setDepthGrowth(1.35);
      setDatasetSize(16_384);
      setValidationSize(4096);
      setSteps(1000);
      setBatchSize(isTransformer ? 64 : 256);
      setMicrobatchSize(isTransformer ? 16 : 64);
      setTargetDevice("cuda");
      applyDifficulty("stress", true);
      if (isTransformer) {
        setVocabSize(256);
        setContextLength(128);
        setHeadDimension(64);
      }
    }
    setJob(null);
  }

  function chooseBlock(next: BlockType) {
    if (studyLocked) return;
    setBlockType(next);
    setManualScales(null);
    applyRunProfile(runProfile, next);
    if (next === "pre_norm_moe" || next === "normalized_transformer") setOptimizer("adam");
    setActivation(next === "normalized_transformer" ? "silu" : "gelu");
    setSelectedNode("residual");
    setJob(null);
  }

  function chooseDatasetTask(next: DatasetTask) {
    if (next === "synthetic_markov") chooseBlock("normalized_transformer");
    else if (blockType === "normalized_transformer") chooseBlock("pre_norm_mlp");
  }

  function chooseActivation(next: Activation) {
    if (studyLocked) return;
    setActivation(next);
    markProfileEdited();
  }

  async function startStudy() {
    if (!planValid || studyLocked) return;
    setError(null);
    try {
      const response = await fetch(`${API_BASE}/api/studies`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ spec, device: targetDevice }),
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
          <p className="eyebrow">Explicit-budget scaling workbench</p>
          <h1>Build the model. <span>Measure the law.</span></h1>
          <p className="hero-copy">Compose a typed model, tune one scale-normalized learning rate, and earn a largest-model forecast through held-out calibration.</p>
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
            <div className="panel-title"><span>Component library</span><small>3 blocks</small></div>
            <p className="panel-note">Drag or click to replace the residual stack.</p>
            {(["normalized_transformer", "pre_norm_mlp", "pre_norm_moe"] as BlockType[]).map((item) => (
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
                <span className={`block-glyph ${item === "pre_norm_moe" ? "moe" : item === "normalized_transformer" ? "attention" : "mlp"}`}><i /><i /></span>
                <span>
                  <strong>{item === "pre_norm_moe" ? "Sparse MoE" : item === "normalized_transformer" ? "νGPT" : "Pre-norm MLP"}</strong>
                  <small>{item === "pre_norm_moe" ? "Top-k expert routing" : item === "normalized_transformer" ? "Normalized Transformer with LR transfer" : "Dense residual cell"}</small>
                </span>
                <b>{blockType === item ? "In use" : "Add"}</b>
              </button>
            ))}
            <div className="coming-soon"><span>Later</span><p>Automatic batch-size scaling, the 2026 nGPT recipe, convolution, AdamW, Muon, and general DAGs remain outside this validated slice.</p></div>
          </aside>

          <div className="panel canvas-panel">
            <div className="panel-title"><span>Model canvas</span><small>Typed linear graph</small></div>
            <div className="canvas-grid" />
            <div className="model-graph">
              <button className={`model-node compact ${selectedNode === "embed" ? "selected" : ""}`} onClick={() => setSelectedNode("embed")}>
                <span className="node-index">01</span><span><small>Input adapter</small><strong>Embed</strong><em>{blockType === "normalized_transformer" ? `token → ${vocabSize} × D` : `${inputDimension} → D`}</em></span><b>Fixed</b>
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
                  if (item === "pre_norm_mlp" || item === "pre_norm_moe" || item === "normalized_transformer") chooseBlock(item);
                }}
              >
                <span className="node-index">02</span>
                <span><small>Repeatable cell</small><strong>{blockType === "normalized_transformer" ? "νGPT stack" : "Residual stack"}</strong><em>{blockType === "pre_norm_moe" ? `Sparse MoE · ${numExperts} experts · top ${activeExperts}` : blockType === "normalized_transformer" ? `Causal attention · head dim ${headDimension} · SwiGLU` : "Pre-norm MLP"}</em></span>
                <div className="repeat-badge">× R</div>
              </button>
              <div className="connector"><span /></div>
              <button className={`model-node compact ${selectedNode === "unembed" ? "selected" : ""}`} onClick={() => setSelectedNode("unembed")}>
                <span className="node-index">03</span><span><small>Output adapter</small><strong>Unembed</strong><em>{blockType === "normalized_transformer" ? `D → ${vocabSize} logits` : "D → 1"}</em></span><b>Fixed</b>
              </button>
            </div>
            <div className="canvas-caption"><span /> Shape-safe by construction</div>
          </div>

          <aside className="panel inspector-panel">
            <div className="panel-title"><span>Inspector</span><small>{selectedNode}</small></div>
            {selectedNode === "residual" ? (
              <>
                <div className="field-label">Activation</div>
                {blockType === "normalized_transformer" ? (
                  <div className="segmented"><button disabled className="active">SiLU / SwiGLU</button></div>
                ) : (
                  <div className="segmented">
                    <button disabled={studyLocked} className={activation === "gelu" ? "active" : ""} onClick={() => chooseActivation("gelu")}>GELU</button>
                    <button disabled={studyLocked} className={activation === "relu" ? "active" : ""} onClick={() => chooseActivation("relu")}>ReLU</button>
                  </div>
                )}
                <div className="field-label">Residual parameterization</div>
                {blockType === "normalized_transformer" ? (
                  <>
                    <div className="paired-inputs inspector-inputs">
                      <label><span>Vocabulary</span><input disabled={studyLocked} type="number" min="8" step="8" value={vocabSize} onChange={(event) => { setVocabSize(Math.max(8, Number(event.target.value))); markProfileEdited(); }} /></label>
                      <label><span>Context</span><input disabled={studyLocked} type="number" min="2" step="2" value={contextLength} onChange={(event) => { setContextLength(Math.max(2, Number(event.target.value))); markProfileEdited(); }} /></label>
                      <label><span>Head dimension</span><input disabled={studyLocked} type="number" min="2" step="2" value={headDimension} onChange={(event) => { setHeadDimension(Math.max(2, Number(event.target.value))); markProfileEdited(); }} /></label>
                      <label><span>MLP multiple</span><input disabled={studyLocked} type="number" min="1" value={mlpMultiplier} onChange={(event) => { setMlpMultiplier(Math.max(1, Number(event.target.value))); markProfileEdited(); }} /></label>
                    </div>
                    <div className="readonly-field"><span>Hidden geometry</span><strong>Unit sphere</strong></div>
                    <div className="readonly-field"><span>Weight projection</span><strong>After every step</strong></div>
                    <div className="formula-card"><small>νGPT transfer</small><code>input η·mD⁻¹ᐟ² · hidden η·mD⁻³ᐟ⁴ · α₀ 0.05·mL⁻¹</code></div>
                  </>
                ) : blockType === "pre_norm_moe" ? (
                  <>
                    <div className="paired-inputs inspector-inputs">
                      <label><span>Stored experts E</span><input disabled={studyLocked} type="number" min="1" value={numExperts} onChange={(event) => { const value = Math.max(1, Number(event.target.value)); setNumExperts(value); setActiveExperts((current) => Math.min(current, value)); markProfileEdited(); }} /></label>
                      <label><span>Active experts a</span><input disabled={studyLocked} type="number" min="1" max={numExperts} value={activeExperts} onChange={(event) => { setActiveExperts(Math.min(numExperts, Math.max(1, Number(event.target.value)))); markProfileEdited(); }} /></label>
                    </div>
                    <div className="readonly-field"><span>Active fraction κ</span><strong>{activeExperts}/{numExperts}</strong></div>
                    <div className="readonly-field"><span>Branch multiplier</span><strong>1 / L</strong></div>
                    <div className="formula-card"><small>Transfer contract</small><code>keep LM/D fixed · router/up η/D · down η/M</code></div>
                  </>
                ) : (
                  <>
                    <div className="readonly-field"><span>Branch multiplier</span><strong>1 / R</strong></div>
                    <div className="formula-card"><small>Cell definition</small><code>x ← x + f(LN(x)) / R</code></div>
                  </>
                )}
                <div className="readonly-field"><span>Normalization</span><strong>{blockType === "normalized_transformer" ? "Hyperspherical · no norm layers" : "Pre-norm"}</strong></div>
              </>
            ) : (
              <div className="locked-inspector"><span>Locked</span><h3>{selectedNode === "embed" ? "Dataset embedding" : "Validation head"}</h3><p>This adapter is inferred from the task and held constant across scale levels.</p></div>
            )}
          </aside>
        </div>
      </section>

      <section className="study-section section-shell" id="study">
        <div className="section-heading">
          <div><p className="eyebrow">Study design</p><h2>Choose the regime. Generate the ladder.</h2></div>
          <p>Data complexity, training budget, and model scale are explicit. Keep data fixed for clean model scaling, or declare a geometric joint model–data study.</p>
        </div>

        <div className="profile-grid" aria-label="Run profile">
          {(["smoke", "pilot", "a100", "custom"] as RunProfile[]).map((profile) => (
            <button disabled={studyLocked} key={profile} className={`profile-card ${runProfile === profile ? "active" : ""}`} onClick={() => applyRunProfile(profile)}>
              <strong>{profile === "a100" ? "A100 study" : profile[0].toUpperCase() + profile.slice(1)}</strong>
              <span>{profile === "smoke" ? "Plumbing only · forecasts disabled" : profile === "pilot" ? "Find transfer and loss range" : profile === "a100" ? "Forecast-grade CUDA campaign" : "Fully editable contract"}</span>
            </button>
          ))}
        </div>

        <div className="setup-grid">
          <div className="panel data-panel">
            <div className="panel-title"><span>Dataset</span><small>Fixed held-out stream</small></div>
            <div className="field-label">Task template</div>
            <div className="segmented data-task-choice">
              <button disabled={studyLocked} className={datasetTask === "nonlinear_regression" ? "active" : ""} onClick={() => chooseDatasetTask("nonlinear_regression")}>Teacher regression</button>
              <button disabled={studyLocked} className={datasetTask === "synthetic_markov" ? "active" : ""} onClick={() => chooseDatasetTask("synthetic_markov")}>Markov language</button>
            </div>
            <div className="field-label">Difficulty</div>
            <div className="difficulty-choice">
              {(["easy", "scaling", "stress"] as Difficulty[]).map((level) => <button disabled={studyLocked} key={level} className={difficulty === level ? "active" : ""} onClick={() => applyDifficulty(level)}>{level}</button>)}
            </div>
            <div className="paired-inputs dense-inputs">
              {datasetTask === "nonlinear_regression" ? (
                <>
                  <label><span>Input dimension</span><input disabled={studyLocked} type="number" min="2" value={inputDimension} onChange={(event) => { setInputDimension(Math.max(2, Number(event.target.value))); setDifficulty("custom"); markProfileEdited(); }} /></label>
                  <label><span>Teacher width</span><input disabled={studyLocked} type="number" min="4" value={teacherWidth} onChange={(event) => { setTeacherWidth(Math.max(4, Number(event.target.value))); setDifficulty("custom"); markProfileEdited(); }} /></label>
                  <label><span>Teacher depth</span><input disabled={studyLocked} type="number" min="1" value={teacherDepth} onChange={(event) => { setTeacherDepth(Math.max(1, Number(event.target.value))); setDifficulty("custom"); markProfileEdited(); }} /></label>
                </>
              ) : (
                <>
                  <label><span>Markov order</span><input disabled={studyLocked} type="number" min="1" max={contextLength - 1} value={markovOrder} onChange={(event) => { setMarkovOrder(Math.max(1, Math.min(contextLength - 1, Number(event.target.value)))); setDifficulty("custom"); markProfileEdited(); }} /></label>
                  <label><span>Latent states</span><input disabled={studyLocked} type="number" min="2" value={markovStates} onChange={(event) => { setMarkovStates(Math.max(2, Number(event.target.value))); setDifficulty("custom"); markProfileEdited(); }} /></label>
                  <label><span>Vocabulary</span><input disabled={studyLocked} type="number" min="8" step="8" value={vocabSize} onChange={(event) => { setVocabSize(Math.max(8, Number(event.target.value))); setDifficulty("custom"); markProfileEdited(); }} /></label>
                  <label><span>Context</span><input disabled={studyLocked} type="number" min={markovOrder + 1} value={contextLength} onChange={(event) => { setContextLength(Math.max(markovOrder + 1, Number(event.target.value))); setDifficulty("custom"); markProfileEdited(); }} /></label>
                </>
              )}
              <label><span>{datasetTask === "synthetic_markov" ? "Random-token probability" : "Label noise"}</span><input disabled={studyLocked} type="number" min="0" max={datasetTask === "synthetic_markov" ? 1 : undefined} step="0.01" value={noiseStd} onChange={(event) => { setNoiseStd(Math.max(0, Number(event.target.value))); setDifficulty("custom"); markProfileEdited(); }} /></label>
            </div>
            <p className="data-note">Difficulty increases teacher structure or sequence memory. Noise remains separate because raising the irreducible floor can hide a real scaling law.</p>
          </div>

          <aside className="panel protocol-panel">
            <div className="panel-title"><span>Training budget</span><small>{targetDevice === "cuda" ? "CUDA runner" : "Local CPU"}</small></div>
            <div className="field-label">Optimizer</div>
            <div className="optimizer-choice">
              <button disabled={studyLocked || blockType === "pre_norm_moe" || blockType === "normalized_transformer"} className={optimizer === "sgd" ? "active" : ""} onClick={() => { setOptimizer("sgd"); markProfileEdited(); }}><strong>SGD</strong><small>Momentum 0</small></button>
              <button disabled={studyLocked} className={optimizer === "adam" ? "active" : ""} onClick={() => { setOptimizer("adam"); markProfileEdited(); }}><strong>Adam</strong><small>{blockType === "normalized_transformer" ? "β₂ .95" : "β₂ .999"}</small></button>
            </div>
            <div className="paired-inputs dense-inputs">
              <label><span>Base training {datasetTask === "synthetic_markov" ? "sequences" : "points"}</span><input disabled={studyLocked} type="number" min={batchSize} value={datasetSize} onChange={(event) => { setDatasetSize(Math.max(batchSize, Number(event.target.value))); markProfileEdited(); }} /></label>
              <label><span>Validation size</span><input disabled={studyLocked} type="number" min="8" value={validationSize} onChange={(event) => { setValidationSize(Math.max(8, Number(event.target.value))); markProfileEdited(); }} /></label>
              <label><span>Batch size</span><input disabled={studyLocked} type="number" min="1" value={batchSize} onChange={(event) => { const value = Math.max(1, Number(event.target.value)); setBatchSize(value); setMicrobatchSize((current) => current ? Math.min(current, value) : null); markProfileEdited(); }} /></label>
              <label><span>Microbatch</span><input disabled={studyLocked} type="number" min="1" max={batchSize} value={microbatchSize ?? batchSize} onChange={(event) => { setMicrobatchSize(Math.max(1, Math.min(batchSize, Number(event.target.value)))); markProfileEdited(); }} /></label>
              <label><span>Base update steps</span><input disabled={studyLocked} type="number" min="1" value={steps} onChange={(event) => { setSteps(Math.max(1, Number(event.target.value))); markProfileEdited(); }} /></label>
              <label><span>{datasetTask === "synthetic_markov" ? "Token budget" : "Sample budget"}</span><input disabled={studyLocked} type="number" min={presentationsPerStep} value={tokenBudget} onChange={(event) => { setSteps(Math.max(1, Math.ceil(Number(event.target.value) / presentationsPerStep))); markProfileEdited(); }} /></label>
            </div>
            <div className="fixed-callout"><b>{runProfile === "smoke" ? "Smoke validation only" : `${runProfile.toUpperCase()} profile`}</b><span>{formatNumber(tokenBudget)} base {datasetTask === "synthetic_markov" ? "tokens" : "sample presentations"} · batch {batchSize} · microbatch {microbatchSize ?? batchSize}</span></div>
          </aside>
        </div>

        <div className="panel joint-panel">
          <div><strong>Data across model scales</strong><span>Validation examples always remain fixed and identical.</span></div>
          <div className="joint-choice">
            <button disabled={studyLocked} className={dataScalingMode === "fixed" ? "active" : ""} onClick={() => { setDataScalingMode("fixed"); setHorizonPolicy("fixed_updates"); markProfileEdited(); }}>Fixed data</button>
            <button disabled={studyLocked} className={dataScalingMode === "geometric" ? "active" : ""} onClick={() => { setDataScalingMode("geometric"); setHorizonPolicy("constant_epochs"); markProfileEdited(); }}>Joint model + data</button>
          </div>
          {dataScalingMode === "geometric" && <div className="joint-fields"><label><span>Data growth / level</span><input disabled={studyLocked} type="number" min="1.05" step="0.05" value={dataGrowthFactor} onChange={(event) => { setDataGrowthFactor(Math.max(1.05, Number(event.target.value))); markProfileEdited(); }} /></label><label><span>Horizon policy</span><select disabled={studyLocked} value={horizonPolicy} onChange={(event) => { setHorizonPolicy(event.target.value as HorizonPolicy); markProfileEdited(); }}><option value="constant_epochs">Match data growth</option><option value="fixed_updates">Fixed updates</option></select></label></div>}
        </div>

        <div className="panel scale-panel generated-scale-panel">
            <div className="panel-title"><span>Automatic scale ladder</span><small>{manualScales ? "Manual override" : `${scaleCount} generated levels · largest held out`}</small></div>
            <div className="ladder-builder">
              <label><span>Scaling path</span><select disabled={studyLocked || blockType === "pre_norm_moe"} value={blockType === "pre_norm_moe" ? "moe_lmd" : scalePath} onChange={(event) => { setScalePath(event.target.value as ScalePath); setManualScales(null); markProfileEdited(); }}><option value="width">Width only</option><option value="depth">Depth only</option><option value="joint">Joint width + depth</option>{blockType === "pre_norm_moe" && <option value="moe_lmd">LM/D constant</option>}</select></label>
              <label><span>Levels</span><input disabled={studyLocked} type="number" min="5" max="10" value={scaleCount} onChange={(event) => { setScaleCount(Math.max(5, Math.min(10, Number(event.target.value)))); setManualScales(null); markProfileEdited(); }} /></label>
              <label><span>Start width</span><input disabled={studyLocked} type="number" min="4" value={startWidth} onChange={(event) => { setStartWidth(Math.max(4, Number(event.target.value))); setManualScales(null); markProfileEdited(); }} /></label>
              <label><span>Start depth</span><input disabled={studyLocked} type="number" min="1" value={startDepth} onChange={(event) => { setStartDepth(Math.max(1, Number(event.target.value))); setManualScales(null); markProfileEdited(); }} /></label>
              <label><span>Width growth</span><input disabled={studyLocked || scalePath === "depth"} type="number" min="1.05" step="0.05" value={widthGrowth} onChange={(event) => { setWidthGrowth(Math.max(1.05, Number(event.target.value))); setManualScales(null); markProfileEdited(); }} /></label>
              <label><span>Depth growth</span><input disabled={studyLocked || scalePath === "width"} type="number" min="1.05" step="0.05" value={depthGrowth} onChange={(event) => { setDepthGrowth(Math.max(1.05, Number(event.target.value))); setManualScales(null); markProfileEdited(); }} /></label>
              {blockType === "pre_norm_moe" && <label><span>Start expert width</span><input disabled={studyLocked} type="number" min="2" value={startExpertWidth} onChange={(event) => { setStartExpertWidth(Math.max(2, Number(event.target.value))); setManualScales(null); markProfileEdited(); }} /></label>}
              {manualScales && <button className="reset-ladder" onClick={() => { setManualScales(null); markProfileEdited(); }}>Return to generated ladder</button>}
            </div>
            <div className="scale-table" role="table" aria-label="Model scale ladder">
              <div className={`scale-row table-head ${blockType !== "pre_norm_mlp" ? "moe-scale-row" : ""}`} role="row"><span>Level</span><span>Width D</span><span>{blockType === "pre_norm_moe" || blockType === "normalized_transformer" ? "Depth L" : "Repeats R"}</span>{blockType === "pre_norm_moe" && <span>Expert M</span>}{blockType === "normalized_transformer" && <span>Heads</span>}<span>Parameters</span><span>Role</span></div>
              {scales.map((scale, index) => (
                <div className={`scale-row ${blockType !== "pre_norm_mlp" ? "moe-scale-row" : ""}`} role="row" key={scale.name}>
                  <strong>{scale.name}</strong>
                  <input disabled={studyLocked} aria-label={`${scale.name} width`} type="number" min="4" step="4" value={scale.width} onChange={(event) => updateScale(index, "width", Number(event.target.value))} />
                  <input disabled={studyLocked} aria-label={`${scale.name} repeats`} type="number" min="1" value={scale.repeats} onChange={(event) => updateScale(index, "repeats", Number(event.target.value))} />
                  {blockType === "pre_norm_moe" && <input disabled={studyLocked} aria-label={`${scale.name} expert width`} type="number" min="2" value={scale.expert_width} onChange={(event) => updateScale(index, "expert_width", Number(event.target.value))} />}
                  {blockType === "normalized_transformer" && <span>{scale.width / headDimension}</span>}
                  <span>{formatNumber(parameterCount(scale, blockType, numExperts, vocabSize, mlpMultiplier))}</span>
                  <span className={index === scales.length - 1 ? "role holdout" : "role"}>{index === scales.length - 1 ? "Holdout" : "Fit"}</span>
                </div>
              ))}
            </div>
            {!planValid && <p className="validation-error">Each level must have strictly more parameters than the previous one.</p>}
            {blockType === "normalized_transformer" && !planValid && <p className="validation-warning">Head dimension must be even and divide every width in the ladder.</p>}
            {blockType === "pre_norm_moe" && !moeInvariant && <p className="validation-warning">For the strongest transfer, keep L × M / D constant across the ladder.</p>}
            <div className="protocol-preview">{scales.map((scale, index) => { const protocol = effectiveProtocol(index); return <div key={scale.name}><strong>{scale.name}</strong><span>{formatNumber(protocol.nTrain)} train</span><span>{formatNumber(protocol.steps)} steps</span><span>{formatNumber(protocol.tokenBudget)} {datasetTask === "synthetic_markov" ? "tokens" : "samples"}</span></div>; })}</div>
        </div>

        <div className="review-bar">
          <div><span className="review-check">✓</span><p><strong>{runProfile === "smoke" ? "Smoke only" : `${runProfile.toUpperCase()} contract`}</strong><small>{runProfile === "smoke" ? "Forecasts deliberately disabled" : `${formatNumber(totalParameters)} largest model`}</small></p></div>
          <div><span className="review-check">✓</span><p><strong>{dataScalingMode === "fixed" ? "Fixed data" : "Joint scaling"}</strong><small>{dataScalingMode === "fixed" ? "Same train set at every scale" : `${dataGrowthFactor}× data per level · ${horizonPolicy === "constant_epochs" ? "matched horizon" : "fixed updates"}`}</small></p></div>
          <div><span className="review-check">✓</span><p><strong>Honest holdout</strong><small>{scales[scales.length - 1].name} excluded from fit · common validation</small></p></div>
          <div className="budget"><small>Planned trials</small><strong>{estimatedTrials}</strong><span>+ edge expansion if needed</span></div>
          <button className="run-button" disabled={!planValid || studyLocked} onClick={startStudy}>
            {job?.status === "running" ? "Study running" : targetDevice === "cuda" ? "Launch on CUDA" : "Lock plan & run"}<span>→</span>
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
            <div><p className="eyebrow">No study yet</p><h3>Your held-out result lands here.</h3><p>Launch the immutable plan above. The system tunes normalized η, checks transfer, scores power-law readiness, and recommends the next model/data regime before issuing any forecast.</p></div>
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
                <div><small>Estimated loss floor</small><strong>{formatLoss(result.scaling_law.loss_floor)}</strong><span>{dataScalingMode === "fixed" ? "Model scaling" : "Joint compute scaling"}</span></div>
              </div>
              <div className={`readiness-card ${result.pilot_readiness.ready ? "ready" : "needs-work"}`}>
                <div><small>Power-law readiness</small><strong>{result.pilot_readiness.ready ? "Ready for the larger campaign" : "Pilot recommends another pass"}</strong></div>
                <div className="readiness-metrics"><span><b>{result.pilot_readiness.parameter_span_ratio.toFixed(1)}×</b> parameter span</span><span><b>{result.pilot_readiness.dynamic_range_to_noise.toFixed(1)}×</b> signal / noise</span><span><b>{Math.round(result.pilot_readiness.monotone_transition_fraction * 100)}%</b> decreasing transitions</span></div>
                {result.pilot_readiness.recommendations.length > 0 && <ul>{result.pilot_readiness.recommendations.map((recommendation) => <li key={recommendation}>{recommendation}</li>)}</ul>}
                <p>Suggested next level: D {result.pilot_readiness.suggested_next_scale.width} · L/R {result.pilot_readiness.suggested_next_scale.repeats} · {formatNumber(result.pilot_readiness.suggested_next_training_points)} training examples</p>
              </div>
            </div>
            <aside className="panel evidence-panel">
              <div className="panel-title"><span>Held-out evidence</span><small>{result.holdout_calibration.length} check</small></div>
              {result.normalization_quality.applicable && (
                <div className="calibration-card">
                  <div><strong>Unit-sphere invariants</strong><span className={result.normalization_quality.accepted ? "accepted" : "rejected"}>{result.normalization_quality.accepted ? "Pass" : "Fail"}</span></div>
                  <p><span>Worst matrix error</span><strong>{Math.max(...result.normalization_quality.scales.map((row) => row.maximum_matrix_norm_error)).toExponential(1)}</strong></p>
                  <p><span>Worst hidden error</span><strong>{Math.max(...result.normalization_quality.scales.map((row) => row.maximum_hidden_norm_error)).toExponential(1)}</strong></p>
                  <p><span>Required ceiling</span><strong>{result.normalization_quality.maximum_norm_error_tolerance.toExponential(0)}</strong></p>
                </div>
              )}
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
