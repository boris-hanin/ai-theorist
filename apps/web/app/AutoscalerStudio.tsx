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
type BatchCampaign = "standard_pretraining_census" | "transformer_census" | "constant_tpp" | "horizon_transfer" | "joint_horizon_batch" | "real_text_scaling_ladder";
type HorizonSchedule = "cosine_to_10_percent" | "linear_warmup_decay_to_zero" | "wsd";
type HorizonParameterization = "jiang_chizat" | "nugpt";
type Precision = "fp32" | "bf16";
type AttentionBackend = "auto" | "math" | "flash";
type DistributedMode = "none" | "ddp" | "fsdp";
type PretrainingOptimizer = "sgd" | "adam" | "adamw";
type TokenizerId = "byte_v1" | "uint16_bin_v1" | "uint32_bin_v1" | "olmo2_1124";
type TokenizerCatalogItem = {
  id: "byte_v1" | "olmo2_1124";
  name: string;
  kind: "builtin" | "pinned_remote";
  vocab_size: number;
  repository?: string;
  revision: string | null;
  definition_fingerprint: string;
  tokenizer_fingerprint: string | null;
  document_separator_token_id: number;
};
type CorpusChoice = "fineweb_edu" | "openwebtext" | "local";
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
  pilot_readiness: null | {
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
  status: "queued" | "running" | "completed" | "failed" | "interrupted";
  device?: "cpu" | "cuda" | string;
  spec?: StudySpec;
  progress: Progress;
  result: StudyResult | null;
  error: string | null;
};
type StudyHistoryItem = {
  id: string;
  status: StudyJob["status"];
  created_at: string | null;
  updated_at: string | null;
  device: string;
  name: string;
  run_profile: RunProfile;
  architecture: BlockType;
  optimizer: string;
  dataset: DatasetTask;
  progress: Progress;
  error: string | null;
  result_summary: null | {
    forecastable: boolean;
    selected_normalized_learning_rate: number;
    scaling_exponent: number;
    r_squared: number;
    holdout_relative_error: number;
    transfer_checks_accepted: boolean | null;
    refusal_reasons: string[];
    trial_count: number | null;
  };
};
type BatchTransferResult = {
  rule: string;
  valid: boolean;
  target: null | {
    name: string;
    learning_rate: number;
    momentum: number;
    beta1: number;
    beta2: number;
    epsilon: number;
    weight_decay: number;
  };
  multipliers: Record<string, number>;
  assumptions: string[];
  refusal_reasons: string[];
};
type CriticalBatchEstimate = {
  critical_batch_tokens: number | null;
  qualified: boolean;
  refusal_reasons: string[];
};
type BatchCampaignAnalysis = {
  scale: { name: string };
  optimizer: string;
  consensus: CriticalBatchEstimate;
};
type BatchCampaignResult = {
  status: string;
  campaign: string;
  records?: unknown[];
  dataset?: {
    tokenizer?: string;
    tokenizer_id?: string;
    fingerprint: string;
    identity_fingerprint?: string;
    tokenizer_fingerprint?: string | null;
    tokenizer_is_pinned?: boolean;
    training_tokens: number;
    validation_tokens: number;
  };
  runtime?: {
    precision: Precision;
    attention_backend: AttentionBackend;
    distributed: DistributedMode;
    num_processes: number;
    device: string;
  };
  scale_optimizer_analyses?: BatchCampaignAnalysis[];
  analyses?: BatchCampaignAnalysis[];
  geometry?: {
    role?: string;
    scale?: { name: string; width: number; repeats: number };
    parameters: number;
    total_tokens?: number;
    unique_tokens?: number;
    presented_tokens?: number;
    optimizer_steps?: number;
    batch_tokens: number;
    realized_tpp?: number;
    tokens_per_parameter?: number;
    presented_to_unique_token_ratio?: number;
  }[];
  tpp_spread_ratio?: number;
  fitted_horizon_exponent?: number;
  fit_qualification?: {
    all_fit_optima_are_interior: boolean;
    source_optimum_is_interior: boolean;
  };
  heldout_scale?: string;
  heldout_oracle?: {
    mean_loss: number;
    learning_rate: number;
    optimum_is_interior: boolean;
  };
  transfer_results?: {
    rule: string;
    valid: boolean;
    evaluated: boolean;
    recommendable?: boolean;
    relative_regret?: number;
    mean_heldout_loss?: number;
    oracle_mean_heldout_loss?: number;
    refusal_reasons: string[];
  }[];
  heldout_horizon?: number;
  fit_horizon_span_ratio?: number;
  schedule_analyses?: {
    schedule_name: string;
    fit_qualified: boolean;
    fit_refusal_reasons: string[];
    fitted_power_law: { exponent: number; r_squared: number };
    mechanism_identifiable: boolean;
    heldout_oracle: { mean_loss: number; learning_rate: number; optimum_is_interior: boolean };
    frozen_rule_results: {
      rule: string;
      exponent: number;
      predicted_peak_learning_rate: number;
      mean_heldout_loss: number;
      relative_oracle_regret: number;
      transfer_certified: boolean;
      mechanism_discrimination_certified: boolean;
    }[];
  }[];
  certified_schedule_rules?: unknown[];
  axis_fit_qualification?: {
    qualified: boolean;
    refusal_reasons: string[];
    horizon_exponent: number;
    batch_exponent: number;
    horizon_fit: { r_squared: number };
    batch_fit: { r_squared: number };
  };
  composition_crosscheck?: {
    presented_tokens: number;
    batch_examples: number;
    batch_tokens: number;
    candidate_results: JointCandidateResult[];
  };
  heldout_corner?: {
    presented_tokens: number;
    batch_examples: number;
    batch_tokens: number;
    composition_identifiable: boolean;
    candidate_results: JointCandidateResult[];
  };
  joint_transfer_settled?: boolean;
  certified_joint_rules?: JointCandidateResult[];
  joint_recommendation?: null | JointCandidateResult;
  recommendation?: null | {
    schedule: string;
    rule: string;
    exponent: number;
    predicted_peak_learning_rate: number;
    mean_heldout_loss: number;
    relative_oracle_regret: number;
    transfer_certified: boolean;
  };
  forecastable?: boolean;
  scales?: {
    name: string;
    parameters: number;
    presented_tokens: number;
    tokens_per_parameter: number;
    repetition_ratio: number;
    heldout: boolean;
    mean_validation_loss: number;
    sem_validation_loss: number;
  }[];
  hidden_scale_backtests?: {
    scale: string;
    parameters: number;
    observed_loss: number;
    predicted_loss: number;
    relative_error: number;
    passed: boolean;
  }[];
  forecasts?: {
    target_size: number;
    prediction: number | null;
    exploratory_prediction: number;
    prediction_interval_95: [number, number] | null;
    extrapolation_factor: number;
    certified: boolean;
    refusal_reasons: string[];
  }[];
  refusal_reasons?: string[];
};
type JointCandidateResult = {
  rule: string;
  valid: boolean;
  evaluated: boolean;
  joint_rule: boolean;
  optimizer?: { learning_rate: number; beta1: number; beta2: number; epsilon: number };
  peak_parameter_group_contract?: { name: string; peak_learning_rate: number; epsilon: number }[];
  mean_loss?: number;
  relative_oracle_regret?: number | null;
  composition_crosscheck_passed?: boolean;
  transfer_certified?: boolean;
  mechanism_discrimination_certified?: boolean;
  theory_assumption_status?: string;
  theory_transfer_certified?: boolean;
  refusal_reasons: string[];
};
type BatchCampaignJob = {
  id: string;
  campaign: BatchCampaign;
  device?: string;
  status: "queued" | "running" | "completed" | "failed" | "interrupted";
  progress: Progress;
  result: BatchCampaignResult | null;
  error: string | null;
  config?: {
    dataset?: { train_path?: string; validation_path?: string; tokenizer?: TokenizerId; token_stream_manifest_path?: string };
    runtime?: { precision?: Precision; attention_backend?: AttentionBackend; distributed?: DistributedMode; num_processes?: number; gradient_accumulation_steps?: number; checkpoint_interval_steps?: number };
    architecture?: { block_type?: string; context_length?: number };
    ladder?: { target_parameters?: number[]; depths?: number[]; tokens_per_parameter?: number; target_forecasts?: number[] };
    batch_examples?: number;
    optimizer?: { learning_rates?: number[] };
    optimizers?: { name?: PretrainingOptimizer }[];
    target_validation_loss?: number;
    validation_interval?: number;
  };
};
type PublicCorpusJob = {
  id: string;
  status: "queued" | "running" | "completed" | "failed" | "interrupted";
  progress: Progress;
  error: string | null;
  result: null | {
    source: { dataset: string; config: string; revision: string; license: string; data_card_url: string };
    tokenizer: "byte_v1" | "olmo2_1124";
    tokenizer_definition_fingerprint: string;
    tokenizer_fingerprint: string;
    tokenizer_manifest_path: string;
    tokenizer_vocab_size: number;
    token_stream_manifest_path: string | null;
    dataset_identity_fingerprint: string;
    corpus_fingerprint: string;
    training_tokens: number;
    validation_tokens: number;
    splits: {
      train: { path: string; documents: number; text_bytes: number; first_source_row: number; last_source_row: number };
      validation: { path: string; documents: number; text_bytes: number; first_source_row: number; last_source_row: number };
    };
  };
};

type BatchHistoryItem = {
  id: string;
  campaign: BatchCampaign;
  status: BatchCampaignJob["status"];
  progress: Progress;
  result_summary?: null | {
    record_count: number;
    qualified_analyses: number;
    analysis_count: number;
    recommendable_rules: number;
    corpus_fingerprint: string | null;
    forecastable?: boolean;
    certified_forecasts?: number;
    forecast_count?: number;
    passed_hidden_backtests?: number;
    hidden_backtest_count?: number;
  };
};

type EvidenceCase = {
  id: string;
  title: string;
  architecture: string;
  optimizer: string;
  data: string;
  scalePath: string;
  trials: number;
  hardware: string;
  eta: number;
  transferAccepted: boolean;
  negativeControlRejected: boolean;
  forecastable: boolean;
  r2: number;
  exponent: number;
  holdoutError: number;
  conclusion: string;
  fingerprint: string;
};

const PUBLISHED_EVIDENCE: EvidenceCase[] = [
  {
    id: "moe-lmd-adam-a100",
    title: "Sparse MoE · LM/D constant",
    architecture: "Top-1 MoE residual stack",
    optimizer: "Adam",
    data: "Fixed nonlinear teacher · 16,384 points",
    scalePath: "D, L and expert M · LM/D invariant",
    trials: 42,
    hardware: "A100 80 GB",
    eta: 0.03,
    transferAccepted: true,
    negativeControlRejected: true,
    forecastable: true,
    r2: 0.9424914405166642,
    exponent: 0.06869443506581856,
    holdoutError: 0.0030342822119086444,
    conclusion: "Clean positive control: every transfer probe passed, the wrong global-rate control failed, and the largest held-out model landed within 0.3%.",
    fingerprint: "05494395132d4814",
  },
  {
    id: "mlp-adam-a100",
    title: "Dense MLP · fixed-data width/depth",
    architecture: "Pre-norm residual MLP",
    optimizer: "Adam",
    data: "Fixed nonlinear teacher · 16,384 points",
    scalePath: "Joint width + repeats",
    trials: 39,
    hardware: "A100 80 GB",
    eta: 0.003,
    transferAccepted: true,
    negativeControlRejected: false,
    forecastable: false,
    r2: 0.9907623861901285,
    exponent: 0.18071057999876844,
    holdoutError: 0.017468536338471237,
    conclusion: "The fit and 1.7% holdout look strong, but the declared wrong rule was not separated. The app correctly withholds extrapolation.",
    fingerprint: "cc29e690be68d646",
  },
  {
    id: "mlp-sgd-a100",
    title: "Dense MLP · SGD",
    architecture: "Pre-norm residual MLP",
    optimizer: "SGD",
    data: "Fixed nonlinear teacher · 16,384 points",
    scalePath: "Joint width + repeats",
    trials: 39,
    hardware: "A100 80 GB",
    eta: 0.15,
    transferAccepted: true,
    negativeControlRejected: false,
    forecastable: false,
    r2: -2.175069010569538,
    exponent: 0.001,
    holdoutError: 0.0033660970455103288,
    conclusion: "A useful refusal: HP transfer is locally non-inferior, yet loss barely moves relative to seed noise, so there is no defensible scaling law.",
    fingerprint: "0520d28c58e724b6",
  },
  {
    id: "nugpt-width-a100",
    title: "νGPT · width only",
    architecture: "Normalized Transformer",
    optimizer: "Adam β₂=.95",
    data: "Fixed synthetic Markov language",
    scalePath: "128 → 512 width · depth 8",
    trials: 96,
    hardware: "A100 80 GB",
    eta: 0.01,
    transferAccepted: true,
    negativeControlRejected: false,
    forecastable: false,
    r2: -0.1796876971214334,
    exponent: 0.001,
    holdoutError: 0.006976768218329974,
    conclusion: "Normalized η transfers and unit-sphere checks pass, but the task saturates: loss is not monotone and the baseline control is indistinguishable.",
    fingerprint: "1b081704bcdb9758",
  },
  {
    id: "nugpt-depth-a100",
    title: "νGPT · depth only",
    architecture: "Normalized Transformer",
    optimizer: "Adam β₂=.95",
    data: "Fixed synthetic Markov language",
    scalePath: "Depth 2 → 24 · width 256",
    trials: 96,
    hardware: "A100 80 GB",
    eta: 0.01,
    transferAccepted: true,
    negativeControlRejected: false,
    forecastable: false,
    r2: 0.7435039401545588,
    exponent: 1.6572526199767479,
    holdoutError: 0.013585054296579599,
    conclusion: "The held-out model is predicted within 1.4%, but floor identifiability and the negative control fail the stricter forecast contract.",
    fingerprint: "1d3487474b0c89ff",
  },
  {
    id: "nugpt-joint-a100",
    title: "νGPT · width + depth",
    architecture: "Normalized Transformer",
    optimizer: "Adam β₂=.95",
    data: "Fixed synthetic Markov language",
    scalePath: "Width 128 → 512 · depth 4 → 16",
    trials: 96,
    hardware: "A100 80 GB",
    eta: 0.01,
    transferAccepted: true,
    negativeControlRejected: false,
    forecastable: false,
    r2: -0.0018668371572312381,
    exponent: 0.0010000000000026008,
    holdoutError: 0.022162783912068693,
    conclusion: "Normalized η and every sphere invariant pass, but fit loss saturates and turns upward. The global-rate control is better and the held-out point misses its narrow interval, so no law is issued.",
    fingerprint: "674fe6c0938e8284",
  },
];

const WEB_UI_EVIDENCE = [
  { id: "ada53251904f", workflow: "MLP · Adam", coverage: "26 CPU trials", primary: "3.0% held-out error", secondary: "R² 0.984", verdict: "Smoke passed · forecast withheld" },
  { id: "8fe3d0ddc101", workflow: "MLP · SGD", coverage: "24 CPU trials", primary: "Held-out miss", secondary: "2.6% relative error", verdict: "Failure surfaced correctly" },
  { id: "a6151c86e759", workflow: "MoE · Adam · LM/D", coverage: "24 CPU trials", primary: "1.6% held-out error", secondary: "R² 0.836", verdict: "Smoke passed · forecast withheld" },
  { id: "d3c3dfd23d5e", workflow: "νGPT · width", coverage: "50 CPU trials", primary: "0.8% held-out error", secondary: "Sphere invariants pass", verdict: "Smoke passed · span too narrow" },
  { id: "8ed5ecb7beb5", workflow: "νGPT · depth", coverage: "50 CPU trials", primary: "5.6% held-out error", secondary: "Law non-identifiable", verdict: "Refusal surfaced correctly" },
  { id: "d8270ce31988", workflow: "νGPT · width + depth", coverage: "50 CPU trials", primary: "1.0% held-out error", secondary: "Sphere invariants pass", verdict: "Smoke passed · forecast withheld" },
  { id: "43ebfc92e5e6", workflow: "Real-text GPT census", coverage: "24 AdamW trials", primary: "4,012 train tokens", secondary: "0/2 assays qualified", verdict: "Estimator disagreement withheld" },
  { id: "a522064cf12d", workflow: "Real-text GPT · SGD", coverage: "24 SGD trials", primary: "Bcrit assays differ 2.3–2.6×", secondary: "0/2 consensus qualified", verdict: "Near-threshold disagreement withheld" },
  { id: "09a133642297", workflow: "Real-text GPT · Adam", coverage: "24 Adam trials", primary: "Bcrit assays differ 2.3–2.6×", secondary: "0/2 consensus qualified", verdict: "Near-threshold disagreement withheld" },
  { id: "137f7fb48b4f", workflow: "νGPT batch census", coverage: "216 SGD/Adam trials", primary: "S2 Adam Bcrit 13.71", secondary: "1/6 assays qualified", verdict: "Selective qualification" },
  { id: "49e2c7f6f731", workflow: "Constant T/P holdout", coverage: "4 scales · 5 rules", primary: "1.008× T/P spread", secondary: "Best regret −1.9%", verdict: "Boundary optimum · no recommendation" },
] as const;

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

function parsePositiveNumberList(value: string) {
  return value.split(",").map((item) => Number(item.trim())).filter((item) => Number.isFinite(item) && item > 0);
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
  const [targetBatchMultiplier, setTargetBatchMultiplier] = useState(1);
  const [targetHorizonMultiplier, setTargetHorizonMultiplier] = useState(1);
  const [batchRule, setBatchRule] = useState("complete_dp_joint");
  const [batchTransfer, setBatchTransfer] = useState<BatchTransferResult | null>(null);
  const [batchTransferError, setBatchTransferError] = useState<string | null>(null);
  const [batchCampaign, setBatchCampaign] = useState<BatchCampaign>("standard_pretraining_census");
  const [corpusChoice, setCorpusChoice] = useState<CorpusChoice>("fineweb_edu");
  const [corpusTrainMiB, setCorpusTrainMiB] = useState(2);
  const [corpusValidationMiB, setCorpusValidationMiB] = useState(0.5);
  const [corpusJob, setCorpusJob] = useState<PublicCorpusJob | null>(null);
  const [corpusError, setCorpusError] = useState<string | null>(null);
  const [trainingPath, setTrainingPath] = useState("data/pretraining/sample_train.txt");
  const [validationPath, setValidationPath] = useState("data/pretraining/sample_validation.txt");
  const [tokenizer, setTokenizer] = useState<TokenizerId>("byte_v1");
  const [tokenizerCatalog, setTokenizerCatalog] = useState<TokenizerCatalogItem[]>([]);
  const [tokenStreamManifestPath, setTokenStreamManifestPath] = useState("");
  const [precision, setPrecision] = useState<Precision>("fp32");
  const [attentionBackend, setAttentionBackend] = useState<AttentionBackend>("math");
  const [distributedMode, setDistributedMode] = useState<DistributedMode>("none");
  const [pretrainingOptimizer, setPretrainingOptimizer] = useState<PretrainingOptimizer>("adamw");
  const [pretrainingTargetLoss, setPretrainingTargetLoss] = useState(5.4);
  const [pretrainingValidationInterval, setPretrainingValidationInterval] = useState(8);
  const [gpuCount, setGpuCount] = useState(2);
  const [horizonValues, setHorizonValues] = useState("256, 512, 1024, 2048");
  const [horizonRateGrid, setHorizonRateGrid] = useState("0.0003, 0.001, 0.003, 0.01, 0.03, 0.1");
  const [horizonBatchExamples, setHorizonBatchExamples] = useState(2);
  const [horizonSchedules, setHorizonSchedules] = useState<HorizonSchedule[]>([
    "cosine_to_10_percent",
    "linear_warmup_decay_to_zero",
    "wsd",
  ]);
  const [horizonParameterization, setHorizonParameterization] = useState<HorizonParameterization>("jiang_chizat");
  const [jointFitHorizons, setJointFitHorizons] = useState("384, 768, 1536");
  const [jointHeldoutHorizon, setJointHeldoutHorizon] = useState(3072);
  const [jointFitBatches, setJointFitBatches] = useState("2, 4, 8");
  const [jointHeldoutBatch, setJointHeldoutBatch] = useState(16);
  const [jointSchedule, setJointSchedule] = useState<HorizonSchedule>("linear_warmup_decay_to_zero");
  const [forecastTargets, setForecastTargets] = useState("7000000, 12000000, 20000000, 35000000, 60000000, 100000000");
  const [forecastDepths, setForecastDepths] = useState("2, 2, 4, 4, 6, 8");
  const [forecastPredictionTargets, setForecastPredictionTargets] = useState("200000000, 1000000000");
  const [forecastTokensPerParameter, setForecastTokensPerParameter] = useState(20);
  const [forecastContextLength, setForecastContextLength] = useState(512);
  const [forecastBatchExamples, setForecastBatchExamples] = useState(16);
  const [gradientAccumulationSteps, setGradientAccumulationSteps] = useState(1);
  const [checkpointIntervalSteps, setCheckpointIntervalSteps] = useState(100);
  const [batchJob, setBatchJob] = useState<BatchCampaignJob | null>(null);
  const [batchJobError, setBatchJobError] = useState<string | null>(null);
  const [studyHistory, setStudyHistory] = useState<StudyHistoryItem[]>([]);
  const [batchHistory, setBatchHistory] = useState<BatchHistoryItem[]>([]);
  const [historyError, setHistoryError] = useState<string | null>(null);

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
  const baseParameters = parameterCount(scales[0], blockType, numExperts, vocabSize, mlpMultiplier);
  const parameterMultiplier = totalParameters / baseParameters;
  const baseBatchTokens = batchSize * (blockType === "normalized_transformer" ? contextLength : 1);
  const batchRuleOptions = useMemo(() => optimizer === "sgd"
    ? ["none", "sgd_linear_batch"]
    : ["none", "adam_sde_sqrt", "complete_dp_joint", "exact_token_half_life", "horizon_power_fit"], [optimizer]);
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
  const batchJobLocked = batchJob?.status === "queued" || batchJob?.status === "running";
  const corpusJobLocked = corpusJob?.status === "queued" || corpusJob?.status === "running";
  const publicCorpusReady = corpusChoice === "local" || corpusJob?.status === "completed";
  const pinnedTokenizer = tokenizerCatalog.find((item) => item.id === tokenizer);
  const usesPinnedTokenStream = tokenizer === "olmo2_1124";
  const tokenizerVocabSize = tokenizer === "byte_v1" || tokenizer === "olmo2_1124"
    ? pinnedTokenizer?.vocab_size ?? (tokenizer === "byte_v1" ? 260 : 100278)
    : 32768;
  const corpusPathsReady = usesPinnedTokenStream
    ? tokenStreamManifestPath.trim().length > 0
    : trainingPath.trim().length > 0 && validationPath.trim().length > 0;
  const parsedHorizonValues = parsePositiveNumberList(horizonValues).map((value) => Math.round(value));
  const parsedHorizonRates = parsePositiveNumberList(horizonRateGrid);
  const horizonBatchTokens = horizonBatchExamples * (targetDevice === "cuda" ? 64 : 8);
  const horizonConfigValid = batchCampaign !== "horizon_transfer" || (
    parsedHorizonValues.length >= 4
    && parsedHorizonValues.every((value, index) => index === 0 || value > parsedHorizonValues[index - 1])
    && parsedHorizonValues.every((value) => value % horizonBatchTokens === 0)
    && parsedHorizonRates.length >= 3
    && parsedHorizonRates.every((value, index) => index === 0 || value > parsedHorizonRates[index - 1])
    && horizonSchedules.length > 0
    && horizonBatchExamples >= 1
  );
  const parsedJointFitHorizons = parsePositiveNumberList(jointFitHorizons).map((value) => Math.round(value));
  const parsedJointFitBatches = parsePositiveNumberList(jointFitBatches).map((value) => Math.round(value));
  const jointContext = targetDevice === "cuda" ? 128 : 8;
  const jointCells = [
    ...parsedJointFitHorizons.map((tokens) => [tokens, parsedJointFitBatches[0]] as const),
    ...parsedJointFitBatches.slice(1).map((batch) => [parsedJointFitHorizons[0], batch] as const),
    [parsedJointFitHorizons.at(-1), parsedJointFitBatches.at(-1)] as const,
    [jointHeldoutHorizon, jointHeldoutBatch] as const,
  ];
  const jointConfigValid = batchCampaign !== "joint_horizon_batch" || (
    parsedJointFitHorizons.length >= 3
    && parsedJointFitBatches.length >= 3
    && parsedJointFitHorizons.every((value, index) => index === 0 || value > parsedJointFitHorizons[index - 1])
    && parsedJointFitBatches.every((value, index) => index === 0 || value > parsedJointFitBatches[index - 1])
    && jointHeldoutHorizon > (parsedJointFitHorizons.at(-1) ?? Infinity)
    && jointHeldoutBatch > (parsedJointFitBatches.at(-1) ?? Infinity)
    && parsedHorizonRates.length >= 3
    && parsedHorizonRates.every((value, index) => index === 0 || value > parsedHorizonRates[index - 1])
    && jointCells.every(([tokens, batch]) => tokens !== undefined && batch !== undefined && tokens % (batch * jointContext) === 0)
  );
  const parsedForecastTargets = parsePositiveNumberList(forecastTargets).map((value) => Math.round(value));
  const parsedForecastDepths = parsePositiveNumberList(forecastDepths).map((value) => Math.round(value));
  const parsedForecastPredictionTargets = parsePositiveNumberList(forecastPredictionTargets).map((value) => Math.round(value));
  const forecastProcessCount = distributedMode === "none" ? 1 : gpuCount;
  const forecastConfigValid = batchCampaign !== "real_text_scaling_ladder" || (
    targetDevice === "cuda"
    && tokenStreamManifestPath.trim().length > 0
    && parsedForecastTargets.length >= 6
    && parsedForecastDepths.length === parsedForecastTargets.length
    && parsedForecastTargets.every((value, index) => index === 0 || value > parsedForecastTargets[index - 1])
    && parsedForecastPredictionTargets.length > 0
    && parsedForecastPredictionTargets.every((value, index) => index === 0 || value > parsedForecastPredictionTargets[index - 1])
    && (parsedForecastPredictionTargets[0] ?? 0) > (parsedForecastTargets.at(-1) ?? Infinity)
    && Number.isFinite(forecastTokensPerParameter) && forecastTokensPerParameter > 0
    && Number.isInteger(forecastContextLength) && forecastContextLength >= 2
    && Number.isInteger(forecastBatchExamples) && forecastBatchExamples >= 1
    && Number.isInteger(gradientAccumulationSteps) && gradientAccumulationSteps >= 1
    && forecastBatchExamples % (forecastProcessCount * gradientAccumulationSteps) === 0
    && Number.isInteger(checkpointIntervalSteps) && checkpointIntervalSteps >= 0
    && parsedHorizonRates.length >= 3
    && parsedHorizonRates.every((value, index) => index === 0 || value > parsedHorizonRates[index - 1])
    && !(horizonParameterization === "nugpt" && distributedMode === "fsdp")
  );
  const corpusRequired = batchCampaign === "standard_pretraining_census" || batchCampaign === "horizon_transfer" || batchCampaign === "real_text_scaling_ladder";
  const standardRuntimeValid = horizonConfigValid && jointConfigValid && forecastConfigValid && (!corpusRequired
    || (publicCorpusReady
      && corpusPathsReady
      && (batchCampaign !== "standard_pretraining_census" || (Number.isFinite(pretrainingTargetLoss)
        && pretrainingTargetLoss > 0
        && Number.isInteger(pretrainingValidationInterval)
        && pretrainingValidationInterval > 0))
      && (attentionBackend !== "flash" || (precision === "bf16" && targetDevice === "cuda"))
      && (distributedMode === "none" || (targetDevice === "cuda" && gpuCount >= 2))));
  const batchModelContract = batchCampaign === "real_text_scaling_ladder"
    ? horizonParameterization === "jiang_chizat" ? "Jiang + Chizat forecast ladder" : "νGPT forecast ladder"
    : batchCampaign === "standard_pretraining_census" ? "Standard pre-norm GPT"
      : batchCampaign === "transformer_census" ? "Normalized Transformer"
        : batchCampaign === "horizon_transfer" ? horizonParameterization === "jiang_chizat" ? "Jiang MHSA + Chizat FFN · frozen real text" : "Fixed-model νGPT · frozen real text"
          : batchCampaign === "joint_horizon_batch" ? "νGPT joint T × B transfer" : "νGPT constant T/P";
  const batchModelDetail = batchCampaign === "real_text_scaling_ladder"
    ? horizonParameterization === "jiang_chizat"
      ? "Exact CompleteP parameter groups · L·M/D fixed · tied embeddings · hidden upper-rung backtest"
      : "Post-step sphere projection · iteration-aware group LRs · hidden upper-rung backtest"
    : batchCampaign === "standard_pretraining_census" ? "Learned token + position embeddings · causal MHSA · GELU MLP · tied unembed"
      : batchCampaign === "horizon_transfer" ? horizonParameterization === "jiang_chizat" ? "Full CompleteP LR/epsilon groups · 1/L MHSA and mean-field FFN branches · held-out horizon" : "Compare schedule shape and peak-LR laws on one fingerprinted corpus sample without model/batch confounding"
        : batchCampaign === "joint_horizon_batch" ? "Freeze schedule, peak LR, Adam moments, and epsilon before the doubly held-out corner" : "Theory-specific synthetic control";
  const batchExecution = (batchCampaign === "standard_pretraining_census" || batchCampaign === "real_text_scaling_ladder")
    ? `${batchCampaign === "real_text_scaling_ladder" ? "ADAM" : pretrainingOptimizer.toUpperCase()} · ${precision.toUpperCase()} · ${attentionBackend === "flash" ? "FlashAttention" : "PyTorch SDPA"}`
    : "FP32 reference";
  const batchExecutionDetail = distributedMode !== "none" && (batchCampaign === "standard_pretraining_census" || batchCampaign === "real_text_scaling_ladder")
    ? `${gpuCount} GPUs · torchrun ${distributedMode.toUpperCase()} · ${gradientAccumulationSteps}× accumulation`
    : targetDevice === "cuda" ? "Single CUDA process" : "Local smoke profile";

  useEffect(() => {
    const controller = new AbortController();
    fetch(`${API_BASE}/api/health`, { signal: controller.signal })
      .then((response) => setApiOnline(response.ok))
      .catch(() => setApiOnline(false));
    fetch(`${API_BASE}/api/tokenizers`, { signal: controller.signal })
      .then(async (response) => {
        if (!response.ok) throw new Error("Tokenizer registry unavailable");
        const payload = await response.json() as { tokenizers: TokenizerCatalogItem[] };
        setTokenizerCatalog(payload.tokenizers);
      })
      .catch(() => setTokenizerCatalog([]));
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

  useEffect(() => {
    if (!batchJob || !["queued", "running"].includes(batchJob.status)) return;
    const timer = window.setTimeout(async () => {
      try {
        const response = await fetch(`${API_BASE}/api/batch/jobs/${batchJob.id}`);
        if (!response.ok) throw new Error("The batch campaign monitor lost contact with compute.");
        setBatchJob(await response.json() as BatchCampaignJob);
      } catch (caught) {
        setBatchJobError(caught instanceof Error ? caught.message : "Could not refresh this campaign.");
      }
    }, 1200);
    return () => window.clearTimeout(timer);
  }, [batchJob]);

  useEffect(() => {
    if (!corpusJob || !["queued", "running"].includes(corpusJob.status)) return;
    const timer = window.setTimeout(async () => {
      try {
        const response = await fetch(`${API_BASE}/api/corpora/${corpusJob.id}`);
        const payload = await response.json() as PublicCorpusJob & { error?: string };
        if (!response.ok) throw new Error(payload.error ?? "The corpus monitor lost contact with compute.");
        setCorpusJob(payload);
        if (payload.status === "completed" && payload.result) {
          setTokenizer(payload.result.tokenizer);
          setTrainingPath(payload.result.splits.train.path);
          setValidationPath(payload.result.splits.validation.path);
          setTokenStreamManifestPath(payload.result.token_stream_manifest_path ?? "");
        }
      } catch (caught) {
        setCorpusError(caught instanceof Error ? caught.message : "Could not refresh the corpus job.");
      }
    }, 1200);
    return () => window.clearTimeout(timer);
  }, [corpusJob]);

  async function refreshHistory() {
    try {
      const [studyResponse, campaignResponse] = await Promise.all([
        fetch(`${API_BASE}/api/studies`),
        fetch(`${API_BASE}/api/batch/jobs`),
      ]);
      if (!studyResponse.ok || !campaignResponse.ok) {
        throw new Error("The compute service did not return its run ledger.");
      }
      const studies = await studyResponse.json() as { studies: StudyHistoryItem[] };
      const campaigns = await campaignResponse.json() as { jobs: BatchHistoryItem[] };
      setStudyHistory(studies.studies);
      setBatchHistory(campaigns.jobs);
      setHistoryError(null);
    } catch (caught) {
      setHistoryError(caught instanceof Error ? caught.message : "Run history is unavailable.");
    }
  }

  useEffect(() => {
    const timer = window.setTimeout(() => void refreshHistory(), 0);
    return () => window.clearTimeout(timer);
  }, []);

  useEffect(() => {
    if (["completed", "failed", "interrupted"].includes(job?.status ?? "")
      || ["completed", "failed", "interrupted"].includes(batchJob?.status ?? "")) {
      const timer = window.setTimeout(() => void refreshHistory(), 0);
      return () => window.clearTimeout(timer);
    }
  }, [job?.status, batchJob?.status]);

  async function loadStudy(studyId: string) {
    try {
      const response = await fetch(`${API_BASE}/api/studies/${studyId}`);
      const payload = await response.json() as StudyJob & { error?: string };
      if (!response.ok) throw new Error(payload.error ?? "The study could not be loaded.");
      if (payload.spec) {
        const saved = payload.spec;
        setRunProfile(saved.run_profile);
        setBlockType(saved.architecture.block_type);
        setActivation(saved.architecture.activation);
        setInputDimension(saved.architecture.input_dim);
        setNumExperts(saved.architecture.num_experts ?? 4);
        setActiveExperts(saved.architecture.active_experts ?? 1);
        setVocabSize(saved.architecture.vocab_size ?? 32);
        setContextLength(saved.architecture.context_length ?? 16);
        setHeadDimension(saved.architecture.head_dimension ?? 8);
        setMlpMultiplier(saved.architecture.mlp_multiplier ?? 4);
        setOptimizer(saved.optimizer.name);
        setBatchRule(saved.optimizer.name === "sgd" ? "sgd_linear_batch" : "complete_dp_joint");
        setDifficulty(saved.dataset.difficulty);
        setDatasetSize(saved.dataset.n_train);
        setValidationSize(saved.dataset.n_validation);
        setNoiseStd(saved.dataset.noise_std);
        setTeacherWidth(saved.dataset.teacher_width);
        setTeacherDepth(saved.dataset.teacher_depth);
        setMarkovOrder(saved.dataset.markov_order);
        setMarkovStates(saved.dataset.markov_states);
        setSteps(saved.horizon.steps);
        setBatchSize(saved.horizon.batch_size);
        setMicrobatchSize(saved.horizon.microbatch_size);
        setDataScalingMode(saved.data_scaling.mode);
        setDataGrowthFactor(saved.data_scaling.growth_factor);
        setHorizonPolicy(saved.data_scaling.horizon_policy);
        setManualScales(saved.scales);
        setScaleCount(saved.scales.length);
        setTargetDevice(payload.device?.startsWith("cuda") ? "cuda" : "cpu");
        const widthFixed = saved.scales.every((scale) => scale.width === saved.scales[0].width);
        const depthFixed = saved.scales.every((scale) => scale.repeats === saved.scales[0].repeats);
        setScalePath(saved.architecture.block_type === "pre_norm_moe" ? "moe_lmd" : widthFixed ? "depth" : depthFixed ? "width" : "joint");
      }
      setJob(payload);
      setError(null);
      document.getElementById("run")?.scrollIntoView({ behavior: "smooth", block: "start" });
    } catch (caught) {
      setHistoryError(caught instanceof Error ? caught.message : "The study could not be loaded.");
    }
  }

  async function loadBatchCampaign(jobId: string) {
    try {
      const response = await fetch(`${API_BASE}/api/batch/jobs/${jobId}`);
      const payload = await response.json() as BatchCampaignJob & { error?: string };
      if (!response.ok) throw new Error(payload.error ?? "The campaign could not be loaded.");
      setBatchCampaign(payload.campaign);
      setTargetDevice(payload.device?.startsWith("cuda") ? "cuda" : "cpu");
      if ((payload.campaign === "standard_pretraining_census" || payload.campaign === "real_text_scaling_ladder") && payload.config) {
        setCorpusChoice("local");
        setCorpusJob(null);
        const savedDataset = payload.config.dataset;
        const savedRuntime = payload.config.runtime;
        if (savedDataset?.train_path) setTrainingPath(savedDataset.train_path);
        if (savedDataset?.validation_path) setValidationPath(savedDataset.validation_path);
        if (savedDataset?.token_stream_manifest_path) setTokenStreamManifestPath(savedDataset.token_stream_manifest_path);
        if (savedDataset?.tokenizer) setTokenizer(savedDataset.tokenizer);
        if (savedRuntime?.precision) setPrecision(savedRuntime.precision);
        if (savedRuntime?.attention_backend) setAttentionBackend(savedRuntime.attention_backend);
        if (savedRuntime?.distributed) setDistributedMode(savedRuntime.distributed);
        if (savedRuntime?.num_processes) setGpuCount(Math.max(2, savedRuntime.num_processes));
        if (savedRuntime?.gradient_accumulation_steps) setGradientAccumulationSteps(savedRuntime.gradient_accumulation_steps);
        if (savedRuntime?.checkpoint_interval_steps !== undefined) setCheckpointIntervalSteps(savedRuntime.checkpoint_interval_steps);
        const savedOptimizer = payload.config.optimizers?.[0]?.name;
        if (savedOptimizer) setPretrainingOptimizer(savedOptimizer);
        if (payload.config.target_validation_loss) setPretrainingTargetLoss(payload.config.target_validation_loss);
        if (payload.config.validation_interval) setPretrainingValidationInterval(payload.config.validation_interval);
        if (payload.campaign === "real_text_scaling_ladder") {
          setHorizonParameterization(payload.config.architecture?.block_type === "normalized_transformer" ? "nugpt" : "jiang_chizat");
          if (payload.config.architecture?.context_length) setForecastContextLength(payload.config.architecture.context_length);
          if (payload.config.ladder?.target_parameters) setForecastTargets(payload.config.ladder.target_parameters.join(", "));
          if (payload.config.ladder?.depths) setForecastDepths(payload.config.ladder.depths.join(", "));
          if (payload.config.ladder?.tokens_per_parameter) setForecastTokensPerParameter(payload.config.ladder.tokens_per_parameter);
          if (payload.config.ladder?.target_forecasts) setForecastPredictionTargets(payload.config.ladder.target_forecasts.join(", "));
          if (payload.config.batch_examples) setForecastBatchExamples(payload.config.batch_examples);
          if (payload.config.optimizer?.learning_rates) setHorizonRateGrid(payload.config.optimizer.learning_rates.join(", "));
        }
      }
      setBatchJob(payload);
      setBatchJobError(null);
      document.getElementById("campaign")?.scrollIntoView({ behavior: "smooth", block: "start" });
    } catch (caught) {
      setHistoryError(caught instanceof Error ? caught.message : "The campaign could not be loaded.");
    }
  }

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

  function applyWorkflow(workflow: "mlp-adam" | "mlp-sgd" | "moe-adam" | "nugpt-width" | "nugpt-depth" | "nugpt-joint") {
    if (studyLocked) return;
    const architecture: BlockType = workflow.startsWith("nugpt")
      ? "normalized_transformer"
      : workflow === "moe-adam"
        ? "pre_norm_moe"
        : "pre_norm_mlp";
    applyRunProfile("smoke", architecture);
    setBlockType(architecture);
    setActivation(architecture === "normalized_transformer" ? "silu" : "gelu");
    setOptimizer(workflow === "mlp-sgd" ? "sgd" : "adam");
    setBatchRule(workflow === "mlp-sgd" ? "sgd_linear_batch" : "complete_dp_joint");
    setScalePath(workflow === "nugpt-width" ? "width" : workflow === "nugpt-depth" ? "depth" : architecture === "pre_norm_moe" ? "moe_lmd" : "joint");
    setBatchTransfer(null);
    setJob(null);
    window.setTimeout(() => document.getElementById("study")?.scrollIntoView({ behavior: "smooth", block: "start" }), 0);
  }

  function applyBatchWorkflow(campaign: BatchCampaign) {
    if (batchJobLocked) return;
    setBatchCampaign(campaign);
    setTargetDevice("cpu");
    setDistributedMode("none");
    setAttentionBackend("math");
    setPrecision("fp32");
    if (campaign === "standard_pretraining_census") setPretrainingOptimizer("adamw");
    if (campaign === "horizon_transfer") {
      setHorizonParameterization("jiang_chizat");
      setHorizonValues("256, 512, 1024, 2048");
      setHorizonRateGrid("0.0003, 0.001, 0.003, 0.01, 0.03, 0.1");
      setHorizonBatchExamples(2);
      setHorizonSchedules(["cosine_to_10_percent", "linear_warmup_decay_to_zero", "wsd"]);
      setCorpusChoice("fineweb_edu");
      setCorpusTrainMiB(2);
      setCorpusValidationMiB(0.5);
      setCorpusJob(null);
    }
    if (campaign === "joint_horizon_batch") {
      setJointFitHorizons("384, 768, 1536");
      setJointHeldoutHorizon(3072);
      setJointFitBatches("2, 4, 8");
      setJointHeldoutBatch(16);
      setJointSchedule("linear_warmup_decay_to_zero");
      setHorizonRateGrid("0.0003, 0.001, 0.003, 0.01, 0.03, 0.1");
    }
    if (campaign === "real_text_scaling_ladder") {
      setTargetDevice("cuda");
      setPrecision("bf16");
      setAttentionBackend("auto");
      setDistributedMode("ddp");
      setGpuCount(2);
      setHorizonParameterization("jiang_chizat");
      setTokenizer("olmo2_1124");
      setTokenStreamManifestPath("");
      setCorpusChoice("fineweb_edu");
      setCorpusTrainMiB(12_288);
      setCorpusValidationMiB(512);
      setCorpusJob(null);
      setForecastTargets("7000000, 12000000, 20000000, 35000000, 60000000, 100000000");
      setForecastDepths("2, 2, 4, 4, 6, 8");
      setForecastPredictionTargets("200000000, 1000000000");
      setForecastTokensPerParameter(20);
      setForecastContextLength(512);
      setForecastBatchExamples(16);
      setGradientAccumulationSteps(1);
      setCheckpointIntervalSteps(100);
      setHorizonRateGrid("0.0001, 0.0003, 0.001, 0.003, 0.01");
    }
    setPretrainingTargetLoss(5.4);
    setPretrainingValidationInterval(8);
    setBatchJob(null);
    setBatchJobError(null);
    window.setTimeout(() => document.getElementById("campaign")?.scrollIntoView({ behavior: "smooth", block: "start" }), 0);
  }

  function chooseBlock(next: BlockType) {
    if (studyLocked) return;
    setBlockType(next);
    setManualScales(null);
    applyRunProfile(runProfile, next);
    if (next === "pre_norm_moe" || next === "normalized_transformer") {
      setOptimizer("adam");
      setBatchRule("complete_dp_joint");
      setBatchTransfer(null);
    }
    setActivation(next === "normalized_transformer" ? "silu" : "gelu");
    setSelectedNode("residual");
    setJob(null);
  }

  function markBatchCampaignEdited() {
    if (!batchJobLocked) {
      setBatchJob(null);
      setBatchJobError(null);
    }
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

  async function previewBatchTransfer() {
    setBatchTransferError(null);
    try {
      const sourceRate = learningRates[Math.floor(learningRates.length / 2)];
      const response = await fetch(`${API_BASE}/api/batch/transfer`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          rule: batchRule,
          optimizer: optimizer === "sgd"
            ? { name: "sgd", learning_rate: sourceRate, momentum: 0 }
            : {
              name: "adam",
              learning_rate: sourceRate,
              beta1: 0.9,
              beta2: blockType === "normalized_transformer" ? 0.95 : 0.999,
              epsilon: blockType === "normalized_transformer" ? 1e-16 : 1e-8,
            },
          context: {
            base_parameters: baseParameters,
            target_parameters: totalParameters,
            base_total_tokens: tokenBudget,
            target_total_tokens: Math.max(1, Math.round(tokenBudget * targetHorizonMultiplier)),
            base_batch_tokens: baseBatchTokens,
            target_batch_tokens: Math.max(1, Math.round(baseBatchTokens * targetBatchMultiplier)),
          },
        }),
      });
      const payload = await response.json() as BatchTransferResult & { error?: string };
      if (!response.ok) throw new Error(payload.error ?? "The transfer rule could not be evaluated.");
      setBatchTransfer(payload);
    } catch (caught) {
      setBatchTransferError(caught instanceof Error ? caught.message : "Batch transfer is unavailable.");
    }
  }

  async function preparePublicCorpus() {
    if (corpusChoice === "local" || corpusJobLocked) return;
    setCorpusError(null);
    setBatchJob(null);
    try {
      const response = await fetch(`${API_BASE}/api/corpora`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          spec: {
            source: corpusChoice,
            train_bytes: Math.max(65_536, Math.round(corpusTrainMiB * 1024 * 1024)),
            validation_bytes: Math.max(16_384, Math.round(corpusValidationMiB * 1024 * 1024)),
            acquisition_backend: batchCampaign === "real_text_scaling_ladder" ? "parquet" : "viewer_rows",
            source_batch_rows: batchCampaign === "real_text_scaling_ladder" ? 8192 : 128,
            maximum_documents_per_split: batchCampaign === "real_text_scaling_ladder" ? 100_000_000 : 50_000,
            tokenizer,
            token_shard_tokens: 16_777_216,
          },
        }),
      });
      const payload = await response.json() as PublicCorpusJob & { error?: string };
      if (!response.ok) throw new Error(payload.error ?? "The public corpus could not be prepared.");
      setCorpusJob(payload);
      if (payload.status === "completed" && payload.result) {
        setTokenizer(payload.result.tokenizer);
        setTrainingPath(payload.result.splits.train.path);
        setValidationPath(payload.result.splits.validation.path);
        setTokenStreamManifestPath(payload.result.token_stream_manifest_path ?? "");
      }
    } catch (caught) {
      setCorpusError(caught instanceof Error ? caught.message : "The public corpus service is unavailable.");
    }
  }

  function batchCampaignConfig(): Record<string, unknown> {
    const syntheticArchitecture = {
      block_type: "normalized_transformer",
      activation: "silu",
      vocab_size: 16,
      context_length: 8,
      head_dimension: 4,
      mlp_multiplier: 2,
      reference_width: 16,
      reference_depth: 2,
    };
    const syntheticDataset = {
      task_type: "synthetic_markov",
      n_train: 256,
      n_validation: 128,
      noise_std: 0.03,
      seed: 1729,
      markov_order: 2,
      markov_states: 4,
    };
    if (batchCampaign === "real_text_scaling_ladder") {
      const useJiangChizat = horizonParameterization === "jiang_chizat";
      const processCount = distributedMode === "none" ? 1 : gpuCount;
      return {
        run_profile: targetDevice === "cuda" ? "forecast" : "smoke",
        architecture: useJiangChizat ? {
          block_type: "jiang_chizat_transformer",
          vocab_size: tokenizerVocabSize,
          context_length: forecastContextLength,
          head_dimension: 64,
          reference_depth: 2,
          reference_hidden_width: 256,
          reference_residual_width: 128,
        } : {
          block_type: "normalized_transformer",
          activation: "silu",
          vocab_size: tokenizerVocabSize,
          context_length: forecastContextLength,
          head_dimension: 32,
          mlp_multiplier: 4,
          reference_width: 128,
          reference_depth: 4,
        },
        dataset: {
          task_type: "tokenized_text",
          token_stream_manifest_path: tokenStreamManifestPath.trim(),
          tokenizer,
          maximum_bytes: 2_199_023_255_552,
        },
        ladder: {
          target_parameters: parsedForecastTargets,
          depths: parsedForecastDepths,
          ...(useJiangChizat ? { rho_lm_over_d: 4 } : {}),
          tokens_per_parameter: forecastTokensPerParameter,
          heldout_scale_count: 1,
          reference_scale_index: 0,
          maximum_parameter_error_fraction: 0.15,
          minimum_parameter_span: 30,
          maximum_repetition_ratio: 1,
          target_forecasts: parsedForecastPredictionTargets,
          maximum_extrapolation_factor: 10,
          maximum_family_spread: 0.08,
          maximum_backtest_relative_error: 0.1,
        },
        optimizer: {
          name: "adam",
          beta1: 0.9,
          beta2: 0.95,
          epsilon: useJiangChizat ? 1e-12 : 1e-16,
          weight_decay: 0,
          learning_rates: parsedHorizonRates,
          ...(useJiangChizat ? { learning_rate_multipliers: {
            jiang_embeddings: 1,
            jiang_norms: 1,
            jiang_attention_qkv: 0.0625,
            jiang_attention_output: 0.5,
            jiang_ffn_up: 1,
            jiang_ffn_down: 0.0625,
            jiang_other_biases: 1,
          } } : { output_learning_rate_multiplier: 0.5 }),
        },
        schedule: useJiangChizat ? "jiang_half_warmup_constant" : "cosine_to_10_percent",
        batch_examples: forecastBatchExamples,
        validation_examples: targetDevice === "cuda" ? 256 : 16,
        validation_interval: targetDevice === "cuda" ? 100 : 1,
        seeds: targetDevice === "cuda" ? [11, 29, 47] : [3],
        runtime: {
          precision,
          attention_backend: attentionBackend,
          distributed: distributedMode,
          num_processes: processCount,
          gradient_accumulation_steps: gradientAccumulationSteps,
          activation_checkpointing: true,
          checkpoint_interval_steps: checkpointIntervalSteps,
          resume: true,
        },
        bootstrap_samples: targetDevice === "cuda" ? 400 : 0,
        run_negative_control: true,
        negative_control_minimum_degradation: 0,
      };
    }
    if (batchCampaign === "horizon_transfer") {
      const onA100 = targetDevice === "cuda";
      const useJiangChizat = horizonParameterization === "jiang_chizat";
      return {
        architecture: useJiangChizat ? (onA100 ? {
          block_type: "jiang_chizat_transformer",
          vocab_size: tokenizerVocabSize,
          context_length: 64,
          head_dimension: 64,
          reference_depth: 2,
          reference_hidden_width: 128,
          reference_residual_width: 64,
          depth: 2,
          hidden_width: 256,
          residual_width: 128,
        } : {
          block_type: "jiang_chizat_transformer",
          vocab_size: tokenizerVocabSize,
          context_length: 8,
          head_dimension: 4,
          reference_depth: 1,
          reference_hidden_width: 16,
          reference_residual_width: 16,
          depth: 1,
          hidden_width: 16,
          residual_width: 16,
        }) : (onA100 ? { ...syntheticArchitecture, vocab_size: tokenizerVocabSize, context_length: 64, head_dimension: 64, mlp_multiplier: 4, reference_width: 128, reference_depth: 4 } : { ...syntheticArchitecture, vocab_size: tokenizerVocabSize }),
        dataset: {
          task_type: "tokenized_text",
          ...(usesPinnedTokenStream
            ? { token_stream_manifest_path: tokenStreamManifestPath.trim() }
            : { train_path: trainingPath.trim(), validation_path: validationPath.trim() }),
          tokenizer,
          n_train: onA100 ? 16384 : 256,
          n_validation: onA100 ? 1024 : 128,
          seed: 1729,
          maximum_bytes: onA100 ? 68_719_476_736 : 536_870_912,
        },
        ...(!useJiangChizat ? { scale: onA100 ? { name: "fixed-anchor", width: 128, repeats: 4 } : { name: "fixed-anchor", width: 16, repeats: 2 } } : {}),
        optimizer: {
          name: "adam",
          beta1: 0.9,
          beta2: 0.95,
          epsilon: useJiangChizat ? 1e-12 : 1e-16,
          weight_decay: 0,
          learning_rates: parsedHorizonRates,
          ...(useJiangChizat ? { learning_rate_multipliers: {
            jiang_embeddings: 1,
            jiang_norms: 1,
            jiang_attention_qkv: 0.0625,
            jiang_attention_output: 0.5,
            jiang_ffn_up: 1,
            jiang_ffn_down: 0.0625,
            jiang_other_biases: 1,
          } } : {}),
        },
        presented_tokens: parsedHorizonValues,
        batch_examples: horizonBatchExamples,
        schedules: useJiangChizat ? ["jiang_half_warmup_constant"] : horizonSchedules,
        horizon_rules: ["none", "nugpt_one_third", "fitted_power"],
        validation_interval: targetDevice === "cuda" ? 2048 : 128,
        seeds: targetDevice === "cuda" ? [11, 29, 47] : [11, 29],
        minimum_seeds: targetDevice === "cuda" ? 3 : 2,
        minimum_fit_horizon_span: onA100 ? 8 : 4,
        bootstrap_samples: targetDevice === "cuda" ? 400 : 100,
        maximum_relative_oracle_regret: 0.02,
        minimum_recovered_improvement: 0.9,
      };
    }
    if (batchCampaign === "joint_horizon_batch") {
      const onA100 = targetDevice === "cuda";
      return {
        architecture: onA100 ? { ...syntheticArchitecture, vocab_size: 256, context_length: 128, head_dimension: 64, mlp_multiplier: 4, reference_width: 256, reference_depth: 8 } : syntheticArchitecture,
        dataset: onA100 ? { ...syntheticDataset, n_train: 4096, n_validation: 1024, markov_states: 16 } : syntheticDataset,
        scale: onA100 ? { name: "fixed-anchor", width: 128, repeats: 8 } : { name: "fixed-anchor", width: 16, repeats: 2 },
        optimizer: {
          name: "adam",
          beta1: 0.9,
          beta2: 0.95,
          epsilon: 1e-16,
          weight_decay: 0,
          learning_rates: parsedHorizonRates,
        },
        fit_presented_tokens: parsedJointFitHorizons,
        heldout_presented_tokens: jointHeldoutHorizon,
        fit_batch_examples: parsedJointFitBatches,
        heldout_batch_examples: jointHeldoutBatch,
        schedule: jointSchedule,
        joint_rules: ["none", "horizon_fitted_only", "batch_fitted_only", "separable_fitted_peak", "horizon_fitted_x_adam_sde_batch", "one_third_x_adam_sde_batch", "complete_dp_joint", "exact_token_half_life_joint"],
        validation_interval: onA100 ? 32 : 8,
        seeds: onA100 ? [11, 29, 47] : [11, 29],
        minimum_seeds: onA100 ? 3 : 2,
        minimum_fit_horizon_span: onA100 ? 8 : 4,
        minimum_fit_batch_span: 4,
        minimum_axis_fit_r_squared: 0.8,
        bootstrap_samples: onA100 ? 400 : 200,
        maximum_crosscheck_regret: 0.02,
        maximum_relative_oracle_regret: 0.02,
        minimum_recovered_improvement: 0.9,
      };
    }
    if (batchCampaign === "transformer_census") {
      return {
        architecture: syntheticArchitecture,
        dataset: syntheticDataset,
        scales: [
          { name: "S1", width: 8, repeats: 1 },
          { name: "S2", width: 12, repeats: 1 },
          { name: "S3", width: 16, repeats: 2 },
        ],
        batch_examples: [1, 2, 4, 8, 16, 32],
        total_tokens: 2048,
        checkpoint_tokens: 512,
        continuation_tokens: 512,
        target_validation_loss: 2.5,
        validation_interval: 4,
        gradient_noise_samples: 12,
        seeds: [11, 29],
        optimizers: [
          { name: "sgd", momentum: 0, learning_rates: [0.01, 0.03, 0.1] },
          { name: "adam", beta1: 0.9, beta2: 0.99, epsilon: 1e-8, learning_rates: [0.0003, 0.001, 0.003] },
        ],
      };
    }
    if (batchCampaign === "constant_tpp") {
      return {
        architecture: syntheticArchitecture,
        dataset: syntheticDataset,
        scales: [
          { name: "S1", width: 8, repeats: 1 },
          { name: "S2", width: 12, repeats: 1 },
          { name: "S3", width: 16, repeats: 2 },
          { name: "S4-heldout", width: 24, repeats: 3 },
        ],
        optimizer: { name: "adam", beta1: 0.9, beta2: 0.99, epsilon: 1e-8, learning_rates: [0.0003, 0.001, 0.003] },
        tokens_per_parameter: 1,
        base_batch_examples: 2,
        batch_growth_exponent: 0,
        validation_interval: 8,
        seeds: [11, 29],
        transfer_rules: ["none", "adam_sde_sqrt", "complete_dp_joint", "exact_token_half_life", "horizon_power_fit"],
      };
    }
    const onA100 = targetDevice === "cuda";
    const context = onA100 ? 128 : 16;
    const processCount = distributedMode === "none" ? 1 : gpuCount;
    const a100BaseBatch = Math.ceil(8 / processCount) * processCount;
    const batches = onA100
      ? [a100BaseBatch, 2 * a100BaseBatch, 4 * a100BaseBatch, 8 * a100BaseBatch]
      : [1, 2, 4, 8];
    const largestBatch = batches.at(-1) ?? 8;
    const totalTokens = onA100 ? largestBatch * context * 128 : 2048;
    return {
      model: {
        vocab_size: tokenizerVocabSize,
        context_length: context,
        width: onA100 ? 128 : 32,
        depth: onA100 ? 4 : 2,
        num_heads: 4,
        mlp_multiplier: 4,
        dropout: 0,
        tie_embeddings: true,
      },
      dataset: {
        ...(usesPinnedTokenStream
          ? { token_stream_manifest_path: tokenStreamManifestPath.trim() }
          : { train_path: trainingPath.trim(), validation_path: validationPath.trim() }),
        tokenizer,
        maximum_bytes: onA100 ? 68_719_476_736 : 536_870_912,
      },
      runtime: {
        precision,
        attention_backend: attentionBackend,
        distributed: distributedMode,
        num_processes: processCount,
        gradient_accumulation_steps: gradientAccumulationSteps,
        activation_checkpointing: targetDevice === "cuda",
        checkpoint_interval_steps: checkpointIntervalSteps,
        resume: true,
      },
      scales: onA100 ? [
        { name: "S1", width: 128, depth: 4, num_heads: 4 },
        { name: "S2", width: 192, depth: 6, num_heads: 6 },
        { name: "S3", width: 256, depth: 8, num_heads: 8 },
      ] : [
        { name: "S1", width: 24, depth: 1, num_heads: 3 },
        { name: "S2", width: 32, depth: 2, num_heads: 4 },
      ],
      batch_examples: batches,
      total_tokens: totalTokens,
      checkpoint_tokens: onA100 ? largestBatch * context * 8 : 256,
      continuation_tokens: onA100 ? largestBatch * context * 16 : 512,
      target_validation_loss: pretrainingTargetLoss,
      validation_interval: pretrainingValidationInterval,
      validation_examples: onA100 ? 128 : 16,
      gradient_noise_samples: 8,
      warmup_steps: onA100 ? 16 : 4,
      minimum_learning_rate_ratio: 0.1,
      seeds: onA100 ? [11, 29] : [11],
      optimizers: [pretrainingOptimizer === "sgd" ? {
        name: "sgd",
        momentum: 0,
        weight_decay: 0,
        learning_rates: [0.01, 0.03, 0.1],
      } : {
        name: pretrainingOptimizer,
        beta1: 0.9,
        beta2: 0.95,
        epsilon: 1e-8,
        weight_decay: pretrainingOptimizer === "adamw" ? 0.1 : 0,
        learning_rates: onA100 ? [0.0001, 0.0003, 0.001] : [0.0003, 0.001, 0.003],
      }],
    };
  }

  async function startBatchCampaign() {
    if (batchJobLocked || !standardRuntimeValid) return;
    setBatchJobError(null);
    try {
      const response = await fetch(`${API_BASE}/api/batch/jobs`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          campaign: batchCampaign,
          config: batchCampaignConfig(),
          device: targetDevice,
        }),
      });
      const payload = await response.json() as BatchCampaignJob & { error?: string };
      if (!response.ok) throw new Error(payload.error ?? "The batch campaign could not start.");
      setApiOnline(true);
      setBatchJob(payload);
    } catch (caught) {
      setApiOnline(false);
      setBatchJobError(caught instanceof Error ? caught.message : "The batch campaign service is unavailable.");
    }
  }

  const progressPercent = job?.progress.total
    ? Math.min(100, Math.round((job.progress.completed / job.progress.total) * 100))
    : 0;
  const result = job?.result;
  const batchProgressPercent = batchJob?.progress.total
    ? Math.min(100, Math.round((batchJob.progress.completed / batchJob.progress.total) * 100))
    : 0;
  const batchAnalyses = batchJob?.result?.scale_optimizer_analyses
    ?? batchJob?.result?.analyses
    ?? [];

  return (
    <main>
      <header className="topbar">
        <a className="brand" href="#top" aria-label="AI Theorist Autoscaler home">
          <span className="brand-mark"><b /><b /><b /></span>
          <span><strong>AI Theorist</strong><small>Autoscaler</small></span>
        </a>
        <nav aria-label="Workspace sections">
          <a href="#recipes">Recipes</a>
          <a href="#architecture">Architecture</a>
          <a href="#campaign">Campaigns</a>
          <a href="#study">Study</a>
          <a href="#batch">Batch scaling</a>
          <a href="#run">Results</a>
          <a href="#evidence">Evidence</a>
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

      <section className="recipe-section section-shell" id="recipes">
        <div className="section-heading compact-heading">
          <div><p className="eyebrow">Reproducible workflows</p><h2>Start from a validation recipe.</h2></div>
          <p>Each recipe fills a conservative smoke contract first. Inspect every field, then launch it through the same job system used by the larger campaigns.</p>
        </div>
        <div className="recipe-grid">
          <button onClick={() => applyWorkflow("mlp-adam")}><span>Dense</span><strong>MLP · Adam</strong><small>Fixed data · joint width/repeats</small></button>
          <button onClick={() => applyWorkflow("mlp-sgd")}><span>Dense</span><strong>MLP · SGD</strong><small>Raw LR conversion stress test</small></button>
          <button onClick={() => applyWorkflow("moe-adam")}><span>Sparse</span><strong>MoE · Adam</strong><small>Keep LM/D constant</small></button>
          <button onClick={() => applyWorkflow("nugpt-width")}><span>Normalized</span><strong>νGPT · width</strong><small>Width-only transfer</small></button>
          <button onClick={() => applyWorkflow("nugpt-depth")}><span>Normalized</span><strong>νGPT · depth</strong><small>Depth-only transfer</small></button>
          <button onClick={() => applyWorkflow("nugpt-joint")}><span>Normalized</span><strong>νGPT · joint</strong><small>Width + depth transfer</small></button>
          <button onClick={() => applyBatchWorkflow("standard_pretraining_census")}><span>Real text</span><strong>GPT batch census</strong><small>Tokenized corpus · critical batch</small></button>
          <button onClick={() => applyBatchWorkflow("horizon_transfer")}><span>Real text</span><strong>Horizon transfer</strong><small>Jiang+Chizat or νGPT · frozen holdout</small></button>
          <button onClick={() => applyBatchWorkflow("real_text_scaling_ladder")}><span>Forecast grade</span><strong>Real-text scaling ladder</strong><small>Exact Jiang or νGPT · hidden upper rung</small></button>
          <button onClick={() => applyBatchWorkflow("transformer_census")}><span>Synthetic</span><strong>νGPT batch census</strong><small>Three-estimator qualification</small></button>
          <button onClick={() => applyBatchWorkflow("constant_tpp")}><span>Tokens / parameter</span><strong>Constant T/P</strong><small>Held-out schedule transfer</small></button>
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
            <div className="coming-soon"><span>Later</span><p>Convolution, Muon, and general DAGs remain outside this validated slice. AdamW is available in the standard GPT pretraining census; scaling studies currently expose SGD and Adam.</p></div>
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

      <section className="campaign-section section-shell" id="campaign" style={{ scrollMarginTop: 88 }}>
        <div className="section-heading compact-heading">
          <div><p className="eyebrow">Executable batch campaigns</p><h2>Run the census on real or synthetic tokens.</h2></div>
          <p>Every campaign is persisted, fingerprinted, and resumable. Public corpora are frozen before training. Two-stage holdout designs keep final transfer cells untouched until the rule is fixed.</p>
        </div>
        <div className="panel campaign-panel">
          <div className="panel-title"><span>Runnable campaign</span><small>Persistent · resumable</small></div>
          <div className="campaign-body">
            <div className="campaign-controls">
              <label><span>Campaign</span><select disabled={batchJobLocked} value={batchCampaign} onChange={(event) => { applyBatchWorkflow(event.target.value as BatchCampaign); }}><option value="standard_pretraining_census">Real-text Transformer census</option><option value="real_text_scaling_ladder">Real-text forecast ladder</option><option value="transformer_census">νGPT synthetic census</option><option value="horizon_transfer">LR schedule + horizon transfer</option><option value="joint_horizon_batch">Joint horizon × batch holdout</option><option value="constant_tpp">Constant T/P holdout</option></select></label>
              <label><span>Compute</span><select disabled={batchJobLocked} value={targetDevice} onChange={(event) => { const next = event.target.value as "cpu" | "cuda"; setTargetDevice(next); setPretrainingTargetLoss(tokenizer === "byte_v1" ? (next === "cuda" ? 3 : 5.4) : 9.4); setPretrainingValidationInterval(next === "cuda" ? 16 : 8); if (corpusChoice !== "local") { setCorpusTrainMiB(batchCampaign === "real_text_scaling_ladder" ? 12_288 : next === "cuda" ? 64 : 2); setCorpusValidationMiB(batchCampaign === "real_text_scaling_ladder" ? 512 : next === "cuda" ? 8 : 0.5); setCorpusJob(null); } if (batchCampaign === "horizon_transfer") { setHorizonValues(next === "cuda" ? "65536, 131072, 262144, 524288, 1048576" : "256, 512, 1024, 2048"); setHorizonRateGrid(next === "cuda" ? "0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1" : "0.0003, 0.001, 0.003, 0.01, 0.03, 0.1"); setHorizonBatchExamples(next === "cuda" ? (horizonParameterization === "jiang_chizat" ? 16 : 8) : 2); } if (batchCampaign === "joint_horizon_batch") { setJointFitHorizons(next === "cuda" ? "65536, 131072, 262144, 524288" : "384, 768, 1536"); setJointHeldoutHorizon(next === "cuda" ? 1048576 : 3072); setJointFitBatches("2, 4, 8"); setJointHeldoutBatch(16); setHorizonRateGrid(next === "cuda" ? "0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1" : "0.0003, 0.001, 0.003, 0.01, 0.03, 0.1"); } if (batchCampaign === "real_text_scaling_ladder") { setPrecision(next === "cuda" ? "bf16" : "fp32"); setAttentionBackend(next === "cuda" ? "auto" : "math"); setDistributedMode(next === "cuda" ? "ddp" : "none"); } if (next === "cpu") { setDistributedMode("none"); if (attentionBackend === "flash") setAttentionBackend("math"); } markBatchCampaignEdited(); }}><option value="cpu">Local CPU smoke</option><option value="cuda">A100 / CUDA</option></select></label>
              {batchCampaign === "standard_pretraining_census" && <label><span>Optimizer</span><select disabled={batchJobLocked} value={pretrainingOptimizer} onChange={(event) => { setPretrainingOptimizer(event.target.value as PretrainingOptimizer); markBatchCampaignEdited(); }}><option value="adamw">AdamW</option><option value="adam">Adam</option><option value="sgd">SGD</option></select></label>}
              {(batchCampaign === "standard_pretraining_census" || batchCampaign === "real_text_scaling_ladder") && <>
                <label><span>Precision</span><select disabled={batchJobLocked} value={precision} onChange={(event) => { const next = event.target.value as Precision; setPrecision(next); if (next === "fp32" && attentionBackend === "flash") setAttentionBackend("math"); markBatchCampaignEdited(); }}><option value="fp32">FP32</option><option value="bf16">BF16</option></select></label>
                <label><span>Attention kernel</span><select disabled={batchJobLocked} value={attentionBackend} onChange={(event) => { const next = event.target.value as AttentionBackend; setAttentionBackend(next); if (next === "flash") { setPrecision("bf16"); setTargetDevice("cuda"); setPretrainingTargetLoss(3); setPretrainingValidationInterval(16); } markBatchCampaignEdited(); }}><option value="math">SDPA math</option><option value="auto">SDPA automatic</option><option value="flash">FlashAttention</option></select></label>
                <label><span>Parallel mode</span><select disabled={batchJobLocked} value={distributedMode} onChange={(event) => { const next = event.target.value as DistributedMode; setDistributedMode(next); if (next !== "none") setTargetDevice("cuda"); markBatchCampaignEdited(); }}><option value="none">Single process</option><option value="ddp">Single-node DDP</option>{!(batchCampaign === "real_text_scaling_ladder" && horizonParameterization === "nugpt") && <option value="fsdp">Single-node FSDP</option>}</select></label>
                <label><span>GPU processes</span><input disabled={batchJobLocked || distributedMode === "none"} type="number" min="2" max="8" value={gpuCount} onChange={(event) => { setGpuCount(Math.min(8, Math.max(2, Number(event.target.value)))); markBatchCampaignEdited(); }} /></label>
                <label><span>Gradient accumulation</span><input disabled={batchJobLocked} type="number" min="1" value={gradientAccumulationSteps} onChange={(event) => { setGradientAccumulationSteps(Math.max(1, Math.round(Number(event.target.value)))); markBatchCampaignEdited(); }} /></label>
                <label><span>Checkpoint cadence</span><input disabled={batchJobLocked} type="number" min="0" value={checkpointIntervalSteps} onChange={(event) => { setCheckpointIntervalSteps(Math.max(0, Math.round(Number(event.target.value)))); markBatchCampaignEdited(); }} /><small>steps · 0 disables</small></label>
              </>}
            </div>
            {(batchCampaign === "standard_pretraining_census" || batchCampaign === "horizon_transfer" || batchCampaign === "real_text_scaling_ladder") && (
              <div className="dataset-contract public-corpus-contract">
                <label><span>Corpus source</span><select disabled={batchJobLocked || corpusJobLocked} value={corpusChoice} onChange={(event) => { const next = event.target.value as CorpusChoice; setCorpusChoice(next); setCorpusJob(null); setCorpusError(null); setTokenStreamManifestPath(""); if (next !== "local") { if (batchCampaign === "real_text_scaling_ladder") { setTokenizer("olmo2_1124"); setCorpusTrainMiB(12_288); setCorpusValidationMiB(512); } else { if (tokenizer === "uint16_bin_v1" || tokenizer === "uint32_bin_v1") setTokenizer("byte_v1"); setCorpusTrainMiB(targetDevice === "cuda" ? 64 : 2); setCorpusValidationMiB(targetDevice === "cuda" ? 8 : 0.5); } } markBatchCampaignEdited(); }}><option value="fineweb_edu">FineWeb-Edu sample-10BT</option><option value="openwebtext">OpenWebText</option><option value="local">Local files / pretokenized</option></select></label>
                {corpusChoice === "local" ? <>
                  {usesPinnedTokenStream ? <label><span>Verified token-stream manifest</span><input disabled={batchJobLocked} value={tokenStreamManifestPath} onChange={(event) => { setTokenStreamManifestPath(event.target.value); markBatchCampaignEdited(); }} /></label> : <>
                    <label><span>Training corpus</span><input disabled={batchJobLocked} value={trainingPath} onChange={(event) => { setTrainingPath(event.target.value); markBatchCampaignEdited(); }} /></label>
                    <label><span>Validation corpus</span><input disabled={batchJobLocked} value={validationPath} onChange={(event) => { setValidationPath(event.target.value); markBatchCampaignEdited(); }} /></label>
                  </>}
                </> : <>
                  <label><span>Training text</span><input disabled={batchJobLocked || corpusJobLocked} type="number" min="0.0625" step="0.5" value={corpusTrainMiB} onChange={(event) => { setCorpusTrainMiB(Math.max(0.0625, Number(event.target.value))); setCorpusJob(null); markBatchCampaignEdited(); }} /><small>MiB</small></label>
                  <label><span>Held-out text</span><input disabled={batchJobLocked || corpusJobLocked} type="number" min="0.015625" step="0.5" value={corpusValidationMiB} onChange={(event) => { setCorpusValidationMiB(Math.max(0.015625, Number(event.target.value))); setCorpusJob(null); markBatchCampaignEdited(); }} /><small>MiB</small></label>
                  <button className="corpus-prepare-button" disabled={batchJobLocked || corpusJobLocked} onClick={() => void preparePublicCorpus()}>{corpusJobLocked ? "Preparing frozen snapshot…" : corpusJob?.status === "completed" ? "Snapshot ready" : "Prepare frozen snapshot"}</button>
                </>}
                <label><span>Tokenizer</span><select disabled={batchJobLocked || corpusJobLocked || batchCampaign === "real_text_scaling_ladder"} value={tokenizer} onChange={(event) => { const next = event.target.value as TokenizerId; setTokenizer(next); setCorpusJob(null); setCorpusError(null); setTokenStreamManifestPath(""); setPretrainingTargetLoss(next === "byte_v1" ? (targetDevice === "cuda" ? 3 : 5.4) : 9.4); markBatchCampaignEdited(); }}>{batchCampaign !== "real_text_scaling_ladder" && <option value="byte_v1">UTF-8 bytes · built in</option>}<option value="olmo2_1124">OLMo 2 · immutable revision</option>{batchCampaign !== "real_text_scaling_ladder" && corpusChoice === "local" && <><option value="uint16_bin_v1">Legacy uint16 stream · unpinned</option><option value="uint32_bin_v1">Legacy uint32 stream · unpinned</option></>}</select></label>
                {pinnedTokenizer && <div className="fixed-callout"><b>{pinnedTokenizer.kind === "pinned_remote" ? "Immutable tokenizer contract" : "Built-in tokenizer contract"}</b><span>{pinnedTokenizer.repository ? `${pinnedTokenizer.repository} · revision ${pinnedTokenizer.revision?.slice(0, 12)} · ` : ""}{formatNumber(pinnedTokenizer.vocab_size)} tokens · definition <code>{pinnedTokenizer.definition_fingerprint.slice(0, 12)}</code></span></div>}
                {batchCampaign === "standard_pretraining_census" && <>
                  <label><span>Target validation loss</span><input disabled={batchJobLocked} type="number" min="0.01" step="0.1" value={pretrainingTargetLoss} onChange={(event) => { setPretrainingTargetLoss(Number(event.target.value)); markBatchCampaignEdited(); }} /></label>
                  <label><span>Validation cadence</span><input disabled={batchJobLocked} type="number" min="1" step="1" value={pretrainingValidationInterval} onChange={(event) => { setPretrainingValidationInterval(Math.max(1, Math.round(Number(event.target.value)))); markBatchCampaignEdited(); }} /></label>
                </>}
                {corpusJobLocked && <div className="corpus-progress"><div className="progress-track"><span style={{ width: `${Math.min(100, 100 * corpusJob.progress.completed / Math.max(1, corpusJob.progress.total))}%` }} /></div><small>{corpusJob.progress.message}</small></div>}
                {corpusJob?.status === "completed" && corpusJob.result && <div className="corpus-provenance"><span><b>{formatNumber(corpusJob.result.training_tokens)}</b> training tokens</span><span><b>{formatNumber(corpusJob.result.validation_tokens)}</b> held-out tokens</span><span><b>{corpusJob.result.corpus_fingerprint.slice(0, 12)}</b> content fingerprint</span><span><b>{corpusJob.result.tokenizer_fingerprint.slice(0, 12)}</b> tokenizer fingerprint</span><span><b>{corpusJob.result.dataset_identity_fingerprint.slice(0, 12)}</b> combined identity</span><span><b>{corpusJob.result.source.revision.slice(0, 12)}</b> source revision</span><small>{corpusJob.result.source.dataset} · {corpusJob.result.source.config} · {corpusJob.result.source.license} · disjoint source rows</small></div>}
                {corpusJob?.status === "failed" && <p className="validation-error campaign-error">{corpusJob.error}</p>}
                {corpusError && <p className="validation-error campaign-error">{corpusError}</p>}
              </div>
            )}
            {batchCampaign === "horizon_transfer" && (
              <div className="dataset-contract horizon-contract">
                <label><span>Architecture contract</span><select disabled={batchJobLocked} value={horizonParameterization} onChange={(event) => { const next = event.target.value as HorizonParameterization; setHorizonParameterization(next); setHorizonBatchExamples(targetDevice === "cuda" ? (next === "jiang_chizat" ? 16 : 8) : 2); markBatchCampaignEdited(); }}><option value="jiang_chizat">Jiang MHSA + Chizat FFN</option><option value="nugpt">νGPT normalized Transformer</option></select></label>
                <label><span>Presented-token horizons T</span><input disabled={batchJobLocked} value={horizonValues} onChange={(event) => { setHorizonValues(event.target.value); markBatchCampaignEdited(); }} /></label>
                <label><span>Peak-LR grid</span><input disabled={batchJobLocked} value={horizonRateGrid} onChange={(event) => { setHorizonRateGrid(event.target.value); markBatchCampaignEdited(); }} /></label>
                <label><span>Batch examples</span><input disabled={batchJobLocked} type="number" min="1" value={horizonBatchExamples} onChange={(event) => { setHorizonBatchExamples(Math.max(1, Math.round(Number(event.target.value)))); markBatchCampaignEdited(); }} /></label>
                {horizonParameterization === "nugpt" ? <div className="schedule-picker"><span>Schedule families</span>{(["cosine_to_10_percent", "linear_warmup_decay_to_zero", "wsd"] as HorizonSchedule[]).map((schedule) => <label key={schedule}><input disabled={batchJobLocked} type="checkbox" checked={horizonSchedules.includes(schedule)} onChange={(event) => { setHorizonSchedules((current) => event.target.checked ? [...current, schedule] : current.filter((item) => item !== schedule)); markBatchCampaignEdited(); }} /><b>{schedule.replaceAll("_", " ")}</b></label>)}</div> : <div className="fixed-callout"><b>Source-faithful schedule</b><span>50% linear warmup then constant · group multipliers frozen from the completed FineWeb reference calibration</span></div>}
                <div className="fixed-callout"><b>Frozen real-text holdout</b><span>Corpus and sampled windows fixed · N, U, and B fixed · T varies · largest T stays hidden until every schedule/LR rule is frozen</span></div>
              </div>
            )}
            {batchCampaign === "real_text_scaling_ladder" && (
              <div className="dataset-contract horizon-contract forecast-contract">
                <label><span>Architecture contract</span><select disabled={batchJobLocked} value={horizonParameterization} onChange={(event) => { const next = event.target.value as HorizonParameterization; setHorizonParameterization(next); if (next === "nugpt" && distributedMode === "fsdp") setDistributedMode("ddp"); markBatchCampaignEdited(); }}><option value="jiang_chizat">Jiang MHSA + Chizat FFN</option><option value="nugpt">νGPT normalized Transformer</option></select></label>
                <label><span>Ladder parameter targets</span><input disabled={batchJobLocked} value={forecastTargets} onChange={(event) => { setForecastTargets(event.target.value); markBatchCampaignEdited(); }} /></label>
                <label><span>Depth at each rung</span><input disabled={batchJobLocked} value={forecastDepths} onChange={(event) => { setForecastDepths(event.target.value); markBatchCampaignEdited(); }} /></label>
                <label><span>Constant tokens / parameter</span><input disabled={batchJobLocked} type="number" min="0.1" step="1" value={forecastTokensPerParameter} onChange={(event) => { setForecastTokensPerParameter(Math.max(0.1, Number(event.target.value))); markBatchCampaignEdited(); }} /></label>
                <label><span>Context length</span><input disabled={batchJobLocked} type="number" min="8" step="8" value={forecastContextLength} onChange={(event) => { setForecastContextLength(Math.max(8, Math.round(Number(event.target.value)))); markBatchCampaignEdited(); }} /></label>
                <label><span>Global batch examples</span><input disabled={batchJobLocked} type="number" min="1" value={forecastBatchExamples} onChange={(event) => { setForecastBatchExamples(Math.max(1, Math.round(Number(event.target.value)))); markBatchCampaignEdited(); }} /></label>
                <label><span>Peak-LR tuning grid</span><input disabled={batchJobLocked} value={horizonRateGrid} onChange={(event) => { setHorizonRateGrid(event.target.value); markBatchCampaignEdited(); }} /></label>
                <label><span>Forecast parameter targets</span><input disabled={batchJobLocked} value={forecastPredictionTargets} onChange={(event) => { setForecastPredictionTargets(event.target.value); markBatchCampaignEdited(); }} /></label>
                <div className="fixed-callout"><b>Frozen scientific coordinate</b><span>One pinned tokenizer and token stream · exact vocab-aware parameter counts · constant T/P · one hidden upper rung · competing loss-law families · predictions withheld when any gate fails</span></div>
                {horizonParameterization === "nugpt" && <div className="fixed-callout"><b>Definition-preserving parallelism</b><span>νGPT uses one GPU or replicated DDP so the full post-step matrix projection remains exact. FSDP is deliberately refused.</span></div>}
              </div>
            )}
            {batchCampaign === "joint_horizon_batch" && (
              <div className="dataset-contract horizon-contract joint-contract">
                <label><span>Fit horizons T</span><input disabled={batchJobLocked} value={jointFitHorizons} onChange={(event) => { setJointFitHorizons(event.target.value); markBatchCampaignEdited(); }} /></label>
                <label><span>Held-out horizon T*</span><input disabled={batchJobLocked} type="number" min="1" value={jointHeldoutHorizon} onChange={(event) => { setJointHeldoutHorizon(Math.max(1, Math.round(Number(event.target.value)))); markBatchCampaignEdited(); }} /></label>
                <label><span>Fit batches (examples)</span><input disabled={batchJobLocked} value={jointFitBatches} onChange={(event) => { setJointFitBatches(event.target.value); markBatchCampaignEdited(); }} /></label>
                <label><span>Held-out batch B*</span><input disabled={batchJobLocked} type="number" min="1" value={jointHeldoutBatch} onChange={(event) => { setJointHeldoutBatch(Math.max(1, Math.round(Number(event.target.value)))); markBatchCampaignEdited(); }} /></label>
                <label><span>Peak-LR grid</span><input disabled={batchJobLocked} value={horizonRateGrid} onChange={(event) => { setHorizonRateGrid(event.target.value); markBatchCampaignEdited(); }} /></label>
                <label><span>Frozen schedule</span><select disabled={batchJobLocked} value={jointSchedule} onChange={(event) => { setJointSchedule(event.target.value as HorizonSchedule); markBatchCampaignEdited(); }}><option value="cosine_to_10_percent">Cosine to 10%</option><option value="linear_warmup_decay_to_zero">10% warmup + linear decay</option><option value="wsd">Warmup-stable-decay</option></select></label>
                <div className="fixed-callout"><b>Two-stage holdout</b><span>Fit only the T and B axes · filter rules on the unseen fit-rectangle corner · test once at larger T* and B*</span></div>
              </div>
            )}
            <div className="campaign-summary">
              <div><small>Model contract</small><strong>{batchModelContract}</strong><span>{batchModelDetail}</span></div>
              <div><small>Execution</small><strong>{batchExecution}</strong><span>{batchExecutionDetail}</span></div>
              <button className="run-button campaign-run" disabled={batchJobLocked || !standardRuntimeValid} onClick={startBatchCampaign}>{batchJobLocked ? "Campaign running" : batchJob?.status === "completed" ? "Run or resume" : "Launch campaign"}<span>→</span></button>
            </div>
            <p className="campaign-location-note">CUDA, DDP, and FSDP execute on the machine hosting the compute service. Remote-cluster dispatch is not implied by this browser control.</p>
            {!standardRuntimeValid && <p className="validation-error campaign-error">{corpusRequired && corpusChoice !== "local" && corpusJob?.status !== "completed" ? "Prepare the frozen public-corpus snapshot before launching training." : batchCampaign === "real_text_scaling_ladder" ? "Select A100/CUDA and provide a verified token stream; at least six increasing targets with matching depths; larger forecast targets; an increasing LR grid; and a batch divisible by GPUs × accumulation. νGPT does not permit FSDP." : batchCampaign === "horizon_transfer" ? "Provide both corpus paths, at least four increasing divisible horizons, three learning rates, one schedule, and a positive batch." : batchCampaign === "joint_horizon_batch" ? "Provide three increasing fit horizons and batches, larger held-out values, three learning rates, and divisible T/B geometry." : "Provide both corpus paths, a positive target and cadence. FlashAttention requires BF16 on CUDA; distributed runs require at least two CUDA processes."}</p>}
            {batchJobError && <p className="validation-error campaign-error">{batchJobError}</p>}
            {batchJob && (
              <div className={`campaign-status ${batchJob.status}`}>
                <div className="campaign-status-head"><span className={`status-badge ${batchJob.status}`}>{batchJob.status}</span><code>{batchJob.id}</code><strong>{batchJob.progress.message}</strong></div>
                {batchJobLocked && <div className="progress-track"><span style={{ width: `${batchProgressPercent}%` }} /></div>}
                {batchJobLocked && <small>{batchJob.progress.completed} / {batchJob.progress.total || "?"} grid trials · cached trials resume automatically</small>}
                {batchJob.status === "failed" && <p>{batchJob.error}</p>}
                {batchJob.status === "interrupted" && <p>{batchJob.error} Launch again to continue from its trial cache.</p>}
                {batchJob.result?.dataset && <div className="campaign-data-result"><span><b>{formatNumber(batchJob.result.dataset.training_tokens)}</b> training tokens</span><span><b>{formatNumber(batchJob.result.dataset.validation_tokens)}</b> validation tokens</span><span><b>{(batchJob.result.dataset.tokenizer ?? batchJob.result.dataset.tokenizer_id ?? "unknown").replaceAll("_", " ")}</b> tokenizer</span><span><b>{batchJob.result.dataset.fingerprint.slice(0, 12)}</b> content fingerprint</span>{batchJob.result.dataset.tokenizer_fingerprint && <span><b>{batchJob.result.dataset.tokenizer_fingerprint.slice(0, 12)}</b> tokenizer fingerprint</span>}{batchJob.result.dataset.identity_fingerprint && <span><b>{batchJob.result.dataset.identity_fingerprint.slice(0, 12)}</b> combined identity</span>}</div>}
                {batchJob.result?.campaign === "real_text_scaling_ladder" && batchJob.result.scales && (
                  <div className="tpp-result horizon-result forecast-result">
                    <div className={`tpp-verdict ${batchJob.result.forecastable ? "qualified" : "refused"}`}>
                      <span><small>Forecast verdict</small><strong>{batchJob.result.forecastable ? "Bounded forecasts certified" : "Predictions withheld"}</strong></span>
                      <span><small>Observed rungs</small><strong>{batchJob.result.scales.length}</strong></span>
                      <span><small>Hidden backtests</small><strong>{batchJob.result.hidden_scale_backtests?.filter((row) => row.passed).length ?? 0}/{batchJob.result.hidden_scale_backtests?.length ?? 0} passed</strong></span>
                      <span><small>Certified targets</small><strong>{batchJob.result.forecasts?.filter((row) => row.certified).length ?? 0}/{batchJob.result.forecasts?.length ?? 0}</strong></span>
                    </div>
                    {!batchJob.result.forecastable && <p>{batchJob.result.refusal_reasons?.join(" · ")}</p>}
                    <div className="tpp-geometry">{batchJob.result.scales.map((row) => <span key={row.name}><strong>{row.name}{row.heldout ? " · hidden" : ""}</strong><small>{formatNumber(row.parameters)} parameters · {formatNumber(row.presented_tokens)} tokens · {(row.repetition_ratio ?? 0).toFixed(2)}× stream</small><b>{formatLoss(row.mean_validation_loss)} loss · {row.tokens_per_parameter.toFixed(2)} T/P</b></span>)}</div>
                    {batchJob.result.hidden_scale_backtests && batchJob.result.hidden_scale_backtests.length > 0 && <div className="tpp-rule-list">{batchJob.result.hidden_scale_backtests.map((row) => <div key={row.scale}><strong>{row.scale} hidden-rung prediction</strong><span>predicted {formatLoss(row.predicted_loss)} · observed {formatLoss(row.observed_loss)}</span><b className={row.passed ? "qualified" : "refused"}>{(row.relative_error * 100).toFixed(2)}% error</b></div>)}</div>}
                    <div className="joint-stage-label"><strong>Bounded extrapolations</strong><span>Median of qualified families; interval includes family and seed uncertainty</span></div>
                    <div className="tpp-rule-list">{batchJob.result.forecasts?.map((row) => <div key={row.target_size}><strong>{formatNumber(row.target_size)} parameters</strong><span>{row.certified && row.prediction !== null ? `loss ${formatLoss(row.prediction)} · 95% ${row.prediction_interval_95 ? `${formatLoss(row.prediction_interval_95[0])}–${formatLoss(row.prediction_interval_95[1])}` : "interval unavailable"}` : row.refusal_reasons.join(" · ")}</span><b className={row.certified ? "qualified" : "refused"}>{row.certified ? `${row.extrapolation_factor.toFixed(1)}× certified` : "withheld"}</b></div>)}</div>
                  </div>
                )}
                {batchAnalyses.length > 0 && <div className="campaign-analysis-list">{batchAnalyses.map((analysis) => <div key={`${analysis.scale.name}-${analysis.optimizer}`}><span><strong>{analysis.scale.name}</strong><small>{analysis.optimizer}</small></span><b className={analysis.consensus.qualified ? "qualified" : "refused"}>{analysis.consensus.qualified && analysis.consensus.critical_batch_tokens ? `${formatNumber(analysis.consensus.critical_batch_tokens)} tokens` : "Withheld"}</b><small>{analysis.consensus.qualified ? "Estimator consensus" : analysis.consensus.refusal_reasons.join(" · ")}</small></div>)}</div>}
                {batchJob.result?.geometry && batchJob.result.campaign === "constant_tokens_per_parameter_heldout_transfer" && (
                  <div className="tpp-result">
                    <div className={`tpp-verdict ${batchJob.result.fit_qualification?.source_optimum_is_interior ? "qualified" : "refused"}`}>
                      <span><small>Constant T/P qualification</small><strong>{batchJob.result.fit_qualification?.source_optimum_is_interior ? "Fit range qualified" : "Recommendation withheld"}</strong></span>
                      <span><small>Realized spread</small><strong>{batchJob.result.tpp_spread_ratio?.toFixed(3)}×</strong></span>
                      <span><small>Fitted horizon exponent</small><strong>{batchJob.result.fitted_horizon_exponent?.toFixed(4)}</strong></span>
                      <span><small>Held-out oracle</small><strong>{batchJob.result.heldout_oracle ? formatLoss(batchJob.result.heldout_oracle.mean_loss) : "—"}</strong></span>
                    </div>
                    {!batchJob.result.fit_qualification?.source_optimum_is_interior && <p>The source optimum lies on the tested learning-rate boundary. Regret is reported for diagnosis, but no transfer rule is recommended.</p>}
                    <div className="tpp-geometry">{batchJob.result.geometry.map((row) => <span key={row.scale!.name}><strong>{row.scale!.name}</strong><small>{formatNumber(row.parameters)} parameters · {formatNumber(row.total_tokens!)} tokens</small><b>{row.realized_tpp!.toFixed(3)} T/P</b></span>)}</div>
                    <div className="tpp-rule-list">{batchJob.result.transfer_results?.map((row) => <div key={row.rule}><strong>{row.rule.replaceAll("_", " ")}</strong><span>{row.evaluated && row.mean_heldout_loss !== undefined ? formatLoss(row.mean_heldout_loss) : "Not evaluated"}</span><b className={row.recommendable ? "qualified" : "refused"}>{row.relative_regret !== undefined ? `${row.relative_regret >= 0 ? "+" : ""}${(row.relative_regret * 100).toFixed(1)}% regret` : row.refusal_reasons.join(" · ")}</b></div>)}</div>
                  </div>
                )}
                {batchJob.result?.campaign === "horizon_transfer" && batchJob.result.geometry && (
                  <div className="tpp-result horizon-result">
                    <div className={`tpp-verdict ${batchJob.result.recommendation ? "qualified" : "refused"}`}>
                      <span><small>Held-out horizon verdict</small><strong>{batchJob.result.recommendation ? "Transfer rule certified" : "Recommendation withheld"}</strong></span>
                      <span><small>Held-out T</small><strong>{formatNumber(batchJob.result.heldout_horizon ?? 0)}</strong></span>
                      <span><small>Fit horizon span</small><strong>{batchJob.result.fit_horizon_span_ratio?.toFixed(1)}×</strong></span>
                      <span><small>Best frozen rule</small><strong>{batchJob.result.recommendation ? `${batchJob.result.recommendation.rule.replaceAll("_", " ")} · β ${batchJob.result.recommendation.exponent.toFixed(3)}` : "—"}</strong></span>
                    </div>
                    {!batchJob.result.recommendation && <p>{batchJob.result.refusal_reasons?.join(" · ")}</p>}
                    <div className="tpp-geometry">{batchJob.result.geometry.map((row) => <span key={row.presented_tokens}><strong>T {formatNumber(row.presented_tokens ?? 0)}</strong><small>U {formatNumber(row.unique_tokens ?? 0)} · B {formatNumber(row.batch_tokens)} · S {formatNumber(row.optimizer_steps ?? 0)}</small><b>{(row.tokens_per_parameter ?? 0).toFixed(3)} T/P · {(row.presented_to_unique_token_ratio ?? 0).toFixed(2)}× repeat</b></span>)}</div>
                    {batchJob.result.schedule_analyses?.map((analysis) => <div className="horizon-schedule-card" key={analysis.schedule_name}>
                      <div><strong>{analysis.schedule_name.replaceAll("_", " ")}</strong><span className={analysis.fit_qualified ? "qualified" : "refused"}>{analysis.fit_qualified ? `β ${analysis.fitted_power_law.exponent.toFixed(3)} · R² ${analysis.fitted_power_law.r_squared.toFixed(3)}` : "Fit refused"}</span></div>
                      {!analysis.fit_qualified && <small>{analysis.fit_refusal_reasons.join(" · ")}</small>}
                      <div className="tpp-rule-list">{analysis.frozen_rule_results.map((row) => <div key={row.rule}><strong>{row.rule.replaceAll("_", " ")}</strong><span>η {row.predicted_peak_learning_rate.toExponential(2)} · loss {formatLoss(row.mean_heldout_loss)} · {row.mechanism_discrimination_certified ? "mechanism pass" : row.transfer_certified ? "non-inferior" : "refused"}</span><b className={row.transfer_certified ? "qualified" : "refused"}>{`${row.relative_oracle_regret >= 0 ? "+" : ""}${(row.relative_oracle_regret * 100).toFixed(2)}% oracle regret`}</b></div>)}</div>
                    </div>)}
                  </div>
                )}
                {batchJob.result?.campaign === "joint_horizon_batch_transfer" && batchJob.result.geometry && (
                  <div className="tpp-result horizon-result joint-result">
                    <div className={`tpp-verdict ${batchJob.result.joint_transfer_settled ? "qualified" : "refused"}`}>
                      <span><small>Joint-transfer verdict</small><strong>{batchJob.result.joint_transfer_settled ? "Empirical transfer + mechanism settled" : batchJob.result.joint_recommendation ? "Non-inferior; mechanism unresolved" : "Recommendation withheld"}</strong></span>
                      <span><small>Fitted horizon β</small><strong>{batchJob.result.axis_fit_qualification?.horizon_exponent.toFixed(3) ?? "—"}</strong></span>
                      <span><small>Fitted batch γ</small><strong>{batchJob.result.axis_fit_qualification?.batch_exponent.toFixed(3) ?? "—"}</strong></span>
                      <span><small>Best frozen rule</small><strong>{batchJob.result.joint_recommendation?.rule.replaceAll("_", " ") ?? "—"}</strong></span>
                    </div>
                    {!batchJob.result.axis_fit_qualification?.qualified && <p>{batchJob.result.axis_fit_qualification?.refusal_reasons.join(" · ")}</p>}
                    <div className="tpp-geometry">{batchJob.result.geometry.map((row) => <span key={`${row.role}-${row.presented_tokens}-${row.batch_tokens}`}><strong>{row.role?.replaceAll("_", " ")}</strong><small>T {formatNumber(row.presented_tokens ?? 0)} · B {formatNumber(row.batch_tokens)} · S {formatNumber(row.optimizer_steps ?? 0)}</small><b>{(row.tokens_per_parameter ?? 0).toFixed(3)} T/P</b></span>)}</div>
                    <div className="joint-stage-label"><strong>Composition cross-check</strong><span>T {formatNumber(batchJob.result.composition_crosscheck?.presented_tokens ?? 0)} · B {formatNumber(batchJob.result.composition_crosscheck?.batch_tokens ?? 0)}</span></div>
                    <div className="tpp-rule-list">{batchJob.result.composition_crosscheck?.candidate_results.filter((row) => row.joint_rule).map((row) => <div key={row.rule}><strong>{row.rule.replaceAll("_", " ")}</strong><span>{row.evaluated && row.mean_loss !== undefined ? `loss ${formatLoss(row.mean_loss)}` : row.refusal_reasons.join(" · ")}</span><b className={row.composition_crosscheck_passed ? "qualified" : "refused"}>{row.relative_oracle_regret !== null && row.relative_oracle_regret !== undefined ? `${(row.relative_oracle_regret * 100).toFixed(2)}% regret` : "not evaluated"}</b></div>)}</div>
                    <div className="joint-stage-label"><strong>Doubly held-out corner</strong><span>T {formatNumber(batchJob.result.heldout_corner?.presented_tokens ?? 0)} · B {formatNumber(batchJob.result.heldout_corner?.batch_tokens ?? 0)} · {batchJob.result.heldout_corner?.composition_identifiable ? "joint effect identifiable" : "flat versus partial controls"}</span></div>
                    <div className="tpp-rule-list">{batchJob.result.heldout_corner?.candidate_results.map((row) => <div key={row.rule}><strong>{row.rule.replaceAll("_", " ")}</strong><span>{row.evaluated && row.mean_loss !== undefined ? `η ${row.optimizer?.learning_rate.toExponential(2)} · β (${row.optimizer?.beta1.toFixed(3)}, ${row.optimizer?.beta2.toFixed(3)}) · ε ${row.optimizer?.epsilon.toExponential(1)} · groups ${row.peak_parameter_group_contract?.map((group) => `${group.name}:${group.peak_learning_rate.toExponential(1)}`).join(", ")} · loss ${formatLoss(row.mean_loss)} · ${row.mechanism_discrimination_certified ? `mechanism pass · ${row.theory_transfer_certified ? "theory regime qualified" : "empirical only"}` : row.transfer_certified ? "non-inferior" : "refused"}` : row.refusal_reasons.join(" · ")}</span><b className={row.transfer_certified ? "qualified" : "refused"}>{row.relative_oracle_regret !== null && row.relative_oracle_regret !== undefined ? `${(row.relative_oracle_regret * 100).toFixed(2)}% oracle regret` : "not evaluated"}</b></div>)}</div>
                  </div>
                )}
              </div>
            )}
          </div>
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
              <button disabled={studyLocked || blockType === "pre_norm_moe" || blockType === "normalized_transformer"} className={optimizer === "sgd" ? "active" : ""} onClick={() => { setOptimizer("sgd"); setBatchRule("sgd_linear_batch"); setBatchTransfer(null); markProfileEdited(); }}><strong>SGD</strong><small>Momentum 0</small></button>
              <button disabled={studyLocked} className={optimizer === "adam" ? "active" : ""} onClick={() => { setOptimizer("adam"); setBatchRule("complete_dp_joint"); setBatchTransfer(null); markProfileEdited(); }}><strong>Adam</strong><small>{blockType === "normalized_transformer" ? "β₂ .95" : "β₂ .999"}</small></button>
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

      <section className="study-section section-shell batch-section" id="batch">
        <div className="section-heading">
          <div><p className="eyebrow">Batch scaling</p><h2>Qualify the transition before changing the schedule.</h2></div>
          <p>Every rule is inspectable. Three independent critical-batch assays must agree before the app can unlock a dynamic Seesaw schedule.</p>
        </div>

        <div className="setup-grid">
          <div className="panel data-panel">
            <div className="panel-title"><span>Transfer preview</span><small>Static rule registry</small></div>
            <div className="paired-inputs dense-inputs">
              <label><span>Target batch multiplier</span><input type="number" min="0.125" step="0.25" value={targetBatchMultiplier} onChange={(event) => { setTargetBatchMultiplier(Math.max(0.125, Number(event.target.value))); setBatchTransfer(null); }} /></label>
              <label><span>Target token-horizon multiplier</span><input type="number" min="0.125" step="0.25" value={targetHorizonMultiplier} onChange={(event) => { setTargetHorizonMultiplier(Math.max(0.125, Number(event.target.value))); setBatchTransfer(null); }} /></label>
              <label><span>Transfer rule</span><select value={batchRule} onChange={(event) => { setBatchRule(event.target.value); setBatchTransfer(null); }}>{batchRuleOptions.map((rule) => <option key={rule} value={rule}>{rule.replaceAll("_", " ")}</option>)}</select></label>
              <label><span>Normalized-time ratio q</span><input readOnly value={(targetBatchMultiplier / targetHorizonMultiplier).toPrecision(4)} /></label>
            </div>
            <div className="batch-shortcuts">
              <button onClick={() => { setTargetHorizonMultiplier(parameterMultiplier); setBatchTransfer(null); }}>Set constant T/P</button>
              <button onClick={() => { setTargetBatchMultiplier(1); setTargetHorizonMultiplier(1); setBatchTransfer(null); }}>Reset ratios</button>
              <button className="run-button compact-run" onClick={previewBatchTransfer}>Preview rule <span>→</span></button>
            </div>
            <div className="fixed-callout"><b>Canonical geometry</b><span>{formatNumber(baseBatchTokens)} base batch tokens · {parameterMultiplier.toFixed(2)}× parameter span · T/P target {(tokenBudget * targetHorizonMultiplier / totalParameters).toPrecision(3)}</span></div>
            {batchTransferError && <p className="validation-error">{batchTransferError}</p>}
            {batchTransfer && (
              <div className={`batch-rule-result ${batchTransfer.valid ? "accepted" : "refused"}`}>
                <strong>{batchTransfer.valid ? "Rule is algebraically valid" : "Rule refused"}</strong>
                {batchTransfer.target ? <span>η {batchTransfer.target.learning_rate.toExponential(3)} · β₁ {batchTransfer.target.beta1.toPrecision(5)} · β₂ {batchTransfer.target.beta2.toPrecision(6)} · ε {batchTransfer.target.epsilon.toExponential(2)}</span> : <span>{batchTransfer.refusal_reasons.join(" · ")}</span>}
                <small>{batchTransfer.assumptions.join(" ")}</small>
              </div>
            )}
          </div>

          <aside className="panel protocol-panel">
            <div className="panel-title"><span>Qualification gate</span><small>Seesaw locked</small></div>
            <div className="batch-estimators">
              <div><b>1</b><span><strong>Steps to target</strong><small>Fit S(B) = a + b/B; require a bracketed 20% transition.</small></span><em>Required</em></div>
              <div><b>2</b><span><strong>Checkpoint fork</strong><small>Continue one checkpoint at matched tokens across the batch sweep.</small></span><em>Required</em></div>
              <div><b>3</b><span><strong>Gradient noise</strong><small>Debiased trace covariance over squared mean-gradient norm.</small></span><em>Required</em></div>
            </div>
            <div className="batch-gate"><span>🔒</span><div><strong>Dynamic Seesaw schedule</strong><small>Unlocks only when two or more estimators qualify, agree within 2×, and late training is variance dominated.</small></div></div>
          </aside>
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
              {result.pilot_readiness ? (
                <div className={`readiness-card ${result.pilot_readiness.ready ? "ready" : "needs-work"}`}>
                  <div><small>Power-law readiness</small><strong>{result.pilot_readiness.ready ? "Ready for the larger campaign" : "Pilot recommends another pass"}</strong></div>
                  <div className="readiness-metrics"><span><b>{result.pilot_readiness.parameter_span_ratio.toFixed(1)}×</b> parameter span</span><span><b>{result.pilot_readiness.dynamic_range_to_noise.toFixed(1)}×</b> signal / noise</span><span><b>{Math.round(result.pilot_readiness.monotone_transition_fraction * 100)}%</b> decreasing transitions</span></div>
                  {result.pilot_readiness.recommendations.length > 0 && <ul>{result.pilot_readiness.recommendations.map((recommendation) => <li key={recommendation}>{recommendation}</li>)}</ul>}
                  <p>Suggested next level: D {result.pilot_readiness.suggested_next_scale.width} · L/R {result.pilot_readiness.suggested_next_scale.repeats} · {formatNumber(result.pilot_readiness.suggested_next_training_points)} training examples</p>
                </div>
              ) : (
                <div className="readiness-card needs-work"><div><small>Power-law readiness</small><strong>Legacy campaign · readiness audit not recorded</strong></div></div>
              )}
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

      <section className="evidence-section section-shell" id="evidence">
        <div className="section-heading evidence-heading">
          <div><p className="eyebrow">Validation atlas</p><h2>Transfer success and scaling-law quality are separate claims.</h2></div>
          <p>These are observed campaign outcomes, including the refusals. A good one-step prediction does not earn an extrapolation unless the loss law, negative control, and uncertainty gates also pass.</p>
        </div>

        <div className="atlas-summary">
          <div><small>Published campaigns</small><strong>{PUBLISHED_EVIDENCE.length}</strong><span>Dense, sparse, normalized</span></div>
          <div><small>Recorded trials</small><strong>{PUBLISHED_EVIDENCE.reduce((sum, item) => sum + item.trials, 0)}</strong><span>Paired seeds and holdouts</span></div>
          <div><small>HP transfer</small><strong>{PUBLISHED_EVIDENCE.filter((item) => item.transferAccepted).length}/{PUBLISHED_EVIDENCE.length}</strong><span>Largest-scale probes accepted</span></div>
          <div><small>Forecasts issued</small><strong>{PUBLISHED_EVIDENCE.filter((item) => item.forecastable).length}/{PUBLISHED_EVIDENCE.length}</strong><span>Strict refusal contract</span></div>
        </div>

        <div className="evidence-grid">
          {PUBLISHED_EVIDENCE.map((item) => (
            <article className={`evidence-case ${item.forecastable ? "forecast" : "withheld"}`} key={item.id}>
              <div className="evidence-case-head">
                <div><small>{item.hardware} · {item.trials} trials</small><h3>{item.title}</h3></div>
                <span>{item.forecastable ? "Forecast issued" : "Forecast withheld"}</span>
              </div>
              <p>{item.conclusion}</p>
              <dl>
                <div><dt>Architecture</dt><dd>{item.architecture}</dd></div>
                <div><dt>Data</dt><dd>{item.data}</dd></div>
                <div><dt>Scale path</dt><dd>{item.scalePath}</dd></div>
                <div><dt>Optimizer</dt><dd>{item.optimizer}</dd></div>
              </dl>
              <div className="evidence-metrics">
                <div><small>normalized η</small><strong>{item.eta.toPrecision(3)}</strong></div>
                <div><small>holdout error</small><strong>{(item.holdoutError * 100).toFixed(1)}%</strong></div>
                <div><small>R²</small><strong>{item.r2.toFixed(3)}</strong></div>
                <div><small>α</small><strong>{item.exponent.toFixed(3)}</strong></div>
              </div>
              <div className="gate-strip">
                <span className={item.transferAccepted ? "pass" : "fail"}>HP transfer {item.transferAccepted ? "pass" : "fail"}</span>
                <span className={item.negativeControlRejected ? "pass" : "fail"}>Negative control {item.negativeControlRejected ? "separated" : "not separated"}</span>
              </div>
              <code>fingerprint prefix {item.fingerprint}</code>
            </article>
          ))}
        </div>

        <div className="web-snapshot">
          <div className="ledger-heading">
            <div><p className="eyebrow">Observed UI walkthroughs</p><h3>{WEB_UI_EVIDENCE.length} workflows executed end-to-end through the web app</h3></div>
            <span>2026-08-10 · localhost compute · <a href="/evidence/autoscaler-validation.json" download>evidence JSON</a></span>
          </div>
          <div className="snapshot-table" role="table" aria-label="Observed web workflow evidence">
            <div className="snapshot-row snapshot-head" role="row"><span>Workflow</span><span>Coverage</span><span>Primary observation</span><span>Control</span><span>Outcome</span><span>Job</span></div>
            {WEB_UI_EVIDENCE.map((item) => <div className="snapshot-row" role="row" key={item.id}><strong>{item.workflow}</strong><span>{item.coverage}</span><span>{item.primary}</span><span>{item.secondary}</span><b>{item.verdict}</b><code>{item.id}</code></div>)}
          </div>
        </div>

        <div className="runtime-canary">
          <div className="runtime-canary-head">
            <div><p className="eyebrow">Frozen FineWeb-Edu horizon holdout</p><h3>Schedule transfer is architecture-specific</h3></div>
            <span>465 completed A100 trials</span>
          </div>
          <div className="runtime-canary-grid">
            <div><small>Presented-token span</small><strong>65k → 1.05M</strong><span>8× fit span · largest horizon hidden</span></div>
            <div><small>Normalized control</small><strong>351 trials</strong><span>cosine · warmup/decay · WSD</span></div>
            <div><small>Jiang + Chizat MHSA</small><strong>114 trials</strong><span>seven CompleteP LR groups</span></div>
            <div><small>Frozen data</small><strong>16,384 / 1,024</strong><span>train / validation windows · 3 seeds</span></div>
          </div>
          <div className="runtime-canary-verdict">
            <span className="verdict-mark pass">✓</span>
            <div><strong>Normalized Transformer: one-third rule certified on all three schedules</strong><p>Held-out regret is 0.40% for cosine, 0% for warmup/decay, and 0.20% for WSD. The corresponding flat controls are 5.2–9.5% above their scoring oracles, so the mechanism gate is identifiable and passes.</p></div>
          </div>
          <div className="runtime-canary-verdict">
            <span className="verdict-mark refuse">!</span>
            <div><strong>Jiang + Chizat MHSA: one-third duration transfer rejected</strong><p>The source-faithful half-warmup/constant run preserves all seven CompleteP group rules, yet T<sup>−1/3</sup> has 2.93% oracle regret. A flat peak LR is only 0.049% above oracle, so the app refuses a horizon-scaling claim; the fitted exponent is negative with R² 0.353 and is rejected too.</p></div>
          </div>
          <p className="runtime-provenance">Observed on A100 80 GB · corpus <code>666710b377c444e7</code> · byte_v1 · context 64 · held-out 1,048,576 presented tokens · seeds <code>11, 29, 47</code> · the historical 0.32 sensitivity point is grouped with one-third and removed from new defaults</p>
        </div>

        <div className="runtime-canary">
          <div className="runtime-canary-head">
            <div><p className="eyebrow">Frozen FineWeb-Edu transfer assay</p><h3>Jiang attention + Chizat FFN and full Jiang sparse MoE</h3></div>
            <span>429 full trials + 12 feature probes</span>
          </div>
          <div className="runtime-canary-grid">
            <div><small>Frozen corpus</small><strong>67.16M / 8.40M</strong><span>training / held-out byte tokens</span></div>
            <div><small>Transfer grid</small><strong>4 × 7 × 3</strong><span>scales × normalized η × paired seeds</span></div>
            <div><small>Base-selected η</small><strong>0.03 / 0.03</strong><span>dense / sparse MoE</span></div>
            <div><small>Worst oracle ratio</small><strong>1.000× / 1.000×</strong><span>fixed base η versus each scale optimum</span></div>
          </div>
          <div className="runtime-canary-verdict">
            <span className="verdict-mark pass">✓</span>
            <div><strong>Dense HP transfer certified · mechanism claim withheld</strong><p>Every scale chose η=0.03; fixed-rate progress has log-slope 0.023 and all seven parameter-group feature-velocity probes pass. None of four deliberately wrong controls was separated on this short byte-level task, so the app does not claim unique theoretical identification.</p></div>
          </div>
          <div className="runtime-canary-verdict">
            <span className="verdict-mark pass">✓</span>
            <div><strong>Sparse MoE HP transfer certified · mechanism claim withheld</strong><p>Every scale again chose η=0.03 and fixed-rate progress has log-slope 0.022. None of three wrong controls was separated; maximum routing-load deviation reaches 33.3% at the largest shape, which is recorded as a follow-up rather than hidden.</p></div>
          </div>
          <p className="runtime-provenance">Observed on A100 80 GB · FineWeb-Edu sample-10BT revision <code>87f09149ef47</code> · corpus <code>666710b377c444e7</code> · disjoint rows <code>0–14,291</code> and <code>5,000,000–5,001,948</code> · byte_v1 · context 64 · 300 steps · batch 16</p>
        </div>

        <div className="runtime-canary">
          <div className="runtime-canary-head">
            <div><p className="eyebrow">A100 runtime assay</p><h3>Real text · AdamW + Adam + SGD · BF16 · explicit FlashAttention</h3></div>
            <span>432 completed trials</span>
          </div>
          <div className="runtime-canary-grid">
            <div><small>Scale ladder</small><strong>121k → 835k</strong><span>parameters · 3 Transformer sizes</span></div>
            <div><small>Batch horizon</small><strong>256 → 2,048</strong><span>tokens/update · independently tuned</span></div>
            <div><small>Direct-checkpoint Bcrit</small><strong>724 / 362</strong><span>tokens · Adam(W) / SGD</span></div>
            <div><small>Gradient-noise Bcrit</small><strong>30 → 54</strong><span>tokens · rises with model size</span></div>
          </div>
          <div className="runtime-canary-verdict">
            <span className="verdict-mark refuse">!</span>
            <div><strong>No batch recommendation issued</strong><p>Adam and AdamW estimators disagree by 13.4–23.9× and SGD by 6.7–12.0×. Targets 4.8, 3.0, 2.8, and 2.5 expose early, quantized, and missing crossings; the new dynamic-range gate refuses flat curves instead of turning roundoff into evidence.</p></div>
          </div>
          <p className="runtime-provenance">Observed on A100 80 GB · CUDA · Torch SDPA Flash path · corpus <code>0e6fad5d74666d17</code> · AdamW targets <code>4.8</code>, <code>3.0</code>, <code>2.8</code>, <code>2.5</code> · Adam/SGD target <code>3.0</code></p>
        </div>

        <div className="run-ledger">
          <div className="ledger-heading">
            <div><p className="eyebrow">Web job ledger</p><h3>Workflows launched through this interface</h3></div>
            <button onClick={() => void refreshHistory()}>Refresh ledger</button>
          </div>
          {historyError && <p className="ledger-offline">{apiOnline === false ? "Connect the local compute service to load private run history." : historyError}</p>}
          <div className="ledger-columns">
            <div>
              <h4>Scaling studies <span>{studyHistory.length}</span></h4>
              {studyHistory.length === 0 ? <p className="ledger-empty">No persisted web studies are available on this compute service yet.</p> : studyHistory.map((item) => (
                <button className="ledger-row" key={item.id} onClick={() => void loadStudy(item.id)}>
                  <span className={`status-dot ${item.status}`} />
                  <span><strong>{item.name}</strong><small>{item.architecture?.replaceAll("_", " ")} · {item.optimizer} · {item.device}</small></span>
                  <span>{item.result_summary ? `${(item.result_summary.holdout_relative_error * 100).toFixed(1)}% holdout` : item.progress.phase.replaceAll("-", " ")}</span>
                  <code>{item.id}</code>
                </button>
              ))}
            </div>
            <div>
              <h4>Batch campaigns <span>{batchHistory.length}</span></h4>
              {batchHistory.length === 0 ? <p className="ledger-empty">No persisted batch campaigns are available on this compute service yet.</p> : batchHistory.map((item) => (
                <button className="ledger-row" key={item.id} onClick={() => void loadBatchCampaign(item.id)}>
                  <span className={`status-dot ${item.status}`} />
                  <span><strong>{item.campaign.replaceAll("_", " ")}</strong><small>{item.progress.completed} / {item.progress.total || "?"} trials</small></span>
                  <span>{item.result_summary ? item.campaign === "real_text_scaling_ladder" ? `${item.result_summary.certified_forecasts ?? 0}/${item.result_summary.forecast_count ?? 0} forecasts certified` : item.campaign === "constant_tpp" || item.campaign === "horizon_transfer" || item.campaign === "joint_horizon_batch" ? `${item.result_summary.recommendable_rules} rules qualified` : `${item.result_summary.qualified_analyses}/${item.result_summary.analysis_count} assays qualified` : item.status}</span>
                  <code>{item.id}</code>
                </button>
              ))}
            </div>
          </div>
        </div>
      </section>

      <footer><span>AI Theorist · Autoscaler v0.1</span><p>Typed architectures. Paired evidence. Refusal before false precision.</p><span>Largest plan level: {formatNumber(totalParameters)} parameters</span></footer>
    </main>
  );
}
