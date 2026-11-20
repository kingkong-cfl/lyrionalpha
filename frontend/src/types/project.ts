export type TrackType = "audio" | "stem" | "midi" | "mic_recording";
export type StemRole = "full_mix" | "vocals" | "drums" | "bass" | "other" | "custom";
export type RepaintMode =
  | "repaint_and_restem"
  | "single_track_repaint"
  | "add_instrument_layer"
  | "lego_layer";
export type JobStatusType = "queued" | "running" | "completed" | "failed";
export type EnhanceMode = "all" | "tags" | "lyrics";

export interface EqSettings {
  low_shelf_db: number;
  mid_gain_db: number;
  mid_freq_hz: number;
  high_shelf_db: number;
}

export interface CompressorSettings {
  enabled: boolean;
  threshold_db: number;
  ratio: number;
  attack_ms: number;
  release_ms: number;
}

export interface ReverbSettings {
  enabled: boolean;
  room_size: number;
  damping: number;
  wet_level: number;
}

export interface TrackDspChain {
  eq: EqSettings;
  compressor: CompressorSettings;
  reverb: ReverbSettings;
}

export interface TrackSchema {
  track_id: string;
  name: string;
  track_type: TrackType;
  stem_role: StemRole;
  parent_track_id: string | null;
  file_rel_path: string | null;
  midi_rel_path: string | null;
  revision: number;
  sample_rate: number;
  start_sample: number;
  end_sample: number;
  muted: boolean;
  soloed: boolean;
  volume_db: number;
  pan: number;
  dsp_chain: TrackDspChain;
  plugin_metadata: Record<string, unknown>;
}

export interface BeatGridSchema {
  sample_rate: number;
  total_samples: number;
  duration_sec: number;
  bpm: number;
  time_signature: string;
  musical_key: string;
  beat_timestamps: number[];
  beat_samples: number[];
  downbeat_beat_indices: number[];
}

export interface ProducerSpecSchema {
  raw_prompt: string;
  style_tags: string[];
  structured_lyrics: string;
  seed: number | null;
  guidance_scale: number;
  inference_steps: number;
  shift?: number;
}

export interface RepaintHistoryEntry {
  repaint_id: string;
  created_at: string;
  mode: RepaintMode;
  start_sample: number;
  end_sample: number;
  start_sec: number;
  end_sec: number;
  crossfade_samples: number;
  prompt_override: string | null;
  lyrics_override: string | null;
  target_track_id: string | null;
  previous_revision: number;
  new_revision: number;
  previous_files?: Record<string, string>;
  previous_track_revisions?: Record<string, number>;
}

export interface ProjectSchema {
  schema_version: string;
  project_id: string;
  name: string;
  created_at: string;
  updated_at: string;
  grid: BeatGridSchema;
  producer_spec: ProducerSpecSchema;
  tracks: TrackSchema[];
  repaint_history: RepaintHistoryEntry[];
}

export interface HealthResponse {
  status: "ok";
  mock_gpu: boolean;
  gpu_name: string;
  vram_used_mb: number;
  vram_total_mb: number;
  active_model: string;
  drive_root: string;
  drive_writable: boolean;
  queue_depth: number;
}

export interface ProducerEnhanceResponse {
  provider: string;
  mode?: EnhanceMode;
  bpm: number;
  musical_key: string;
  time_signature: string;
  duration_sec: number;
  style_tags: string[];
  structured_lyrics: string;
}

export interface JobCreatedResponse {
  job_id: string;
  project_id: string;
  job_type: string;
  status: JobStatusType;
}

export interface JobStatusResponse {
  job_id: string;
  project_id: string;
  job_type: string;
  status: JobStatusType;
  progress: number;
  stage: string;
  vram_allocated_mb: number;
  created_at: string;
  updated_at: string;
  result: Record<string, unknown> | null;
  error: string | null;
}

export type SnapMode = "bar" | "beat" | "off";

export interface RegionSelection {
  startSec: number;
  endSec: number;
  startSample: number;
  endSample: number;
  startLabel: string;
  endLabel: string;
}
