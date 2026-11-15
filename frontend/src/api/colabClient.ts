import type {
  HealthResponse,
  JobCreatedResponse,
  JobStatusResponse,
  ProducerEnhanceResponse,
  ProjectSchema,
  StemRole,
} from "../types/project";

const STORAGE_KEY = "lyrion_gpu_server_url";
const DEFAULT_SERVER_URL = "http://localhost:8000";

export function getSavedServerUrl(): string {
  const saved = localStorage.getItem(STORAGE_KEY);
  return saved && saved.trim() ? saved.trim().replace(/\/+$/, "") : DEFAULT_SERVER_URL;
}

export function setSavedServerUrl(url: string): string {
  const cleaned = url.trim().replace(/\/+$/, "") || DEFAULT_SERVER_URL;
  localStorage.setItem(STORAGE_KEY, cleaned);
  return cleaned;
}

async function apiFetch<T>(baseUrl: string, path: string, init?: RequestInit): Promise<T> {
  const url = `${baseUrl.replace(/\/+$/, "")}${path}`;
  const resp = await fetch(url, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers || {}),
    },
  });

  if (!resp.ok) {
    let detail = `${resp.status} ${resp.statusText}`;
    try {
      const data = await resp.json();
      if (data?.detail) detail = String(data.detail);
    } catch {
      // Ignore parse error on non-JSON error body
    }
    throw new Error(detail);
  }
  return (await resp.json()) as T;
}

export function buildTrackAudioUrl(
  baseUrl: string,
  projectId: string,
  fileRelPath: string,
  revision: number
): string {
  const cleanBase = baseUrl.replace(/\/+$/, "");
  return `${cleanBase}/api/v1/projects/${encodeURIComponent(projectId)}/audio/${fileRelPath}?rev=${revision}`;
}

export async function fetchHealth(baseUrl: string): Promise<{ health: HealthResponse; latencyMs: number }> {
  const t0 = performance.now();
  const health = await apiFetch<HealthResponse>(baseUrl, "/api/v1/health");
  return { health, latencyMs: Math.round(performance.now() - t0) };
}

export async function enhanceWithProducer(
  baseUrl: string,
  rawPrompt: string,
  preferredDurationSec?: number
): Promise<ProducerEnhanceResponse> {
  return apiFetch<ProducerEnhanceResponse>(baseUrl, "/api/v1/producer/enhance", {
    method: "POST",
    body: JSON.stringify({
      raw_prompt: rawPrompt,
      preferred_duration_sec: preferredDurationSec ?? null,
    }),
  });
}

export async function listProjects(baseUrl: string): Promise<ProjectSchema[]> {
  return apiFetch<ProjectSchema[]>(baseUrl, "/api/v1/projects");
}

export async function createProject(
  baseUrl: string,
  name: string,
  bpm = 120,
  musicalKey = "A minor",
  rawPrompt = ""
): Promise<ProjectSchema> {
  return apiFetch<ProjectSchema>(baseUrl, "/api/v1/projects", {
    method: "POST",
    body: JSON.stringify({
      name,
      bpm,
      musical_key: musicalKey,
      raw_prompt: rawPrompt,
    }),
  });
}

export async function getProject(baseUrl: string, projectId: string): Promise<ProjectSchema> {
  return apiFetch<ProjectSchema>(baseUrl, `/api/v1/projects/${encodeURIComponent(projectId)}`);
}

export async function saveProject(baseUrl: string, project: ProjectSchema): Promise<ProjectSchema> {
  return apiFetch<ProjectSchema>(baseUrl, `/api/v1/projects/${encodeURIComponent(project.project_id)}`, {
    method: "PUT",
    body: JSON.stringify({
      name: project.name,
      producer_spec: project.producer_spec,
      tracks: project.tracks,
    }),
  });
}

export async function triggerGenerateSong(
  baseUrl: string,
  projectId: string,
  payload: {
    raw_prompt: string;
    style_tags: string[];
    structured_lyrics: string;
    bpm: number;
    musical_key: string;
    duration_sec: number;
    seed: number;
    separate_stems: boolean;
  }
): Promise<JobCreatedResponse> {
  return apiFetch<JobCreatedResponse>(baseUrl, `/api/v1/projects/${encodeURIComponent(projectId)}/generate`, {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export async function triggerRepaintRegion(
  baseUrl: string,
  projectId: string,
  payload: {
    start_sample: number;
    end_sample: number;
    prompt_override?: string;
    lyrics_override?: string;
    style_tags?: string[];
    seed?: number;
    restem_window?: boolean;
  }
): Promise<JobCreatedResponse> {
  return apiFetch<JobCreatedResponse>(baseUrl, `/api/v1/projects/${encodeURIComponent(projectId)}/repaint`, {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export async function triggerLegoLayer(
  baseUrl: string,
  projectId: string,
  payload: {
    reference_track_id: string;
    target_stem_role: StemRole;
    start_sample: number;
    end_sample: number;
    prompt_override?: string;
    seed?: number;
  }
): Promise<JobCreatedResponse> {
  return apiFetch<JobCreatedResponse>(baseUrl, `/api/v1/projects/${encodeURIComponent(projectId)}/lego`, {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export async function triggerSingleTrackRepaint(
  baseUrl: string,
  projectId: string,
  payload: {
    target_track_id: string;
    start_sample: number;
    end_sample: number;
    prompt_override?: string;
    lyrics_override?: string;
    seed?: number;
  }
): Promise<JobCreatedResponse> {
  return apiFetch<JobCreatedResponse>(baseUrl, `/api/v1/projects/${encodeURIComponent(projectId)}/repaint-track`, {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export async function triggerAddInstrumentLayer(
  baseUrl: string,
  projectId: string,
  payload: {
    track_name: string;
    instrument_prompt: string;
    start_sample: number;
    end_sample: number;
    seed?: number;
  }
): Promise<JobCreatedResponse> {
  return apiFetch<JobCreatedResponse>(baseUrl, `/api/v1/projects/${encodeURIComponent(projectId)}/add-instrument`, {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export async function undoLastRepaint(baseUrl: string, projectId: string): Promise<ProjectSchema> {
  return apiFetch<ProjectSchema>(baseUrl, `/api/v1/projects/${encodeURIComponent(projectId)}/undo-repaint`, {
    method: "POST",
  });
}

export async function triggerExportProject(
  baseUrl: string,
  projectId: string,
  format: "wav" | "stems_zip" | "single_track_wav",
  options?: {
    trackId?: string;
    startSample?: number;
    endSample?: number;
    normalizeLufs?: boolean;
  }
): Promise<JobCreatedResponse> {
  return apiFetch<JobCreatedResponse>(baseUrl, `/api/v1/projects/${encodeURIComponent(projectId)}/export`, {
    method: "POST",
    body: JSON.stringify({
      format,
      track_id: options?.trackId ?? null,
      start_sample: options?.startSample ?? null,
      end_sample: options?.endSample ?? null,
      include_muted: false,
      normalize_lufs: options?.normalizeLufs ?? true,
    }),
  });
}

export async function pollJobUntilDone(
  baseUrl: string,
  jobId: string,
  onProgress: (job: JobStatusResponse) => void,
  intervalMs = 450
): Promise<JobStatusResponse> {
  while (true) {
    const job = await apiFetch<JobStatusResponse>(baseUrl, `/api/v1/jobs/${encodeURIComponent(jobId)}`);
    onProgress(job);
    if (job.status === "completed") {
      return job;
    }
    if (job.status === "failed") {
      throw new Error(job.error || job.stage || "GPU job failed.");
    }
    await new Promise((resolve) => setTimeout(resolve, intervalMs));
  }
}
