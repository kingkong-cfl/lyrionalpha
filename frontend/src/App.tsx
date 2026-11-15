import { useCallback, useEffect, useRef, useState } from "react";
import { AlertCircle, CheckCircle2, Disc3, Layers, PlusCircle } from "lucide-react";
import {
  buildDownloadAudioUrl,
  createProject,
  enhanceWithProducer,
  fetchHealth,
  getProject,
  getSavedApiToken,
  getSavedServerUrl,
  listProjects,
  pollJobUntilDone,
  saveProject,
  setSavedApiToken,
  setSavedServerUrl,
  triggerAddInstrumentLayer,
  triggerExportProject,
  triggerGenerateSong,
  triggerRepaintRegion,
  triggerSingleTrackRepaint,
  undoLastRepaint,
} from "./api/colabClient";
import { audioEngine } from "./audio/AudioEngine";
import { buildRegionFromSeconds } from "./audio/snapUtils";
import { JobStatusBanner } from "./components/JobStatusBanner";
import { PedalboardDrawer } from "./components/Mixer/PedalboardDrawer";
import { ProducerPanel } from "./components/ProducerSidebar/ProducerPanel";
import { BeatGridRuler } from "./components/Timeline/BeatGridRuler";
import { RepaintActionBar } from "./components/Timeline/RepaintActionBar";
import { TrackLane } from "./components/Timeline/TrackLane";
import { GpuConnectionBar } from "./components/TopBar/GpuConnectionBar";
import { TransportBar } from "./components/TopBar/TransportBar";
import type {
  HealthResponse,
  JobStatusResponse,
  ProjectSchema,
  RegionSelection,
  SnapMode,
  TrackSchema,
} from "./types/project";

function normalizeReferenceMixMute(
  project: ProjectSchema,
  prevTracks?: TrackSchema[]
): ProjectSchema {
  const hasStems = project.tracks.some((t) => t.stem_role !== "full_mix");
  if (!hasStems && !prevTracks) return project;

  const tracks = project.tracks.map((t) => {
    const prev = prevTracks?.find((p) => p.track_id === t.track_id);
    if (hasStems && t.stem_role === "full_mix") {
      return {
        ...t,
        muted: true,
        soloed: false,
        volume_db: prev ? prev.volume_db : t.volume_db,
        pan: prev ? prev.pan : t.pan,
        dsp_chain: prev ? prev.dsp_chain : t.dsp_chain,
      };
    }
    if (prev) {
      return {
        ...t,
        muted: prev.muted,
        soloed: prev.soloed,
        volume_db: prev.volume_db,
        pan: prev.pan,
        dsp_chain: prev.dsp_chain,
      };
    }
    return t;
  });

  return { ...project, tracks };
}

export function App() {
  const [serverUrl, setServerUrl] = useState<string>(() => getSavedServerUrl());
  const [apiToken, setApiToken] = useState<string>(() => getSavedApiToken());
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [latencyMs, setLatencyMs] = useState<number | null>(null);
  const [connectionError, setConnectionError] = useState<string | null>(null);
  const [isCheckingHealth, setIsCheckingHealth] = useState(false);

  const [projects, setProjects] = useState<ProjectSchema[]>([]);
  const [activeProject, setActiveProject] = useState<ProjectSchema | null>(null);
  const [activeJob, setActiveJob] = useState<JobStatusResponse | null>(null);
  const [statusToast, setStatusToast] = useState<{ kind: "info" | "error"; text: string } | null>(null);

  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTimeSec, setCurrentTimeSec] = useState(0);
  const [snapMode, setSnapMode] = useState<SnapMode>("bar");
  const [loopEnabled, setLoopEnabled] = useState(false);
  const [pixelsPerSec, setPixelsPerSec] = useState(85);
  const [timelineViewportWidth, setTimelineViewportWidth] = useState(720);
  const [selection, setSelection] = useState<RegionSelection | null>(null);
  const [exportScope, setExportScope] = useState<"full" | "region">("full");
  const [dspTrackId, setDspTrackId] = useState<string | null>(null);
  const [savingMix, setSavingMix] = useState(false);
  const [loadingAudio, setLoadingAudio] = useState(false);

  const rafRef = useRef<number | null>(null);
  const timelineScrollRef = useRef<HTMLDivElement | null>(null);
  const playheadRef = useRef<HTMLDivElement | null>(null);
  const rulerPlayheadRef = useRef<HTMLDivElement | null>(null);
  const currentTimeRef = useRef<number>(0);
  const effectivePxPerSecRef = useRef<number>(85);
  const lastClockCommitMsRef = useRef<number>(0);
  const latestTracksRef = useRef<TrackSchema[]>([]);
  const selectionRef = useRef<RegionSelection | null>(null);

  latestTracksRef.current = activeProject?.tracks ?? [];
  selectionRef.current = selection;

  const updatePlayheadPosition = useCallback((sec: number) => {
    currentTimeRef.current = sec;
    const x = Math.max(0, sec * effectivePxPerSecRef.current);
    const transform = `translate3d(${x}px, 0, 0)`;
    if (playheadRef.current) {
      playheadRef.current.style.transform = transform;
    }
    if (rulerPlayheadRef.current) {
      rulerPlayheadRef.current.style.transform = transform;
    }
  }, []);

  const showToast = useCallback((kind: "info" | "error", text: string) => {
    setStatusToast({ kind, text });
  }, []);

  const handleSelectRegion = useCallback((nextRegion: RegionSelection | null) => {
    const hadSelection = Boolean(selectionRef.current);
    if (!hadSelection && nextRegion) {
      setExportScope("region");
    } else if (!nextRegion) {
      setExportScope("full");
    }
    setSelection(nextRegion);
  }, []);

  const refreshHealthAndProjects = useCallback(
    async (urlToCheck: string) => {
      setIsCheckingHealth(true);
      try {
        const res = await fetchHealth(urlToCheck);
        setHealth(res.health);
        setLatencyMs(res.latencyMs);
        setConnectionError(null);

        const rawList = await listProjects(urlToCheck);
        const list = rawList.map((p) => normalizeReferenceMixMute(p));
        setProjects(list);
        setActiveProject((prev) => {
          if (prev) {
            const match = list.find((p) => p.project_id === prev.project_id);
            if (match) return normalizeReferenceMixMute(match, prev.tracks);
          }
          return list[0] ?? null;
        });
      } catch (err) {
        setHealth(null);
        setLatencyMs(null);
        setConnectionError(err instanceof Error ? err.message : "Backend unreachable");
      } finally {
        setIsCheckingHealth(false);
      }
    },
    []
  );

  // Initial connection & periodic 8s heartbeat
  useEffect(() => {
    refreshHealthAndProjects(serverUrl);
    const id = window.setInterval(async () => {
      try {
        const res = await fetchHealth(serverUrl);
        setHealth(res.health);
        setLatencyMs(res.latencyMs);
        setConnectionError(null);
      } catch (err) {
        setHealth(null);
        setConnectionError(err instanceof Error ? err.message : "Disconnected");
      }
    }, 8000);
    return () => window.clearInterval(id);
  }, [serverUrl, apiToken, refreshHealthAndProjects]);

  // Preload Tone.js multi-track buffers whenever project or track revisions change
  const trackSignature = activeProject
    ? `${activeProject.project_id}:${activeProject.tracks
        .map((t) => `${t.track_id}@${t.revision}:${t.file_rel_path || ""}`)
        .join("|")}`
    : "";

  useEffect(() => {
    if (!activeProject || activeProject.tracks.length === 0) {
      audioEngine.disposeAll();
      setIsPlaying(false);
      updatePlayheadPosition(0);
      setCurrentTimeSec(0);
      return;
    }

    let cancelled = false;
    setLoadingAudio(true);
    audioEngine
      .loadProjectTracks(serverUrl, activeProject)
      .then(() => {
        if (!cancelled) {
          audioEngine.syncMixerParams(latestTracksRef.current);
          setLoadingAudio(false);
        }
      })
      .catch((err) => {
        if (!cancelled) {
          setLoadingAudio(false);
          showToast("error", err instanceof Error ? err.message : "Audio buffer preload failed.");
        }
      });

    return () => {
      cancelled = true;
    };
  }, [trackSignature, serverUrl, apiToken, updatePlayheadPosition]);

  // High-precision 60fps DOM playhead (zero React re-renders) + 10Hz throttled LCD clock
  useEffect(() => {
    const tick = () => {
      const t = audioEngine.getCurrentTimeSec();
      const duration = activeProject?.grid.duration_sec || 0;
      if (duration > 0 && t >= duration && audioEngine.getTransportState() === "started") {
        audioEngine.stop();
        setIsPlaying(false);
        updatePlayheadPosition(0);
        lastClockCommitMsRef.current = performance.now();
        setCurrentTimeSec(0);
      } else {
        updatePlayheadPosition(t);
        const now = performance.now();
        if (now - lastClockCommitMsRef.current >= 100) {
          lastClockCommitMsRef.current = now;
          setCurrentTimeSec(t);
        }
      }
      rafRef.current = window.requestAnimationFrame(tick);
    };

    if (isPlaying) {
      lastClockCommitMsRef.current = performance.now();
      rafRef.current = window.requestAnimationFrame(tick);
    }
    return () => {
      if (rafRef.current !== null) {
        window.cancelAnimationFrame(rafRef.current);
      }
    };
  }, [isPlaying, activeProject?.grid.duration_sec, updatePlayheadPosition]);

  // Update Tone.Transport loop when selection or loopEnabled toggles
  useEffect(() => {
    if (selection && loopEnabled) {
      audioEngine.setLoop(true, selection.startSec, selection.endSec);
    } else {
      audioEngine.setLoop(false);
    }
    if (!selection) {
      setExportScope("full");
    }
  }, [loopEnabled, selection]);

  // Measure timeline viewport width so zoom-out clamps cleanly at 100% width
  useEffect(() => {
    const el = timelineScrollRef.current;
    if (!el) return;
    const updateWidth = () => {
      // Subtract w-64 (256px) sticky left track header column
      const laneArea = Math.max(320, el.clientWidth - 256);
      setTimelineViewportWidth(laneArea);
    };
    updateWidth();
    const observer = new ResizeObserver(updateWidth);
    observer.observe(el);
    return () => observer.disconnect();
  }, [activeProject?.project_id, activeProject?.tracks.length, dspTrackId]);

  const handleUpdateServerUrl = (nextUrl: string) => {
    const cleaned = setSavedServerUrl(nextUrl);
    setServerUrl(cleaned);
  };

  const handleUpdateApiToken = (nextToken: string) => {
    const cleaned = setSavedApiToken(nextToken);
    setApiToken(cleaned);
  };

  const handleSelectProject = async (projectId: string) => {
    if (!projectId) return;
    audioEngine.stop();
    setIsPlaying(false);
    updatePlayheadPosition(0);
    setCurrentTimeSec(0);
    handleSelectRegion(null);
    setDspTrackId(null);
    try {
      const loaded = await getProject(serverUrl, projectId);
      setActiveProject(normalizeReferenceMixMute(loaded));
    } catch (err) {
      showToast("error", err instanceof Error ? err.message : "Failed to load project");
    }
  };

  const handleCreateProject = async (name: string) => {
    try {
      const created = await createProject(serverUrl, name);
      setProjects((prev) => [created, ...prev]);
      setActiveProject(created);
      handleSelectRegion(null);
      setDspTrackId(null);
      showToast("info", `Created Drive project "${created.name}".`);
    } catch (err) {
      showToast("error", err instanceof Error ? err.message : "Failed creating project");
    }
  };

  const ensureActiveProject = async (): Promise<ProjectSchema> => {
    if (activeProject) return activeProject;
    const created = await createProject(serverUrl, "Untitled Studio Session");
    setProjects((prev) => [created, ...prev]);
    setActiveProject(created);
    return created;
  };

  const syncProjectIntoState = (updated: ProjectSchema, preserveMixer = true) => {
    const normalized = normalizeReferenceMixMute(
      updated,
      preserveMixer ? activeProject?.tracks : undefined
    );
    setActiveProject(normalized);
    setProjects((prev) => {
      const exists = prev.some((p) => p.project_id === normalized.project_id);
      return exists
        ? prev.map((p) => (p.project_id === normalized.project_id ? normalized : p))
        : [normalized, ...prev];
    });
  };

  const handleGenerateSong = async (params: {
    raw_prompt: string;
    style_tags: string[];
    structured_lyrics: string;
    bpm: number;
    musical_key: string;
    duration_sec: number;
    seed: number;
    separate_stems: boolean;
  }) => {
    try {
      audioEngine.stop();
      setIsPlaying(false);
      const target = await ensureActiveProject();
      const jobCreated = await triggerGenerateSong(serverUrl, target.project_id, params);
      const done = await pollJobUntilDone(serverUrl, jobCreated.job_id, setActiveJob);
      const updated = done.result?.project as ProjectSchema | undefined;
      if (updated) {
        syncProjectIntoState(updated, false);
        handleSelectRegion(null);
        showToast("info", `Generated ${updated.tracks.length} tracks at ${updated.grid.bpm} BPM.`);
      }
    } catch (err) {
      showToast("error", err instanceof Error ? err.message : "Song generation failed");
    } finally {
      setActiveJob(null);
    }
  };

  const handleRepaint = async (params: {
    start_sample: number;
    end_sample: number;
    prompt_override?: string;
    lyrics_override?: string;
    restem_window: boolean;
  }) => {
    if (!activeProject) return;
    try {
      audioEngine.stop();
      setIsPlaying(false);
      const jobCreated = await triggerRepaintRegion(serverUrl, activeProject.project_id, params);
      const done = await pollJobUntilDone(serverUrl, jobCreated.job_id, setActiveJob);
      const updated = done.result?.project as ProjectSchema | undefined;
      if (updated) {
        syncProjectIntoState(updated);
        showToast(
          "info",
          `Repainted samples [${params.start_sample.toLocaleString()}..${params.end_sample.toLocaleString()}] & re-synced stems.`
        );
      }
    } catch (err) {
      showToast("error", err instanceof Error ? err.message : "Repaint failed");
    } finally {
      setActiveJob(null);
    }
  };

  const handleSingleTrackRepaint = async (params: {
    target_track_id: string;
    start_sample: number;
    end_sample: number;
    prompt_override?: string;
    lyrics_override?: string;
  }) => {
    if (!activeProject) return;
    try {
      audioEngine.stop();
      setIsPlaying(false);
      const jobCreated = await triggerSingleTrackRepaint(serverUrl, activeProject.project_id, params);
      const done = await pollJobUntilDone(serverUrl, jobCreated.job_id, setActiveJob);
      const updated = done.result?.project as ProjectSchema | undefined;
      if (updated) {
        syncProjectIntoState(updated);
        showToast("info", "Repainted selected track region; all other stems stayed bit-identical.");
      }
    } catch (err) {
      showToast("error", err instanceof Error ? err.message : "Single-track repaint failed");
    } finally {
      setActiveJob(null);
    }
  };

  const handleAddInstrumentLayer = async (params: {
    track_name: string;
    instrument_prompt: string;
    start_sample: number;
    end_sample: number;
  }) => {
    if (!activeProject) return;
    try {
      audioEngine.stop();
      setIsPlaying(false);
      const jobCreated = await triggerAddInstrumentLayer(serverUrl, activeProject.project_id, params);
      const done = await pollJobUntilDone(serverUrl, jobCreated.job_id, setActiveJob);
      const updated = done.result?.project as ProjectSchema | undefined;
      if (updated) {
        syncProjectIntoState(updated);
        showToast("info", `Added new instrument track "${params.track_name}" across selected region.`);
      }
    } catch (err) {
      showToast("error", err instanceof Error ? err.message : "Adding instrument track failed");
    } finally {
      setActiveJob(null);
    }
  };

  const handleUndoRepaint = async () => {
    if (!activeProject) return;
    try {
      audioEngine.stop();
      setIsPlaying(false);
      const restored = await undoLastRepaint(serverUrl, activeProject.project_id);
      syncProjectIntoState(restored);
      showToast("info", "Restored previous audio revision.");
    } catch (err) {
      showToast("error", err instanceof Error ? err.message : "Undo failed");
    }
  };

  const handleExport = async (
    format: "wav" | "stems_zip" | "single_track_wav",
    trackId?: string
  ) => {
    if (!activeProject) return;
    try {
      // Persist current mixer settings first so Pedalboard export reflects exact faders/EQ
      await saveProject(serverUrl, activeProject);
      const useRegion = Boolean(selection && exportScope === "region");
      const jobCreated = await triggerExportProject(serverUrl, activeProject.project_id, format, {
        trackId,
        startSample: useRegion ? selection?.startSample : undefined,
        endSample: useRegion ? selection?.endSample : undefined,
        normalizeLufs: true,
      });
      const done = await pollJobUntilDone(serverUrl, jobCreated.job_id, setActiveJob);
      const rel = done.result?.download_rel_path as string | undefined;
      if (rel) {
        const downloadUrl = buildDownloadAudioUrl(serverUrl, activeProject.project_id, rel);
        const link = document.createElement("a");
        link.href = downloadUrl;
        link.download = rel.split("/").pop() || "export.wav";
        link.style.display = "none";
        document.body.appendChild(link);
        link.click();
        link.remove();
        const scopeLabel =
          useRegion && selection
            ? `Region ${selection.startSec.toFixed(2)}s–${selection.endSec.toFixed(2)}s`
            : "Full Song";
        showToast("info", `Exported ${rel} (${scopeLabel}) via Pedalboard mastering.`);
      }
    } catch (err) {
      showToast("error", err instanceof Error ? err.message : "Export failed");
    } finally {
      setActiveJob(null);
    }
  };

  const handleExportRef = useRef(handleExport);
  handleExportRef.current = handleExport;

  const handleUpdateTrack = useCallback((trackId: string, patch: Partial<TrackSchema>) => {
    setActiveProject((prev) => {
      if (!prev) return prev;
      const nextTracks = prev.tracks.map((t) =>
        t.track_id === trackId ? { ...t, ...patch } : t
      );
      audioEngine.syncMixerParams(nextTracks);
      return { ...prev, tracks: nextTracks };
    });
  }, []);

  const handleToggleDspTrack = useCallback((id: string) => {
    setDspTrackId((prev) => (prev === id ? null : id));
  }, []);

  const handleExportTrack = useCallback((id: string) => {
    handleExportRef.current("single_track_wav", id);
  }, []);

  const handleSaveMixToDrive = async () => {
    if (!activeProject) return;
    setSavingMix(true);
    try {
      const saved = await saveProject(serverUrl, activeProject);
      syncProjectIntoState(saved);
      showToast("info", "Saved mixer & Pedalboard settings to project.json on Drive.");
    } catch (err) {
      showToast("error", err instanceof Error ? err.message : "Failed saving project.json");
    } finally {
      setSavingMix(false);
    }
  };

  const handlePlayPause = async () => {
    if (isPlaying) {
      audioEngine.pause();
      setIsPlaying(false);
    } else {
      await audioEngine.play(loopEnabled && selection ? selection : null);
      setIsPlaying(true);
    }
  };

  const handlePlayPauseRef = useRef(handlePlayPause);
  handlePlayPauseRef.current = handlePlayPause;

  const handleStop = () => {
    audioEngine.stop();
    setIsPlaying(false);
    updatePlayheadPosition(0);
    setCurrentTimeSec(0);
  };

  const handleSeek = useCallback(
    (sec: number) => {
      audioEngine.seekToSec(sec);
      updatePlayheadPosition(sec);
      setCurrentTimeSec(sec);
    },
    [updatePlayheadPosition]
  );

  // Spacebar shortcut for Play/Pause (when not typing in inputs)
  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement | null)?.tagName;
      if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;
      if (e.code === "Space") {
        e.preventDefault();
        handlePlayPauseRef.current();
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, []);

  const busy = Boolean(activeJob && (activeJob.status === "queued" || activeJob.status === "running"));
  const dspTrack = activeProject?.tracks.find((t) => t.track_id === dspTrackId) ?? null;
  const grid = activeProject?.grid ?? {
    sample_rate: 44100,
    total_samples: 0,
    duration_sec: 16,
    bpm: 120,
    time_signature: "4/4",
    musical_key: "A minor",
    beat_timestamps: [],
    beat_samples: [],
    downbeat_beat_indices: [],
  };

  const durationSec = Math.max(grid.duration_sec || 16, 1);
  const minPxPerSec = Math.max(10, timelineViewportWidth / durationSec);
  const effectivePxPerSec = Math.max(minPxPerSec, pixelsPerSec);
  const timelineWidthPx = Math.max(
    Math.round(timelineViewportWidth),
    Math.round(durationSec * effectivePxPerSec)
  );

  useEffect(() => {
    effectivePxPerSecRef.current = effectivePxPerSec;
    updatePlayheadPosition(currentTimeRef.current);
  }, [effectivePxPerSec, activeProject?.project_id, updatePlayheadPosition]);

  const handleOpenDefaultRegion = () => {
    const start = Math.min(4, Math.max(0, durationSec * 0.25));
    const end = Math.min(durationSec, start + Math.min(4, durationSec * 0.35));
    const region = buildRegionFromSeconds(start, end, grid);
    if (region) handleSelectRegion(region);
  };

  return (
    <div className="flex h-screen w-screen flex-col overflow-hidden bg-studio-bg text-slate-100">
      <GpuConnectionBar
        serverUrl={serverUrl}
        onUpdateServerUrl={handleUpdateServerUrl}
        apiToken={apiToken}
        onUpdateApiToken={handleUpdateApiToken}
        health={health}
        latencyMs={latencyMs}
        connectionError={connectionError}
        isChecking={isCheckingHealth}
        onPingNow={() => refreshHealthAndProjects(serverUrl)}
      />

      <TransportBar
        isPlaying={isPlaying}
        currentTimeSec={currentTimeSec}
        grid={grid}
        snapMode={snapMode}
        onChangeSnapMode={setSnapMode}
        loopEnabled={loopEnabled}
        onToggleLoop={() => setLoopEnabled((v) => !v)}
        pixelsPerSec={effectivePxPerSec}
        minPxPerSec={minPxPerSec}
        onChangePixelsPerSec={setPixelsPerSec}
        onPlayPause={handlePlayPause}
        onStop={handleStop}
        canUndoRepaint={Boolean(activeProject && activeProject.repaint_history.length > 0)}
        onUndoRepaint={handleUndoRepaint}
        selection={selection}
        exportScope={exportScope}
        onChangeExportScope={setExportScope}
        onExportWav={() => handleExport("wav")}
        onExportStemsZip={() => handleExport("stems_zip")}
        hasAudioTracks={Boolean(activeProject && activeProject.tracks.length > 0)}
        busy={busy}
      />

      <JobStatusBanner job={activeJob} />

      {statusToast && (
        <div
          className={`flex items-center justify-between border-b px-4 py-1.5 text-xs ${
            statusToast.kind === "error"
              ? "border-rose-500/40 bg-rose-500/10 text-rose-200"
              : "border-emerald-500/30 bg-emerald-500/10 text-emerald-200"
          }`}
        >
          <div className="flex items-center gap-2">
            {statusToast.kind === "error" ? (
              <AlertCircle className="h-3.5 w-3.5 text-rose-400" />
            ) : (
              <CheckCircle2 className="h-3.5 w-3.5 text-emerald-400" />
            )}
            <span>{statusToast.text}</span>
          </div>
          <button
            type="button"
            onClick={() => setStatusToast(null)}
            className="text-[11px] opacity-75 hover:opacity-100"
          >
            Dismiss
          </button>
        </div>
      )}

      {/* Main Split Workspace */}
      <div className="flex flex-1 min-h-0 flex-col lg:flex-row">
        <ProducerPanel
          projects={projects}
          activeProject={activeProject}
          onSelectProject={handleSelectProject}
          onCreateProject={handleCreateProject}
          onEnhancePrompt={(prompt, dur) => enhanceWithProducer(serverUrl, prompt, dur)}
          onGenerateSong={handleGenerateSong}
          busy={busy}
          connected={Boolean(health)}
        />

        {/* Multi-Track Timeline */}
        <main className="flex flex-1 min-w-0 flex-col bg-studio-bg">
          {activeProject && activeProject.tracks.length > 0 ? (
            <>
              <div className="flex flex-wrap items-center justify-between gap-2 border-b border-studio-border bg-studio-panel/60 px-4 py-1.5 text-[11px] text-slate-400">
                <div className="flex items-center gap-2">
                  <Layers className="h-3.5 w-3.5 text-indigo-400" />
                  <span>
                    <strong className="text-slate-200">{activeProject.name}</strong> •{" "}
                    {activeProject.tracks.length} tracks • {grid.sample_rate} Hz •{" "}
                    {grid.total_samples.toLocaleString()} samples
                  </span>
                </div>
                <div className="flex items-center gap-3">
                  <span className="font-mono text-[10px] text-cyan-300/90">
                    {loadingAudio
                      ? "Preloading Web Audio buffers (Promise.all)..."
                      : "Drag across any waveform or ruler to select a region"}
                  </span>
                  {!selection && (
                    <button
                      type="button"
                      onClick={handleOpenDefaultRegion}
                      className="flex items-center gap-1 rounded border border-emerald-500/40 bg-emerald-500/10 px-2 py-0.5 text-[11px] font-medium text-emerald-300 hover:bg-emerald-500/20 transition"
                    >
                      <PlusCircle className="h-3 w-3" />
                      <span>Select Region / + Add Instrument</span>
                    </button>
                  )}
                </div>
              </div>

              <div ref={timelineScrollRef} className="flex-1 overflow-auto">
                <div className="relative min-w-max">
                  {/* Unified 60fps GPU-Composited Playhead across tracks (clipped behind sticky w-64 controls) */}
                  <div
                    ref={playheadRef}
                    style={{ transform: `translate3d(${currentTimeRef.current * effectivePxPerSec}px, 0, 0)` }}
                    className="pointer-events-none absolute top-0 bottom-0 left-64 z-[25] w-[1.5px] bg-rose-500 will-change-transform"
                  />

                  {/* Beat Grid Header */}
                  <div className="sticky top-0 z-30 flex border-b border-studio-border bg-studio-panel">
                    <div className="sticky left-0 z-30 flex w-64 shrink-0 items-center justify-between border-r border-studio-border bg-studio-panel px-3 text-[11px] font-semibold uppercase tracking-wider text-slate-400">
                      <span>Track / Stem</span>
                      <span className="font-mono text-[10px] text-slate-500">M / S / Vol / Pan</span>
                    </div>
                    <BeatGridRuler
                      grid={grid}
                      pixelsPerSec={effectivePxPerSec}
                      timelineWidthPx={timelineWidthPx}
                      snapMode={snapMode}
                      selection={selection}
                      playheadRef={rulerPlayheadRef}
                      onSeek={handleSeek}
                      onSelectRegion={handleSelectRegion}
                    />
                  </div>

                  {/* Multi-Track Lanes */}
                  {activeProject.tracks.map((track) => (
                    <TrackLane
                      key={track.track_id}
                      track={track}
                      waveform={audioEngine.getTrackWaveform(track.track_id)}
                      grid={grid}
                      pixelsPerSec={effectivePxPerSec}
                      timelineWidthPx={timelineWidthPx}
                      snapMode={snapMode}
                      selection={selection}
                      exportScope={exportScope}
                      isDspSelected={dspTrackId === track.track_id}
                      onUpdateTrack={handleUpdateTrack}
                      onOpenDsp={handleToggleDspTrack}
                      onExportTrack={handleExportTrack}
                      onSeek={handleSeek}
                      onSelectRegion={handleSelectRegion}
                      busy={busy}
                    />
                  ))}
                </div>
              </div>

              {selection && (
                <RepaintActionBar
                  selection={selection}
                  grid={grid}
                  snapMode={snapMode}
                  tracks={activeProject.tracks}
                  onUpdateSelection={handleSelectRegion}
                  onClear={() => handleSelectRegion(null)}
                  onRepaint={handleRepaint}
                  onRepaintSingleTrack={handleSingleTrackRepaint}
                  onAddInstrumentLayer={handleAddInstrumentLayer}
                  busy={busy}
                />
              )}
            </>
          ) : (
            <div className="flex flex-1 flex-col items-center justify-center p-8 text-center">
              <div className="mb-4 flex h-16 w-16 items-center justify-center rounded-2xl border border-indigo-500/30 bg-indigo-500/10 text-indigo-400">
                <Disc3 className="h-8 w-8" />
              </div>
              <h2 className="text-lg font-bold text-slate-100">Ready to Produce in Lyrion Studio</h2>
              <p className="mt-2 max-w-md text-xs leading-relaxed text-slate-400">
                Enter a vibe prompt in the AI Producer sidebar, structure your lyrics &amp; BPM with zero
                VRAM, and click <strong className="text-indigo-300">Generate Song &amp; Stems</strong>. Your
                4-stem multi-track timeline and Beat Grid will load directly from Google Drive.
              </p>
            </div>
          )}
        </main>

        {dspTrack && (
          <PedalboardDrawer
            track={dspTrack}
            onUpdateTrack={handleUpdateTrack}
            onSaveToDrive={handleSaveMixToDrive}
            onClose={() => setDspTrackId(null)}
            saving={savingMix}
            busy={busy}
          />
        )}
      </div>
    </div>
  );
}

export default App;
