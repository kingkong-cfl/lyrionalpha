import React, { useEffect, useMemo, useRef } from "react";
import WaveSurfer from "wavesurfer.js";
import { Download, Sliders, Volume2, VolumeX } from "lucide-react";
import type { TrackWaveformData } from "../../audio/AudioEngine";
import { buildSnappedRegion } from "../../audio/snapUtils";
import type { BeatGridSchema, RegionSelection, SnapMode, TrackSchema } from "../../types/project";

interface TrackLaneProps {
  track: TrackSchema;
  waveform: TrackWaveformData | null;
  grid: BeatGridSchema;
  pixelsPerSec: number;
  timelineWidthPx: number;
  snapMode: SnapMode;
  selection: RegionSelection | null;
  exportScope: "full" | "region";
  isDspSelected: boolean;
  onUpdateTrack: (trackId: string, patch: Partial<TrackSchema>) => void;
  onOpenDsp: (trackId: string) => void;
  onExportTrack: (trackId: string) => void;
  onSeek: (sec: number) => void;
  onSelectRegion: (region: RegionSelection | null) => void;
  busy: boolean;
}

const ROLE_COLORS: Record<string, { wave: string; progress: string; badge: string }> = {
  full_mix: { wave: "#475569", progress: "#94a3b8", badge: "bg-slate-700/70 text-slate-200" },
  vocals: { wave: "#c084fc", progress: "#e879f9", badge: "bg-purple-500/20 text-purple-300" },
  drums: { wave: "#fb923c", progress: "#f97316", badge: "bg-orange-500/20 text-orange-300" },
  bass: { wave: "#38bdf8", progress: "#22d3ee", badge: "bg-cyan-500/20 text-cyan-300" },
  other: { wave: "#34d399", progress: "#10b981", badge: "bg-emerald-500/20 text-emerald-300" },
  custom: { wave: "#818cf8", progress: "#6366f1", badge: "bg-indigo-500/20 text-indigo-300" },
};

export const TrackLane: React.FC<TrackLaneProps> = React.memo(({
  track,
  waveform,
  grid,
  pixelsPerSec,
  timelineWidthPx,
  snapMode,
  selection,
  exportScope,
  isDspSelected,
  onUpdateTrack,
  onOpenDsp,
  onExportTrack,
  onSeek,
  onSelectRegion,
  busy,
}) => {
  const waveContainerRef = useRef<HTMLDivElement | null>(null);
  const wsRef = useRef<WaveSurfer | null>(null);
  const wsReadyRef = useRef<boolean>(false);
  const latestPxPerSecRef = useRef<number>(pixelsPerSec);
  const dragStartSecRef = useRef<number | null>(null);

  latestPxPerSecRef.current = pixelsPerSec;

  const palette = ROLE_COLORS[track.stem_role] || ROLE_COLORS.custom;
  const durationSec = Math.max(grid.duration_sec || 16, 1);
  const timelineWidth = Math.max(200, Math.round(timelineWidthPx));

  const barGridTimes = useMemo(
    () =>
      grid.downbeat_beat_indices
        .map((beatIdx) => ({ beatIdx, t: grid.beat_timestamps[beatIdx] }))
        .filter((item): item is { beatIdx: number; t: number } => item.t !== undefined),
    [grid.downbeat_beat_indices, grid.beat_timestamps]
  );

  useEffect(() => {
    if (
      !waveContainerRef.current ||
      !track.file_rel_path ||
      !waveform ||
      waveform.peaks.length === 0 ||
      waveform.duration <= 0
    ) {
      return;
    }
    wsReadyRef.current = false;

    const ws = WaveSurfer.create({
      container: waveContainerRef.current,
      backend: "WebAudio",
      waveColor: track.muted ? "#334155" : palette.wave,
      progressColor: track.muted ? "#475569" : palette.progress,
      cursorWidth: 0,
      height: 82,
      barWidth: 2,
      barGap: 1,
      barRadius: 2,
      normalize: false,
      interact: false,
      minPxPerSec: latestPxPerSecRef.current,
      fillParent: true,
    });

    ws.on("ready", () => {
      wsReadyRef.current = true;
      try {
        ws.zoom(latestPxPerSecRef.current);
      } catch {
        // Ignore zoom race on fast unmount
      }
    });

    ws.load("", waveform.peaks, waveform.duration).catch(() => {
      // Handled gracefully when disconnected or aborted
    });

    wsRef.current = ws;
    return () => {
      wsReadyRef.current = false;
      ws.destroy();
      wsRef.current = null;
    };
  }, [track.file_rel_path, waveform?.url, waveform?.peaks, waveform?.duration]);

  // Synchronize WaveSurfer horizontal scale whenever zoom or viewport width changes
  useEffect(() => {
    if (!wsRef.current || !wsReadyRef.current) return;
    try {
      wsRef.current.zoom(pixelsPerSec);
    } catch {
      // Ignore if audio not yet decoded
    }
  }, [pixelsPerSec, timelineWidth]);

  // Update waveform colors live when Mute state changes without re-fetching WAV
  useEffect(() => {
    if (!wsRef.current) return;
    wsRef.current.setOptions({
      waveColor: track.muted ? "#334155" : palette.wave,
      progressColor: track.muted ? "#475569" : palette.progress,
    });
  }, [track.muted, palette.wave, palette.progress]);

  const timeFromMouseEvent = (e: React.MouseEvent<HTMLDivElement>): number => {
    const rect = e.currentTarget.getBoundingClientRect();
    const x = Math.max(0, e.clientX - rect.left);
    return Math.min(durationSec, x / pixelsPerSec);
  };

  const handleLaneMouseDown = (e: React.MouseEvent<HTMLDivElement>) => {
    const t = timeFromMouseEvent(e);
    dragStartSecRef.current = t;
  };

  const handleLaneMouseMove = (e: React.MouseEvent<HTMLDivElement>) => {
    if (dragStartSecRef.current === null || (e.buttons & 1) === 0) return;
    const t = timeFromMouseEvent(e);
    if (Math.abs(t - dragStartSecRef.current) > 0.08) {
      const region = buildSnappedRegion(dragStartSecRef.current, t, grid, snapMode);
      onSelectRegion(region);
    }
  };

  const handleLaneMouseUp = (e: React.MouseEvent<HTMLDivElement>) => {
    if (dragStartSecRef.current === null) return;
    const t = timeFromMouseEvent(e);
    if (Math.abs(t - dragStartSecRef.current) <= 0.08) {
      onSeek(t);
    }
    dragStartSecRef.current = null;
  };

  return (
    <div className="flex border-b border-studio-border bg-studio-bg/80 hover:bg-studio-panel/40 transition-colors">
      {/* Sticky Left Track Header (Mixer Strip) */}
      <div className="sticky left-0 z-30 flex w-64 shrink-0 flex-col justify-between border-r border-studio-border bg-studio-panel p-2.5">
        <div className="flex items-center justify-between gap-2">
          <div className="min-w-0">
            <div className="flex items-center gap-1.5">
              <span className="truncate text-xs font-semibold text-slate-100">{track.name}</span>
              <span className={`rounded px-1.5 py-0.5 font-mono text-[9px] uppercase ${palette.badge}`}>
                {track.track_type}
              </span>
            </div>
            <div className="mt-0.5 font-mono text-[10px] text-slate-500">
              rev {track.revision} • {track.volume_db >= 0 ? `+${track.volume_db.toFixed(1)}` : track.volume_db.toFixed(1)} dB
            </div>
          </div>

          <div className="flex items-center gap-1">
            <button
              type="button"
              disabled={busy || !track.file_rel_path}
              onClick={() => onExportTrack(track.track_id)}
              title={
                selection && exportScope === "region"
                  ? `Export "${track.name}" Region (${selection.startSec.toFixed(2)}s - ${selection.endSec.toFixed(2)}s) WAV with Pedalboard DSP`
                  : `Export "${track.name}" Full Track WAV with Pedalboard DSP`
              }
              className={`flex items-center gap-1 rounded border p-1.5 disabled:opacity-35 transition ${
                selection && exportScope === "region"
                  ? "border-cyan-500/50 bg-cyan-500/10 text-cyan-300 hover:bg-cyan-500/20"
                  : "border-studio-border bg-studio-bg text-emerald-400 hover:border-emerald-500/60 hover:bg-emerald-500/10"
              }`}
            >
              <Download className="h-3.5 w-3.5" />
              {selection && exportScope === "region" && (
                <span className="font-mono text-[9px] font-bold leading-none">RGN</span>
              )}
            </button>

            <button
              type="button"
              onClick={() => onOpenDsp(track.track_id)}
              title="Pedalboard DSP (EQ / Compressor / Reverb)"
              className={`rounded border p-1.5 transition ${
                isDspSelected
                  ? "border-indigo-500 bg-indigo-500/20 text-indigo-300"
                  : "border-studio-border bg-studio-bg text-slate-400 hover:text-slate-100"
              }`}
            >
              <Sliders className="h-3.5 w-3.5" />
            </button>
          </div>
        </div>

        {/* M / S / Volume / Pan Controls */}
        <div className="mt-2 flex items-center gap-2">
          <button
            type="button"
            onClick={() => onUpdateTrack(track.track_id, { muted: !track.muted })}
            className={`flex h-6 w-7 items-center justify-center rounded border text-[10px] font-bold transition ${
              track.muted
                ? "border-rose-500/60 bg-rose-500/20 text-rose-300"
                : "border-studio-border bg-studio-bg text-slate-400 hover:text-white"
            }`}
            title="Mute"
          >
            {track.muted ? <VolumeX className="h-3 w-3" /> : "M"}
          </button>

          <button
            type="button"
            onClick={() => onUpdateTrack(track.track_id, { soloed: !track.soloed })}
            className={`flex h-6 w-7 items-center justify-center rounded border text-[10px] font-bold transition ${
              track.soloed
                ? "border-amber-400/70 bg-amber-500/20 text-amber-300"
                : "border-studio-border bg-studio-bg text-slate-400 hover:text-white"
            }`}
            title="Solo"
          >
            S
          </button>

          <div className="flex flex-1 items-center gap-1.5">
            <Volume2 className="h-3 w-3 text-slate-500 shrink-0" />
            <input
              type="range"
              min={-36}
              max={6}
              step={0.5}
              value={track.volume_db}
              onChange={(e) => onUpdateTrack(track.track_id, { volume_db: Number(e.target.value) })}
              className="w-full cursor-pointer"
              title={`Volume: ${track.volume_db} dB`}
            />
          </div>

          <input
            type="range"
            min={-1}
            max={1}
            step={0.05}
            value={track.pan}
            onChange={(e) => onUpdateTrack(track.track_id, { pan: Number(e.target.value) })}
            className="w-12 cursor-pointer"
            title={`Pan: ${Math.round(track.pan * 100)}%`}
          />
        </div>
      </div>

      {/* Right Waveform Lane + Grid Lines + Selection Overlay */}
      <div
        onMouseDown={handleLaneMouseDown}
        onMouseMove={handleLaneMouseMove}
        onMouseUp={handleLaneMouseUp}
        style={{ width: `${timelineWidth}px` }}
        className="relative h-[94px] shrink-0 cursor-crosshair select-none overflow-hidden py-1.5"
      >
        {/* Subtle vertical bar gridlines */}
        {barGridTimes.map(({ beatIdx, t }) => (
          <div
            key={beatIdx}
            style={{ left: `${t * pixelsPerSec}px` }}
            className="pointer-events-none absolute top-0 bottom-0 w-px bg-slate-800/80"
          />
        ))}

        {/* Wavesurfer v7 Waveform Container */}
        <div
          ref={waveContainerRef}
          style={{ width: `${Math.round(durationSec * pixelsPerSec)}px` }}
          className="pointer-events-none relative z-10"
        />

        {/* Selected Repaint Region Overlay */}
        {selection && (
          <div
            style={{
              left: `${selection.startSec * pixelsPerSec}px`,
              width: `${Math.max(2, (selection.endSec - selection.startSec) * pixelsPerSec)}px`,
            }}
            className="pointer-events-none absolute top-0 bottom-0 z-10 border-x border-cyan-400/70 bg-cyan-400/12"
          />
        )}

      </div>
    </div>
  );
});
