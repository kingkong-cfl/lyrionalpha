import React from "react";
import {
  Download,
  Magnet,
  Pause,
  Play,
  Repeat,
  Square,
  Undo2,
  ZoomIn,
  ZoomOut,
} from "lucide-react";
import type { BeatGridSchema, RegionSelection, SnapMode } from "../../types/project";

interface TransportBarProps {
  isPlaying: boolean;
  timecodeRef: React.RefObject<HTMLDivElement>;
  barBeatRef: React.RefObject<HTMLDivElement>;
  grid: BeatGridSchema;
  snapMode: SnapMode;
  onChangeSnapMode: (mode: SnapMode) => void;
  loopEnabled: boolean;
  onToggleLoop: () => void;
  pixelsPerSec: number;
  minPxPerSec: number;
  onChangePixelsPerSec: (pps: number) => void;
  onPlayPause: () => void;
  onStop: () => void;
  canUndoRepaint: boolean;
  onUndoRepaint: () => void;
  selection: RegionSelection | null;
  exportScope: "full" | "region";
  onChangeExportScope: (scope: "full" | "region") => void;
  onExportWav: () => void;
  onExportStemsZip: () => void;
  hasAudioTracks: boolean;
  loadingAudio?: boolean;
  busy: boolean;
}

export const TransportBar: React.FC<TransportBarProps> = ({
  isPlaying,
  timecodeRef,
  barBeatRef,
  grid,
  snapMode,
  onChangeSnapMode,
  loopEnabled,
  onToggleLoop,
  pixelsPerSec,
  minPxPerSec,
  onChangePixelsPerSec,
  onPlayPause,
  onStop,
  canUndoRepaint,
  onUndoRepaint,
  selection,
  exportScope,
  onChangeExportScope,
  onExportWav,
  onExportStemsZip,
  hasAudioTracks,
  loadingAudio = false,
  busy,
}) => {
  const sliderMin = Math.max(10, Math.round(minPxPerSec));
  const sliderMax = Math.max(240, Math.round(sliderMin * 4));
  const sliderValue = Math.max(sliderMin, Math.min(sliderMax, Math.round(pixelsPerSec)));
  return (
    <div className="flex flex-wrap items-center justify-between gap-3 border-b border-studio-border bg-studio-surface/70 px-4 py-2">
      {/* Left: Transport Controls + Clock */}
      <div className="flex items-center gap-3">
        <div className="flex items-center gap-1 rounded-lg border border-studio-border bg-studio-bg p-1">
          <button
            type="button"
            disabled={!hasAudioTracks || loadingAudio}
            onClick={onPlayPause}
            className="flex h-8 w-8 items-center justify-center rounded bg-indigo-600 text-white hover:bg-indigo-500 disabled:opacity-40 transition"
            title={loadingAudio ? "Preloading audio buffers..." : isPlaying ? "Pause (Space)" : "Play (Space)"}
          >
            {isPlaying ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4 ml-0.5" />}
          </button>
          <button
            type="button"
            disabled={!hasAudioTracks}
            onClick={onStop}
            className="flex h-8 w-8 items-center justify-center rounded text-slate-300 hover:bg-studio-surface hover:text-white disabled:opacity-40 transition"
            title="Stop & Rewind"
          >
            <Square className="h-3.5 w-3.5" />
          </button>
          <button
            type="button"
            disabled={!hasAudioTracks}
            onClick={onToggleLoop}
            className={`flex h-8 w-8 items-center justify-center rounded transition ${
              loopEnabled
                ? "bg-cyan-500/20 text-cyan-300"
                : "text-slate-400 hover:bg-studio-surface hover:text-slate-200"
            }`}
            title="Loop Selected Region"
          >
            <Repeat className="h-3.5 w-3.5" />
          </button>
        </div>

        {/* LCD Readout (DOM-updated at 20Hz with zero React re-renders) */}
        <div className="flex items-center gap-4 rounded-lg border border-studio-border bg-studio-bg px-3.5 py-1.5 font-mono">
          <div>
            <div className="text-[9px] uppercase tracking-wider text-slate-500">Time</div>
            <div ref={timecodeRef} className="text-sm font-semibold text-cyan-300">00:00.00</div>
          </div>
          <div className="h-6 w-px bg-studio-border" />
          <div>
            <div className="text-[9px] uppercase tracking-wider text-slate-500">Bar : Beat</div>
            <div ref={barBeatRef} className="text-sm font-semibold text-slate-100">1:1</div>
          </div>
          <div className="h-6 w-px bg-studio-border" />
          <div>
            <div className="text-[9px] uppercase tracking-wider text-slate-500">Tempo / Key</div>
            <div className="text-xs font-medium text-indigo-300">
              {Math.round(grid.bpm)} BPM • {grid.musical_key}
            </div>
          </div>
        </div>
      </div>

      {/* Center: Snap Mode & Zoom */}
      <div className="flex items-center gap-3">
        <div className="flex items-center gap-1.5 rounded-lg border border-studio-border bg-studio-bg px-2.5 py-1">
          <Magnet className="h-3.5 w-3.5 text-amber-400" />
          <span className="text-[11px] text-slate-400">Snap:</span>
          {(["bar", "beat", "off"] as SnapMode[]).map((mode) => (
            <button
              key={mode}
              type="button"
              onClick={() => onChangeSnapMode(mode)}
              className={`rounded px-2 py-0.5 text-[11px] font-medium capitalize transition ${
                snapMode === mode
                  ? "bg-amber-500/20 text-amber-300"
                  : "text-slate-400 hover:text-slate-200"
              }`}
            >
              {mode}
            </button>
          ))}
        </div>

        <div className="hidden md:flex items-center gap-2 rounded-lg border border-studio-border bg-studio-bg px-2.5 py-1">
          <ZoomOut className="h-3.5 w-3.5 text-slate-400" />
          <input
            type="range"
            min={sliderMin}
            max={sliderMax}
            value={sliderValue}
            onChange={(e) => onChangePixelsPerSec(Number(e.target.value))}
            className="w-24 cursor-pointer"
          />
          <ZoomIn className="h-3.5 w-3.5 text-slate-400" />
        </div>
      </div>

      {/* Right: Undo & Mastering Exports */}
      <div className="flex flex-wrap items-center gap-2">
        <button
          type="button"
          disabled={!canUndoRepaint || busy}
          onClick={onUndoRepaint}
          className="flex items-center gap-1.5 rounded-md border border-studio-border bg-studio-bg px-2.5 py-1.5 text-xs font-medium text-slate-200 hover:border-amber-500/60 hover:text-amber-300 disabled:opacity-35 transition"
          title="Restore previous audio revision from Drive history/"
        >
          <Undo2 className="h-3.5 w-3.5" />
          <span>Undo Repaint</span>
        </button>

        {selection && (
          <div
            className="flex items-center rounded-md border border-studio-border bg-studio-bg p-0.5 text-[11px]"
            title="Choose whether Master WAV / Stems ZIP / Track WAV exports the Full Song or Selected Region"
          >
            <button
              type="button"
              onClick={() => onChangeExportScope("full")}
              className={`rounded px-2 py-1 font-medium transition ${
                exportScope === "full"
                  ? "bg-indigo-600 text-white"
                  : "text-slate-400 hover:text-slate-200"
              }`}
            >
              Full Song
            </button>
            <button
              type="button"
              onClick={() => onChangeExportScope("region")}
              className={`rounded px-2 py-1 font-medium transition ${
                exportScope === "region"
                  ? "bg-cyan-500/25 text-cyan-300"
                  : "text-slate-400 hover:text-slate-200"
              }`}
            >
              Region Only
            </button>
          </div>
        )}

        <button
          type="button"
          disabled={!hasAudioTracks || busy}
          onClick={onExportWav}
          className="flex items-center gap-1.5 rounded-md border border-studio-border bg-studio-bg px-3 py-1.5 text-xs font-medium text-emerald-300 hover:border-emerald-500/60 hover:bg-emerald-500/10 disabled:opacity-35 transition"
        >
          <Download className="h-3.5 w-3.5" />
          <span>{selection && exportScope === "region" ? "Region WAV" : "Master WAV"}</span>
        </button>

        <button
          type="button"
          disabled={!hasAudioTracks || busy}
          onClick={onExportStemsZip}
          className="flex items-center gap-1.5 rounded-md bg-emerald-600 px-3 py-1.5 text-xs font-semibold text-white hover:bg-emerald-500 disabled:opacity-35 transition"
        >
          <Download className="h-3.5 w-3.5" />
          <span>{selection && exportScope === "region" ? "Region Stems ZIP" : "Export Stems ZIP"}</span>
        </button>
      </div>
    </div>
  );
};
