import React, { useEffect, useRef, useState } from "react";
import { PlusCircle, Sparkles, Target, X } from "lucide-react";
import { buildRegionFromSeconds } from "../../audio/snapUtils";
import type { BeatGridSchema, RegionSelection, SnapMode, TrackSchema } from "../../types/project";

type ActionMode = "full_mix" | "single_track" | "add_instrument";

interface RepaintActionBarProps {
  selection: RegionSelection;
  grid: BeatGridSchema;
  snapMode: SnapMode;
  tracks: TrackSchema[];
  onUpdateSelection: (region: RegionSelection | null) => void;
  onClear: () => void;
  onRepaint: (params: {
    start_sample: number;
    end_sample: number;
    prompt_override?: string;
    lyrics_override?: string;
    restem_window: boolean;
  }) => Promise<void>;
  onRepaintSingleTrack: (params: {
    target_track_id: string;
    start_sample: number;
    end_sample: number;
    prompt_override?: string;
    lyrics_override?: string;
  }) => Promise<void>;
  onAddInstrumentLayer: (params: {
    track_name: string;
    instrument_prompt: string;
    start_sample: number;
    end_sample: number;
  }) => Promise<void>;
  busy: boolean;
}

export const RepaintActionBar: React.FC<RepaintActionBarProps> = ({
  selection,
  grid,
  tracks,
  onUpdateSelection,
  onClear,
  onRepaint,
  onRepaintSingleTrack,
  onAddInstrumentLayer,
  busy,
}) => {
  const [mode, setMode] = useState<ActionMode>("full_mix");
  const [promptOverride, setPromptOverride] = useState("");
  const [lyricsOverride, setLyricsOverride] = useState("");
  const [restemWindow, setRestemWindow] = useState(true);

  const editableTracks = tracks.filter((t) => t.stem_role !== "full_mix" && t.file_rel_path);
  const defaultTargetId = editableTracks[0]?.track_id || tracks[0]?.track_id || "";
  const [targetTrackId, setTargetTrackId] = useState<string>(defaultTargetId);

  const [newTrackName, setNewTrackName] = useState("Electric Guitar");
  const [instrumentPrompt, setInstrumentPrompt] = useState(
    `melodic electric guitar riff, ${Math.round(grid.bpm || 120)} BPM, ${grid.musical_key}`
  );
  const instrumentPromptEditedRef = useRef(false);

  useEffect(() => {
    if (!instrumentPromptEditedRef.current) {
      setInstrumentPrompt(
        `melodic electric guitar riff, ${Math.round(grid.bpm || 120)} BPM, ${grid.musical_key}`
      );
    }
  }, [grid.bpm, grid.musical_key]);

  // Bi-directional numeric Start (s) / End (s) inputs synced with selection
  const [startDraft, setStartDraft] = useState(selection.startSec.toFixed(2));
  const [endDraft, setEndDraft] = useState(selection.endSec.toFixed(2));
  const startInputRef = useRef<HTMLInputElement | null>(null);
  const endInputRef = useRef<HTMLInputElement | null>(null);

  useEffect(() => {
    if (document.activeElement !== startInputRef.current) {
      setStartDraft(selection.startSec.toFixed(2));
    }
    if (document.activeElement !== endInputRef.current) {
      setEndDraft(selection.endSec.toFixed(2));
    }
  }, [selection.startSample, selection.endSample, selection.startSec, selection.endSec]);

  useEffect(() => {
    if (!tracks.some((t) => t.track_id === targetTrackId) && defaultTargetId) {
      setTargetTrackId(defaultTargetId);
    }
  }, [tracks, targetTrackId, defaultTargetId]);

  const maxDur = Math.max(0.1, grid.duration_sec || 30);

  const applyNumericRange = (
    nextStartStr: string,
    nextEndStr: string,
    clamp = false
  ): RegionSelection | null => {
    let s = parseFloat(nextStartStr);
    let e = parseFloat(nextEndStr);
    if (!Number.isFinite(s) || !Number.isFinite(e)) return null;

    if (clamp) {
      s = Math.max(0, Math.min(Math.max(0, maxDur - 0.05), s));
      e = Math.max(Math.min(maxDur, s + 0.05), Math.min(maxDur, e));
    } else if (s < 0 || e <= s || e > maxDur) {
      return null;
    }

    const region = buildRegionFromSeconds(s, e, grid);
    if (region) {
      onUpdateSelection(region);
      return region;
    }
    return null;
  };

  const commitNumericDrafts = () => {
    const region = applyNumericRange(startDraft, endDraft, true);
    if (region) {
      setStartDraft(region.startSec.toFixed(2));
      setEndDraft(region.endSec.toFixed(2));
    } else {
      setStartDraft(selection.startSec.toFixed(2));
      setEndDraft(selection.endSec.toFixed(2));
    }
  };

  const handleNumericKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter") {
      e.preventDefault();
      commitNumericDrafts();
      e.currentTarget.blur();
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (mode === "full_mix") {
      await onRepaint({
        start_sample: selection.startSample,
        end_sample: selection.endSample,
        prompt_override: promptOverride.trim() || undefined,
        lyrics_override: lyricsOverride.trim() || undefined,
        restem_window: restemWindow,
      });
    } else if (mode === "single_track") {
      if (!targetTrackId) return;
      await onRepaintSingleTrack({
        target_track_id: targetTrackId,
        start_sample: selection.startSample,
        end_sample: selection.endSample,
        prompt_override: promptOverride.trim() || undefined,
        lyrics_override: lyricsOverride.trim() || undefined,
      });
    } else {
      if (!newTrackName.trim() || !instrumentPrompt.trim()) return;
      await onAddInstrumentLayer({
        track_name: newTrackName.trim(),
        instrument_prompt: instrumentPrompt.trim(),
        start_sample: selection.startSample,
        end_sample: selection.endSample,
      });
    }
  };

  return (
    <form
      onSubmit={handleSubmit}
      className="border-t border-cyan-500/40 bg-studio-panel/95 px-4 py-3 shadow-2xl"
    >
      <div className="flex flex-wrap items-center justify-between gap-3">
        {/* Sample-Accurate Coordinates + Bi-directional Numeric Inputs */}
        <div className="flex flex-wrap items-center gap-2.5">
          <span className="rounded bg-cyan-500/20 px-2 py-0.5 font-mono text-xs font-semibold text-cyan-300">
            Bars {selection.startLabel} → {selection.endLabel}
          </span>

          <div className="flex items-center gap-1.5 rounded border border-studio-border bg-studio-bg px-2 py-0.5 font-mono text-[11px]">
            <label className="text-slate-400">Start (s)</label>
            <input
              ref={startInputRef}
              type="number"
              min={0}
              max={Math.max(0.1, maxDur - 0.05)}
              step="any"
              value={startDraft}
              onChange={(e) => {
                setStartDraft(e.target.value);
                applyNumericRange(e.target.value, endDraft, false);
              }}
              onBlur={commitNumericDrafts}
              onKeyDown={handleNumericKeyDown}
              className="w-16 rounded border border-studio-border bg-studio-panel px-1.5 py-0.5 text-cyan-300 focus:border-cyan-500 focus:outline-none"
            />
            <label className="text-slate-400 ml-1">End (s)</label>
            <input
              ref={endInputRef}
              type="number"
              min={0}
              max={maxDur}
              step="any"
              value={endDraft}
              onChange={(e) => {
                setEndDraft(e.target.value);
                applyNumericRange(startDraft, e.target.value, false);
              }}
              onBlur={commitNumericDrafts}
              onKeyDown={handleNumericKeyDown}
              className="w-16 rounded border border-studio-border bg-studio-panel px-1.5 py-0.5 text-cyan-300 focus:border-cyan-500 focus:outline-none"
            />
          </div>

          <span className="hidden xl:inline font-mono text-[11px] text-slate-400">
            samples [{selection.startSample.toLocaleString()} .. {selection.endSample.toLocaleString()}]
          </span>
        </div>

        {/* 3-Mode Targeted Switcher */}
        <div className="flex items-center gap-2">
          <div className="flex flex-wrap rounded-md border border-studio-border bg-studio-bg p-0.5 text-xs">
            <button
              type="button"
              onClick={() => setMode("full_mix")}
              className={`flex items-center gap-1 rounded px-2.5 py-1 transition ${
                mode === "full_mix" ? "bg-indigo-600 text-white" : "text-slate-400 hover:text-slate-200"
              }`}
            >
              <Sparkles className="h-3 w-3" />
              <span>Full Mix Repaint</span>
            </button>

            <button
              type="button"
              onClick={() => setMode("single_track")}
              className={`flex items-center gap-1 rounded px-2.5 py-1 transition ${
                mode === "single_track" ? "bg-indigo-600 text-white" : "text-slate-400 hover:text-slate-200"
              }`}
            >
              <Target className="h-3 w-3" />
              <span>Single-Track Repaint</span>
            </button>

            <button
              type="button"
              onClick={() => setMode("add_instrument")}
              className={`flex items-center gap-1 rounded px-2.5 py-1 transition ${
                mode === "add_instrument"
                  ? "bg-emerald-600 text-white"
                  : "text-emerald-300 hover:bg-emerald-500/10"
              }`}
            >
              <PlusCircle className="h-3 w-3" />
              <span>+ Add Instrument Track</span>
            </button>
          </div>

          <button
            type="button"
            onClick={onClear}
            className="rounded p-1 text-slate-400 hover:bg-studio-surface hover:text-white"
            title="Clear Selection"
          >
            <X className="h-4 w-4" />
          </button>
        </div>
      </div>

      {/* Mode-Specific Inputs */}
      {mode === "add_instrument" ? (
        <div className="mt-2.5 flex flex-wrap items-center gap-2.5">
          <input
            type="text"
            value={newTrackName}
            onChange={(e) => setNewTrackName(e.target.value)}
            placeholder="Track Name (e.g., Electric Guitar, Lead Synth)"
            className="w-52 rounded-md border border-studio-border bg-studio-bg px-3 py-1.5 text-xs font-medium text-slate-100 placeholder-slate-500 focus:border-emerald-500 focus:outline-none"
          />
          <input
            type="text"
            value={instrumentPrompt}
            onChange={(e) => {
              instrumentPromptEditedRef.current = true;
              setInstrumentPrompt(e.target.value);
            }}
            placeholder="Instrument musical prompt (e.g., melodic electric guitar riff, 120 BPM, A minor)..."
            className="flex-1 min-w-[260px] rounded-md border border-studio-border bg-studio-bg px-3 py-1.5 text-xs text-slate-200 placeholder-slate-500 focus:border-emerald-500 focus:outline-none"
          />
          <button
            type="submit"
            disabled={busy || !newTrackName.trim() || !instrumentPrompt.trim()}
            className="rounded-md bg-emerald-500 px-4 py-1.5 text-xs font-bold text-slate-950 hover:bg-emerald-400 disabled:opacity-40 transition"
          >
            {busy ? "Generating Layer..." : "Create Instrument Layer"}
          </button>
        </div>
      ) : (
        <div className="mt-2.5 flex flex-wrap items-center gap-2.5">
          {mode === "single_track" && (
            <select
              value={targetTrackId}
              onChange={(e) => setTargetTrackId(e.target.value)}
              className="rounded-md border border-studio-border bg-studio-bg px-2.5 py-1.5 text-xs font-medium text-cyan-300 focus:border-cyan-500 focus:outline-none"
            >
              {(editableTracks.length > 0 ? editableTracks : tracks).map((t) => (
                <option key={t.track_id} value={t.track_id}>
                  Target: {t.name} ({t.stem_role})
                </option>
              ))}
            </select>
          )}

          <input
            type="text"
            value={promptOverride}
            onChange={(e) => setPromptOverride(e.target.value)}
            placeholder={
              mode === "full_mix"
                ? "Style override for full mix section (e.g., soaring synth brass, heavy drop)..."
                : "Track-specific prompt override (other stems stay bit-identical)..."
            }
            className="flex-1 min-w-[240px] rounded-md border border-studio-border bg-studio-bg px-3 py-1.5 text-xs text-slate-200 placeholder-slate-500 focus:border-cyan-500 focus:outline-none"
          />

          <input
            type="text"
            value={lyricsOverride}
            onChange={(e) => setLyricsOverride(e.target.value)}
            placeholder="Optional lyric override for selected region..."
            className="flex-1 min-w-[200px] rounded-md border border-studio-border bg-studio-bg px-3 py-1.5 text-xs text-slate-200 placeholder-slate-500 focus:border-cyan-500 focus:outline-none"
          />

          {mode === "full_mix" && (
            <label className="flex items-center gap-1.5 text-xs text-slate-300 cursor-pointer">
              <input
                type="checkbox"
                checked={restemWindow}
                onChange={(e) => setRestemWindow(e.target.checked)}
                className="rounded border-studio-border bg-studio-bg"
              />
              <span>2s Windowed Demucs Re-Sync</span>
            </label>
          )}

          <button
            type="submit"
            disabled={busy}
            className="rounded-md bg-cyan-500 px-4 py-1.5 text-xs font-bold text-slate-950 hover:bg-cyan-400 disabled:opacity-40 transition"
          >
            {busy
              ? "Processing..."
              : mode === "full_mix"
              ? "Repaint Full Mix"
              : "Repaint Target Track"}
          </button>
        </div>
      )}
    </form>
  );
};
