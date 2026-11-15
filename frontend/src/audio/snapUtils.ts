import type { BeatGridSchema, RegionSelection, SnapMode } from "../types/project";

export function clampSec(sec: number, grid: BeatGridSchema): number {
  return Math.max(0, Math.min(grid.duration_sec || 0, sec));
}

export function secToSample(sec: number, grid: BeatGridSchema): number {
  const sr = grid.sample_rate || 44100;
  const maxSamples = Math.max(0, grid.total_samples);
  return Math.max(0, Math.min(maxSamples, Math.round(sec * sr)));
}

export function formatBarBeatFromTime(sec: number, grid: BeatGridSchema): string {
  const beats = grid.beat_timestamps;
  if (!beats.length) {
    const beatDur = 60 / Math.max(40, grid.bpm || 120);
    const totalBeats = Math.max(0, Math.floor(sec / beatDur));
    const bar = Math.floor(totalBeats / 4) + 1;
    const beat = (totalBeats % 4) + 1;
    return `${bar}:${beat}`;
  }
  let beatIdx = 0;
  for (let i = 0; i < beats.length; i++) {
    if (beats[i] <= sec + 0.005) {
      beatIdx = i;
    } else {
      break;
    }
  }
  const beatsPerBar = parseInt((grid.time_signature || "4/4").split("/")[0], 10) || 4;
  const bar = Math.floor(beatIdx / beatsPerBar) + 1;
  const beat = (beatIdx % beatsPerBar) + 1;
  return `${bar}:${beat}`;
}

export function formatTimecode(sec: number): string {
  const safe = Math.max(0, sec);
  const mins = Math.floor(safe / 60);
  const secs = Math.floor(safe % 60);
  const ms = Math.floor((safe % 1) * 100);
  return `${String(mins).padStart(2, "0")}:${String(secs).padStart(2, "0")}.${String(ms).padStart(2, "0")}`;
}

export function snapPointToGrid(
  rawSec: number,
  grid: BeatGridSchema,
  mode: SnapMode
): { sec: number; sample: number } {
  const clamped = clampSec(rawSec, grid);
  if (mode === "off" || grid.beat_timestamps.length === 0) {
    return { sec: clamped, sample: secToSample(clamped, grid) };
  }

  const candidateIndices =
    mode === "bar" && grid.downbeat_beat_indices.length > 0
      ? grid.downbeat_beat_indices
      : grid.beat_timestamps.map((_, idx) => idx);

  let bestSec = clamped;
  let bestSample = secToSample(clamped, grid);
  let minDiff = Infinity;

  for (const idx of candidateIndices) {
    const t = grid.beat_timestamps[idx];
    const s = grid.beat_samples[idx];
    if (t === undefined || s === undefined) continue;
    const diff = Math.abs(t - clamped);
    if (diff < minDiff) {
      minDiff = diff;
      bestSec = t;
      bestSample = s;
    }
  }

  // Also allow snapping to exact track end
  const endDiff = Math.abs((grid.duration_sec || 0) - clamped);
  if (endDiff < minDiff) {
    bestSec = grid.duration_sec;
    bestSample = grid.total_samples;
  }

  return { sec: bestSec, sample: bestSample };
}

export function buildSnappedRegion(
  rawStartSec: number,
  rawEndSec: number,
  grid: BeatGridSchema,
  mode: SnapMode
): RegionSelection | null {
  const a = Math.min(rawStartSec, rawEndSec);
  const b = Math.max(rawStartSec, rawEndSec);

  const p1 = snapPointToGrid(a, grid, mode);
  let p2 = snapPointToGrid(b, grid, mode);

  // Guarantee minimum 1 beat or 0.5s width if start and end snapped to same point
  if (p2.sample <= p1.sample) {
    const stepSamples =
      mode === "bar"
        ? Math.round((4 * 60 * grid.sample_rate) / Math.max(40, grid.bpm))
        : Math.round((60 * grid.sample_rate) / Math.max(40, grid.bpm));
    const nextSample = Math.min(grid.total_samples, p1.sample + Math.max(stepSamples, 22050));
    p2 = {
      sample: nextSample,
      sec: nextSample / (grid.sample_rate || 44100),
    };
  }

  if (p2.sample <= p1.sample) return null;

  return {
    startSec: p1.sec,
    endSec: p2.sec,
    startSample: p1.sample,
    endSample: p2.sample,
    startLabel: formatBarBeatFromTime(p1.sec, grid),
    endLabel: formatBarBeatFromTime(p2.sec, grid),
  };
}

export function buildRegionFromSeconds(
  rawStartSec: number,
  rawEndSec: number,
  grid: BeatGridSchema
): RegionSelection | null {
  const duration = Math.max(0.1, grid.duration_sec || 0);
  const startSec = Math.max(0, Math.min(duration - 0.05, rawStartSec));
  const endSec = Math.max(startSec + 0.05, Math.min(duration, rawEndSec));
  const startSample = secToSample(startSec, grid);
  const endSample = secToSample(endSec, grid);
  if (endSample <= startSample) return null;
  const sr = grid.sample_rate || 44100;
  return {
    startSec: Number((startSample / sr).toFixed(4)),
    endSec: Number((endSample / sr).toFixed(4)),
    startSample,
    endSample,
    startLabel: formatBarBeatFromTime(startSample / sr, grid),
    endLabel: formatBarBeatFromTime(endSample / sr, grid),
  };
}
