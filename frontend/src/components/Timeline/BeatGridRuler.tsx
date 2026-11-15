import React, { useEffect, useRef } from "react";
import { buildSnappedRegion } from "../../audio/snapUtils";
import type { BeatGridSchema, RegionSelection, SnapMode } from "../../types/project";

interface BeatGridRulerProps {
  grid: BeatGridSchema;
  pixelsPerSec: number;
  timelineWidthPx: number;
  snapMode: SnapMode;
  selection: RegionSelection | null;
  playheadRef?: React.RefObject<HTMLDivElement>;
  onSeek: (sec: number) => void;
  onSelectRegion: (region: RegionSelection | null) => void;
}

export const BeatGridRuler: React.FC<BeatGridRulerProps> = React.memo(({
  grid,
  pixelsPerSec,
  timelineWidthPx,
  snapMode,
  selection,
  playheadRef,
  onSeek,
  onSelectRegion,
}) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const dragStartSecRef = useRef<number | null>(null);

  const durationSec = Math.max(grid.duration_sec || 16, 1);
  const totalWidth = Math.max(200, Math.round(timelineWidthPx));
  const height = 44;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = totalWidth * dpr;
    canvas.height = height * dpr;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, totalWidth, height);

    // Background
    ctx.fillStyle = "#0e131d";
    ctx.fillRect(0, 0, totalWidth, height);

    // Selected region highlight
    if (selection) {
      const x1 = selection.startSec * pixelsPerSec;
      const x2 = selection.endSec * pixelsPerSec;
      ctx.fillStyle = "rgba(34, 211, 238, 0.16)";
      ctx.fillRect(x1, 0, Math.max(2, x2 - x1), height);
      ctx.fillStyle = "#22d3ee";
      ctx.fillRect(x1, 0, Math.max(2, x2 - x1), 3);
    }

    // Beat & Bar ticks from backend beat_timestamps[]
    const beats = grid.beat_timestamps;
    const downbeats = new Set(grid.downbeat_beat_indices);

    if (beats.length > 0) {
      ctx.font = "600 10px ui-monospace, SFMono-Regular, monospace";
      let barNum = 1;
      beats.forEach((t, idx) => {
        const x = Math.round(t * pixelsPerSec) + 0.5;
        const isDownbeat = downbeats.has(idx);
        ctx.strokeStyle = isDownbeat ? "#64748b" : "#1e293b";
        ctx.lineWidth = isDownbeat ? 1.25 : 1;
        ctx.beginPath();
        ctx.moveTo(x, isDownbeat ? 6 : 24);
        ctx.lineTo(x, height);
        ctx.stroke();

        if (isDownbeat) {
          ctx.fillStyle = "#cbd5e1";
          ctx.fillText(`Bar ${barNum}`, x + 4, 17);
          barNum += 1;
        }
      });
    } else {
      // Fallback 1s ruler before first song generation
      ctx.font = "10px ui-monospace, SFMono-Regular, monospace";
      for (let s = 0; s <= Math.ceil(durationSec); s++) {
        const x = Math.round(s * pixelsPerSec) + 0.5;
        ctx.strokeStyle = s % 4 === 0 ? "#475569" : "#1e293b";
        ctx.beginPath();
        ctx.moveTo(x, s % 4 === 0 ? 10 : 26);
        ctx.lineTo(x, height);
        ctx.stroke();
        if (s % 4 === 0) {
          ctx.fillStyle = "#94a3b8";
          ctx.fillText(`${s}s`, x + 4, 18);
        }
      }
    }

    // Bottom border
    ctx.strokeStyle = "#263044";
    ctx.beginPath();
    ctx.moveTo(0, height - 0.5);
    ctx.lineTo(totalWidth, height - 0.5);
    ctx.stroke();
  }, [grid, pixelsPerSec, totalWidth, selection]);

  const timeFromPointer = (clientX: number): number => {
    const canvas = canvasRef.current;
    if (!canvas) return 0;
    const rect = canvas.getBoundingClientRect();
    const x = Math.max(0, clientX - rect.left);
    return Math.min(durationSec, x / pixelsPerSec);
  };

  const handleMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const t = timeFromPointer(e.clientX);
    if (e.shiftKey) {
      // Shift+Drag selects a snapped Repaint region directly on the ruler
      dragStartSecRef.current = t;
    } else {
      dragStartSecRef.current = t;
      onSeek(t);
    }
  };

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (dragStartSecRef.current === null || (e.buttons & 1) === 0) return;
    const currentT = timeFromPointer(e.clientX);
    if (Math.abs(currentT - dragStartSecRef.current) > 0.08) {
      const region = buildSnappedRegion(dragStartSecRef.current, currentT, grid, snapMode);
      onSelectRegion(region);
    }
  };

  const handleMouseUp = () => {
    dragStartSecRef.current = null;
  };

  return (
    <div className="relative select-none overflow-hidden" style={{ width: `${totalWidth}px` }}>
      <canvas
        ref={canvasRef}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
        title="Click to seek • Drag across ruler to select a beat-snapped Repaint region"
        style={{ width: `${totalWidth}px`, height: `${height}px` }}
        className="block cursor-crosshair"
      />
      {playheadRef && (
        <div
          ref={playheadRef}
          style={{ transform: "translate3d(0px, 0, 0)" }}
          className="pointer-events-none absolute top-0 bottom-0 left-0 z-20 will-change-transform"
        >
          <div className="absolute -left-[5.5px] top-0 h-0 w-0 border-x-[6px] border-t-[10px] border-x-transparent border-t-rose-500" />
          <div className="h-full w-[1.5px] bg-rose-500" />
        </div>
      )}
    </div>
  );
});
