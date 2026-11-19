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

const MAX_TILE_WIDTH = 4096;

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
  const containerRef = useRef<HTMLDivElement | null>(null);
  const tileCanvasRefs = useRef<(HTMLCanvasElement | null)[]>([]);
  const dragStartSecRef = useRef<number | null>(null);

  const durationSec = Math.max(grid.duration_sec || 30, 1);
  const totalWidth = Math.max(200, Math.round(timelineWidthPx));
  const height = 44;
  const tileCount = Math.max(1, Math.ceil(totalWidth / MAX_TILE_WIDTH));

  useEffect(() => {
    tileCanvasRefs.current.length = tileCount;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const beats = grid.beat_timestamps;
    const downbeats = new Set(grid.downbeat_beat_indices);
    const downbeatBarNumber = new Map<number, number>();
    let barCounter = 1;
    beats.forEach((_, idx) => {
      if (downbeats.has(idx)) {
        downbeatBarNumber.set(idx, barCounter);
        barCounter += 1;
      }
    });

    for (let tileIdx = 0; tileIdx < tileCount; tileIdx++) {
      const canvas = tileCanvasRefs.current[tileIdx];
      if (!canvas) continue;

      const tileStart = tileIdx * MAX_TILE_WIDTH;
      const tileWidth = Math.min(MAX_TILE_WIDTH, totalWidth - tileStart);
      if (tileWidth <= 0) continue;

      canvas.width = Math.max(1, Math.round(tileWidth * dpr));
      canvas.height = Math.round(height * dpr);
      const ctx = canvas.getContext("2d");
      if (!ctx) continue;

      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, tileWidth, height);

      // Translate world X coordinates into tile local coordinates
      ctx.save();
      ctx.translate(-tileStart, 0);

      // Background
      ctx.fillStyle = "#0e131d";
      ctx.fillRect(tileStart, 0, tileWidth, height);

      // Selected region highlight
      if (selection) {
        const x1 = selection.startSec * pixelsPerSec;
        const x2 = selection.endSec * pixelsPerSec;
        if (x2 >= tileStart && x1 <= tileStart + tileWidth) {
          ctx.fillStyle = "rgba(34, 211, 238, 0.16)";
          ctx.fillRect(x1, 0, Math.max(2, x2 - x1), height);
          ctx.fillStyle = "#22d3ee";
          ctx.fillRect(x1, 0, Math.max(2, x2 - x1), 3);
        }
      }

      if (beats.length > 0) {
        ctx.font = "600 10px ui-monospace, SFMono-Regular, monospace";
        beats.forEach((t, idx) => {
          const x = Math.round(t * pixelsPerSec) + 0.5;
          if (x < tileStart - 80 || x > tileStart + tileWidth + 80) return;
          const isDownbeat = downbeats.has(idx);
          ctx.strokeStyle = isDownbeat ? "#64748b" : "#1e293b";
          ctx.lineWidth = isDownbeat ? 1.25 : 1;
          ctx.beginPath();
          ctx.moveTo(x, isDownbeat ? 6 : 24);
          ctx.lineTo(x, height);
          ctx.stroke();

          if (isDownbeat) {
            ctx.fillStyle = "#cbd5e1";
            ctx.fillText(`Bar ${downbeatBarNumber.get(idx) ?? 1}`, x + 4, 17);
          }
        });
      } else {
        ctx.font = "10px ui-monospace, SFMono-Regular, monospace";
        for (let s = 0; s <= Math.ceil(durationSec); s++) {
          const x = Math.round(s * pixelsPerSec) + 0.5;
          if (x < tileStart - 60 || x > tileStart + tileWidth + 60) continue;
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
      ctx.moveTo(tileStart, height - 0.5);
      ctx.lineTo(tileStart + tileWidth, height - 0.5);
      ctx.stroke();

      ctx.restore();
    }
  }, [grid, pixelsPerSec, totalWidth, selection, durationSec, tileCount]);

  const timeFromPointer = (clientX: number): number => {
    const el = containerRef.current;
    if (!el) return 0;
    const rect = el.getBoundingClientRect();
    const x = Math.max(0, clientX - rect.left);
    return Math.min(durationSec, x / pixelsPerSec);
  };

  const handleMouseDown = (e: React.MouseEvent<HTMLDivElement>) => {
    const t = timeFromPointer(e.clientX);
    if (e.shiftKey) {
      dragStartSecRef.current = t;
    } else {
      dragStartSecRef.current = t;
      onSeek(t);
    }
  };

  const handleMouseMove = (e: React.MouseEvent<HTMLDivElement>) => {
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
    <div
      ref={containerRef}
      onMouseDown={handleMouseDown}
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
      onMouseLeave={handleMouseUp}
      title="Click to seek • Drag across ruler to select a beat-snapped Repaint region"
      className="relative flex select-none overflow-hidden cursor-crosshair"
      style={{ width: `${totalWidth}px`, height: `${height}px` }}
    >
      {Array.from({ length: tileCount }).map((_, idx) => {
        const tileWidth = Math.min(MAX_TILE_WIDTH, totalWidth - idx * MAX_TILE_WIDTH);
        return (
          <canvas
            key={idx}
            ref={(el) => {
              tileCanvasRefs.current[idx] = el;
            }}
            style={{ width: `${tileWidth}px`, height: `${height}px` }}
            className="block shrink-0"
          />
        );
      })}
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
