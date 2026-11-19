import React from "react";
import { RotateCcw, Save, Sliders, X } from "lucide-react";
import type { TrackDspChain, TrackSchema } from "../../types/project";

interface PedalboardDrawerProps {
  track: TrackSchema;
  onUpdateTrack: (trackId: string, patch: Partial<TrackSchema>) => void;
  onSaveToDrive: () => Promise<void>;
  onClose: () => void;
  saving: boolean;
  busy?: boolean;
}

const DEFAULT_DSP: TrackDspChain = {
  eq: { low_shelf_db: 0, mid_gain_db: 0, mid_freq_hz: 1000, high_shelf_db: 0 },
  compressor: { enabled: false, threshold_db: -18, ratio: 2.5, attack_ms: 10, release_ms: 100 },
  reverb: { enabled: false, room_size: 0.4, damping: 0.5, wet_level: 0.15 },
};

export const PedalboardDrawer: React.FC<PedalboardDrawerProps> = ({
  track,
  onUpdateTrack,
  onSaveToDrive,
  onClose,
  saving,
  busy = false,
}) => {
  const dsp = track.dsp_chain;

  const updateDsp = (patch: Partial<TrackDspChain>) => {
    onUpdateTrack(track.track_id, {
      dsp_chain: {
        ...dsp,
        ...patch,
      },
    });
  };

  return (
    <aside className="flex w-80 shrink-0 flex-col border-l border-studio-border bg-studio-panel overflow-y-auto">
      <div className="flex items-center justify-between border-b border-studio-border px-4 py-3">
        <div className="flex items-center gap-2">
          <Sliders className="h-4 w-4 text-indigo-400" />
          <div>
            <h3 className="text-xs font-bold uppercase tracking-wider text-slate-100">
              Pedalboard DSP
            </h3>
            <p className="text-[11px] text-slate-400">{track.name}</p>
          </div>
        </div>
        <button
          type="button"
          onClick={onClose}
          className="rounded p-1 text-slate-400 hover:bg-studio-surface hover:text-white"
        >
          <X className="h-4 w-4" />
        </button>
      </div>

      <div className="flex-1 space-y-5 p-4 text-xs">
        {/* 3-Band EQ */}
        <section className="rounded-lg border border-studio-border bg-studio-bg p-3">
          <div className="mb-3 flex items-center justify-between">
            <span className="font-semibold text-cyan-300">3-Band Channel EQ</span>
            <span className="font-mono text-[10px] text-slate-500">Live + Export</span>
          </div>

          <div className="space-y-3">
            <div>
              <div className="mb-1 flex justify-between text-[11px]">
                <span className="text-slate-400">Low Shelf (140 Hz)</span>
                <span className="font-mono text-slate-200">{dsp.eq.low_shelf_db.toFixed(1)} dB</span>
              </div>
              <input
                type="range"
                min={-12}
                max={12}
                step={0.5}
                value={dsp.eq.low_shelf_db}
                onChange={(e) => updateDsp({ eq: { ...dsp.eq, low_shelf_db: Number(e.target.value) } })}
                className="w-full"
              />
            </div>

            <div>
              <div className="mb-1 flex justify-between text-[11px]">
                <span className="text-slate-400">Mid Peak ({Math.round(dsp.eq.mid_freq_hz)} Hz)</span>
                <span className="font-mono text-slate-200">{dsp.eq.mid_gain_db.toFixed(1)} dB</span>
              </div>
              <input
                type="range"
                min={-12}
                max={12}
                step={0.5}
                value={dsp.eq.mid_gain_db}
                onChange={(e) => updateDsp({ eq: { ...dsp.eq, mid_gain_db: Number(e.target.value) } })}
                className="w-full"
              />
              <input
                type="range"
                min={250}
                max={4000}
                step={50}
                value={dsp.eq.mid_freq_hz}
                onChange={(e) => updateDsp({ eq: { ...dsp.eq, mid_freq_hz: Number(e.target.value) } })}
                className="mt-1 w-full opacity-75"
              />
            </div>

            <div>
              <div className="mb-1 flex justify-between text-[11px]">
                <span className="text-slate-400">High Shelf (4.5 kHz)</span>
                <span className="font-mono text-slate-200">{dsp.eq.high_shelf_db.toFixed(1)} dB</span>
              </div>
              <input
                type="range"
                min={-12}
                max={12}
                step={0.5}
                value={dsp.eq.high_shelf_db}
                onChange={(e) => updateDsp({ eq: { ...dsp.eq, high_shelf_db: Number(e.target.value) } })}
                className="w-full"
              />
            </div>
          </div>
        </section>

        {/* Compressor */}
        <section className="rounded-lg border border-studio-border bg-studio-bg p-3">
          <div className="mb-3 flex items-center justify-between">
            <div>
              <span className="font-semibold text-amber-300">Pedalboard Compressor</span>
              <span className="ml-2 font-mono text-[10px] text-slate-500">Live Preview + Export</span>
            </div>
            <label className="flex items-center gap-1.5 cursor-pointer">
              <input
                type="checkbox"
                checked={dsp.compressor.enabled}
                onChange={(e) =>
                  updateDsp({ compressor: { ...dsp.compressor, enabled: e.target.checked } })
                }
              />
              <span className="text-[11px] text-slate-300">Active</span>
            </label>
          </div>

          <div className={`space-y-2.5 ${dsp.compressor.enabled ? "" : "opacity-45 pointer-events-none"}`}>
            <div>
              <div className="mb-1 flex justify-between text-[11px]">
                <span className="text-slate-400">Threshold</span>
                <span className="font-mono text-slate-200">{dsp.compressor.threshold_db} dB</span>
              </div>
              <input
                type="range"
                min={-40}
                max={0}
                step={1}
                value={dsp.compressor.threshold_db}
                onChange={(e) =>
                  updateDsp({ compressor: { ...dsp.compressor, threshold_db: Number(e.target.value) } })
                }
                className="w-full"
              />
            </div>
            <div>
              <div className="mb-1 flex justify-between text-[11px]">
                <span className="text-slate-400">Ratio</span>
                <span className="font-mono text-slate-200">{dsp.compressor.ratio.toFixed(1)}:1</span>
              </div>
              <input
                type="range"
                min={1}
                max={12}
                step={0.5}
                value={dsp.compressor.ratio}
                onChange={(e) =>
                  updateDsp({ compressor: { ...dsp.compressor, ratio: Number(e.target.value) } })
                }
                className="w-full"
              />
            </div>
          </div>
        </section>

        {/* Reverb */}
        <section className="rounded-lg border border-studio-border bg-studio-bg p-3">
          <div className="mb-3 flex items-center justify-between">
            <div>
              <span className="font-semibold text-purple-300">Studio Reverb</span>
              <span className="ml-2 font-mono text-[10px] text-slate-500">Live Preview + Export</span>
            </div>
            <label className="flex items-center gap-1.5 cursor-pointer">
              <input
                type="checkbox"
                checked={dsp.reverb.enabled}
                onChange={(e) =>
                  updateDsp({ reverb: { ...dsp.reverb, enabled: e.target.checked } })
                }
              />
              <span className="text-[11px] text-slate-300">Active</span>
            </label>
          </div>

          <div className={`space-y-2.5 ${dsp.reverb.enabled ? "" : "opacity-45 pointer-events-none"}`}>
            <div>
              <div className="mb-1 flex justify-between text-[11px]">
                <span className="text-slate-400">Room Size</span>
                <span className="font-mono text-slate-200">{Math.round(dsp.reverb.room_size * 100)}%</span>
              </div>
              <input
                type="range"
                min={0}
                max={1}
                step={0.05}
                value={dsp.reverb.room_size}
                onChange={(e) =>
                  updateDsp({ reverb: { ...dsp.reverb, room_size: Number(e.target.value) } })
                }
                className="w-full"
              />
            </div>
            <div>
              <div className="mb-1 flex justify-between text-[11px]">
                <span className="text-slate-400">Wet Mix</span>
                <span className="font-mono text-slate-200">{Math.round(dsp.reverb.wet_level * 100)}%</span>
              </div>
              <input
                type="range"
                min={0}
                max={0.8}
                step={0.02}
                value={dsp.reverb.wet_level}
                onChange={(e) =>
                  updateDsp({ reverb: { ...dsp.reverb, wet_level: Number(e.target.value) } })
                }
                className="w-full"
              />
            </div>
          </div>
        </section>
      </div>

      {/* Footer Actions */}
      <div className="flex gap-2 border-t border-studio-border p-4">
        <button
          type="button"
          onClick={() => updateDsp(DEFAULT_DSP)}
          className="flex items-center justify-center gap-1 rounded-md border border-studio-border bg-studio-bg px-3 py-2 text-xs text-slate-300 hover:text-white"
        >
          <RotateCcw className="h-3.5 w-3.5" />
          <span>Reset</span>
        </button>
        <button
          type="button"
          disabled={saving || busy}
          onClick={onSaveToDrive}
          title={busy ? "Wait for running GPU job to complete before saving mix" : "Save mixer & DSP settings to Drive"}
          className="flex flex-1 items-center justify-center gap-1.5 rounded-md bg-indigo-600 px-3 py-2 text-xs font-semibold text-white hover:bg-indigo-500 disabled:opacity-50"
        >
          <Save className="h-3.5 w-3.5" />
          <span>{saving ? "Saving..." : "Save Mix to Drive"}</span>
        </button>
      </div>
    </aside>
  );
};
