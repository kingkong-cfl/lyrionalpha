import React, { useState } from "react";
import { Activity, Cpu, HardDrive, KeyRound, Link2, RefreshCw, Wifi, WifiOff } from "lucide-react";
import type { HealthResponse } from "../../types/project";

interface GpuConnectionBarProps {
  serverUrl: string;
  onUpdateServerUrl: (url: string) => void;
  apiToken: string;
  onUpdateApiToken: (token: string) => void;
  health: HealthResponse | null;
  latencyMs: number | null;
  connectionError: string | null;
  isChecking: boolean;
  onPingNow: () => void;
}

export const GpuConnectionBar: React.FC<GpuConnectionBarProps> = ({
  serverUrl,
  onUpdateServerUrl,
  apiToken,
  onUpdateApiToken,
  health,
  latencyMs,
  connectionError,
  isChecking,
  onPingNow,
}) => {
  const [draftUrl, setDraftUrl] = useState(serverUrl);
  const [dirty, setDirty] = useState(false);
  const [draftToken, setDraftToken] = useState(apiToken);
  const [tokenDirty, setTokenDirty] = useState(false);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onUpdateServerUrl(dirty ? draftUrl : serverUrl);
    onUpdateApiToken(tokenDirty ? draftToken : apiToken);
    setDirty(false);
    setTokenDirty(false);
  };

  const vramPercent =
    health && health.vram_total_mb > 0
      ? Math.min(100, Math.round((health.vram_used_mb / health.vram_total_mb) * 100))
      : 0;

  return (
    <header className="flex flex-wrap items-center justify-between gap-3 border-b border-studio-border bg-studio-panel px-4 py-2.5">
      {/* Brand */}
      <div className="flex items-center gap-2.5">
        <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-gradient-to-br from-indigo-500 to-cyan-400 font-bold text-slate-950 shadow-lg shadow-indigo-500/20">
          L
        </div>
        <div>
          <div className="flex items-center gap-2">
            <h1 className="text-sm font-bold tracking-wide text-slate-100">LYRION STUDIO</h1>
            <span className="rounded bg-indigo-500/20 px-1.5 py-0.5 font-mono text-[10px] font-semibold text-indigo-300">
              AI DAW
            </span>
          </div>
          <p className="text-[11px] text-slate-400">Zero-Cost Split Architecture • Colab L4 + React</p>
        </div>
      </div>

      {/* Ephemeral Colab Cloudflared URL & API Token bar */}
      <form onSubmit={handleSubmit} className="flex flex-1 max-w-2xl items-center gap-2">
        <div className="relative flex flex-1 items-center">
          <Link2 className="pointer-events-none absolute left-2.5 h-3.5 w-3.5 text-slate-400" />
          <input
            type="url"
            value={dirty ? draftUrl : serverUrl}
            onChange={(e) => {
              setDraftUrl(e.target.value);
              setDirty(true);
            }}
            placeholder="https://xxxx.trycloudflare.com or http://localhost:8000"
            className="w-full rounded-md border border-studio-border bg-studio-bg py-1.5 pl-8 pr-2 font-mono text-xs text-slate-200 placeholder-slate-500 focus:border-indigo-500 focus:outline-none"
          />
        </div>
        <div className="relative flex w-44 items-center">
          <KeyRound className="pointer-events-none absolute left-2.5 h-3.5 w-3.5 text-slate-400" />
          <input
            type="password"
            value={tokenDirty ? draftToken : apiToken}
            onChange={(e) => {
              setDraftToken(e.target.value);
              setTokenDirty(true);
            }}
            placeholder="API Token (Colab)"
            title="Optional LYRION_API_TOKEN printed by Colab bootstrap"
            className="w-full rounded-md border border-studio-border bg-studio-bg py-1.5 pl-8 pr-2 font-mono text-xs text-slate-200 placeholder-slate-500 focus:border-indigo-500 focus:outline-none"
          />
        </div>
        <button
          type="submit"
          className="rounded bg-indigo-600 px-3 py-1.5 text-[11px] font-medium text-white hover:bg-indigo-500 transition shrink-0"
        >
          Connect
        </button>
        <button
          type="button"
          onClick={onPingNow}
          title="Ping GPU backend health"
          className="flex h-7 w-7 shrink-0 items-center justify-center rounded border border-studio-border bg-studio-surface text-slate-300 hover:border-indigo-500 hover:text-white transition"
        >
          <RefreshCw className={`h-3.5 w-3.5 ${isChecking ? "animate-spin text-indigo-400" : ""}`} />
        </button>
      </form>

      {/* Live L4 Telemetry & Drive Status */}
      <div className="flex flex-wrap items-center gap-3 text-xs">
        {health ? (
          <>
            <div className="flex items-center gap-1.5 rounded-full border border-emerald-500/30 bg-emerald-500/10 px-2.5 py-1 text-emerald-300">
              <Wifi className="h-3.5 w-3.5" />
              <span className="font-medium">{health.mock_gpu ? "Mock GPU" : "Colab Online"}</span>
              {latencyMs !== null && <span className="font-mono text-[10px] opacity-80">{latencyMs}ms</span>}
            </div>

            <div className="hidden md:flex items-center gap-2 rounded-md border border-studio-border bg-studio-bg px-2.5 py-1">
              <Cpu className="h-3.5 w-3.5 text-cyan-400" />
              <div className="flex flex-col">
                <span className="text-[10px] text-slate-400 leading-none">{health.gpu_name}</span>
                <div className="mt-1 flex items-center gap-1.5">
                  <div className="h-1.5 w-20 overflow-hidden rounded-full bg-slate-800">
                    <div
                      className="h-full bg-gradient-to-r from-cyan-400 to-indigo-500 transition-all"
                      style={{ width: `${vramPercent}%` }}
                    />
                  </div>
                  <span className="font-mono text-[10px] text-slate-300">
                    {(health.vram_used_mb / 1024).toFixed(1)}/{(health.vram_total_mb / 1024).toFixed(0)}GB
                  </span>
                </div>
              </div>
            </div>

            <div className="hidden xl:flex items-center gap-1.5 rounded-md border border-studio-border bg-studio-bg px-2.5 py-1 text-slate-300">
              <Activity className="h-3.5 w-3.5 text-indigo-400" />
              <span className="font-mono text-[11px]">VRAM: {health.active_model}</span>
            </div>

            <div
              title={health.drive_root}
              className={`hidden sm:flex items-center gap-1.5 rounded-md border px-2 py-1 ${
                health.drive_writable
                  ? "border-studio-border bg-studio-bg text-slate-300"
                  : "border-rose-500/40 bg-rose-500/10 text-rose-300"
              }`}
            >
              <HardDrive className="h-3.5 w-3.5 text-amber-400" />
              <span className="text-[11px]">{health.drive_writable ? "Drive Synced" : "Drive Read-Only"}</span>
            </div>
          </>
        ) : (
          <div className="flex items-center gap-1.5 rounded-full border border-rose-500/40 bg-rose-500/10 px-3 py-1 text-rose-300">
            <WifiOff className="h-3.5 w-3.5" />
            <span>{connectionError || "Disconnected"}</span>
          </div>
        )}
      </div>
    </header>
  );
};
