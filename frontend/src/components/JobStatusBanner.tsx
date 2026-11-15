import React from "react";
import { Loader2 } from "lucide-react";
import type { JobStatusResponse } from "../types/project";

interface JobStatusBannerProps {
  job: JobStatusResponse | null;
}

export const JobStatusBanner: React.FC<JobStatusBannerProps> = ({ job }) => {
  if (!job || (job.status !== "queued" && job.status !== "running")) return null;

  const pct = Math.round((job.progress || 0) * 100);

  return (
    <div className="border-b border-indigo-500/30 bg-indigo-950/40 px-4 py-2">
      <div className="flex items-center justify-between gap-4 text-xs">
        <div className="flex items-center gap-2 text-indigo-200">
          <Loader2 className="h-3.5 w-3.5 animate-spin text-cyan-400" />
          <span className="font-semibold uppercase tracking-wide text-cyan-300">{job.job_type}</span>
          <span className="text-slate-300">{job.stage}</span>
        </div>
        <div className="flex items-center gap-3 font-mono text-[11px] text-slate-300">
          <span>VRAM: {(job.vram_allocated_mb / 1024).toFixed(1)} GB</span>
          <span className="text-cyan-300">{pct}%</span>
        </div>
      </div>
      <div className="mt-1.5 h-1.5 w-full overflow-hidden rounded-full bg-slate-800">
        <div
          className="h-full bg-gradient-to-r from-indigo-500 via-cyan-400 to-emerald-400 transition-all duration-300"
          style={{ width: `${Math.max(4, pct)}%` }}
        />
      </div>
    </div>
  );
};
