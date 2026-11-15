import React, { useEffect, useState } from "react";
import { FolderPlus, Music2, Sparkles, Wand2 } from "lucide-react";
import type { ProjectSchema } from "../../types/project";

interface ProducerPanelProps {
  projects: ProjectSchema[];
  activeProject: ProjectSchema | null;
  onSelectProject: (projectId: string) => void;
  onCreateProject: (name: string) => Promise<void>;
  onEnhancePrompt: (rawPrompt: string, durationSec: number) => Promise<{
    bpm: number;
    musical_key: string;
    duration_sec: number;
    style_tags: string[];
    structured_lyrics: string;
    provider: string;
  }>;
  onGenerateSong: (params: {
    raw_prompt: string;
    style_tags: string[];
    structured_lyrics: string;
    bpm: number;
    musical_key: string;
    duration_sec: number;
    seed: number;
    separate_stems: boolean;
  }) => Promise<void>;
  busy: boolean;
  connected: boolean;
}

export const ProducerPanel: React.FC<ProducerPanelProps> = ({
  projects,
  activeProject,
  onSelectProject,
  onCreateProject,
  onEnhancePrompt,
  onGenerateSong,
  busy,
  connected,
}) => {
  const [newProjectName, setNewProjectName] = useState("Neon Horizon");
  const [rawPrompt, setRawPrompt] = useState(
    "Melancholic synthwave night drive with punchy retro drums, analog bassline, and expressive vocals"
  );
  const [styleTagsText, setStyleTagsText] = useState(
    "synthwave, retrowave, analog bass, expressive vocals, wide stereo"
  );
  const [structuredLyrics, setStructuredLyrics] = useState(
    "[Intro]\n(Warm analog arpeggio)\n\n[Verse 1]\nStreetlights blur across the midnight glass\nCounting down the miles as shadows pass\n\n[Chorus]\nOut on the neon horizon we run\nChasing the pulse till the rising sun"
  );
  const [bpm, setBpm] = useState(120);
  const [musicalKey, setMusicalKey] = useState("A minor");
  const [durationSec, setDurationSec] = useState(16);
  const [seed, setSeed] = useState(421337);
  const [separateStems, setSeparateStems] = useState(true);
  const [providerBadge, setProviderBadge] = useState<string | null>(null);
  const [enhancing, setEnhancing] = useState(false);

  useEffect(() => {
    if (!activeProject) return;
    setBpm(activeProject.grid.bpm || 120);
    setMusicalKey(activeProject.grid.musical_key || "A minor");
    if (activeProject.grid.duration_sec > 0) {
      setDurationSec(Math.round(activeProject.grid.duration_sec));
    }
    if (activeProject.producer_spec.raw_prompt) {
      setRawPrompt(activeProject.producer_spec.raw_prompt);
    }
    if (activeProject.producer_spec.style_tags.length > 0) {
      setStyleTagsText(activeProject.producer_spec.style_tags.join(", "));
    }
    if (activeProject.producer_spec.structured_lyrics) {
      setStructuredLyrics(activeProject.producer_spec.structured_lyrics);
    }
    setSeed(activeProject.producer_spec.seed || 421337);
  }, [
    activeProject?.project_id,
    activeProject?.grid.bpm,
    activeProject?.grid.duration_sec,
    activeProject?.grid.musical_key,
    activeProject?.producer_spec.seed,
  ]);

  const handleEnhance = async () => {
    if (!rawPrompt.trim()) return;
    setEnhancing(true);
    try {
      const res = await onEnhancePrompt(rawPrompt, durationSec);
      setBpm(res.bpm);
      setMusicalKey(res.musical_key);
      setDurationSec(Math.round(res.duration_sec));
      setStyleTagsText(res.style_tags.join(", "));
      setStructuredLyrics(res.structured_lyrics);
      setProviderBadge(res.provider);
    } finally {
      setEnhancing(false);
    }
  };

  const handleGenerate = async (e: React.FormEvent) => {
    e.preventDefault();
    const tags = styleTagsText
      .split(",")
      .map((s) => s.trim())
      .filter(Boolean);
    await onGenerateSong({
      raw_prompt: rawPrompt,
      style_tags: tags,
      structured_lyrics: structuredLyrics,
      bpm,
      musical_key: musicalKey,
      duration_sec: durationSec,
      seed,
      separate_stems: separateStems,
    });
  };

  return (
    <aside className="flex h-full w-full lg:w-80 xl:w-96 shrink-0 flex-col border-r border-studio-border bg-studio-panel overflow-y-auto">
      {/* Project Switcher */}
      <div className="border-b border-studio-border p-4">
        <div className="mb-2 flex items-center justify-between">
          <span className="text-xs font-semibold uppercase tracking-wider text-slate-400">
            Drive Projects
          </span>
          <span className="font-mono text-[10px] text-slate-500">{projects.length} saved</span>
        </div>

        <div className="flex gap-2 mb-2.5">
          <select
            value={activeProject?.project_id || ""}
            onChange={(e) => onSelectProject(e.target.value)}
            disabled={!connected || projects.length === 0}
            className="flex-1 rounded-md border border-studio-border bg-studio-bg px-2.5 py-1.5 text-xs text-slate-200 focus:border-indigo-500 focus:outline-none"
          >
            {projects.length === 0 ? (
              <option value="">No projects yet</option>
            ) : (
              projects.map((p) => (
                <option key={p.project_id} value={p.project_id}>
                  {p.name} ({p.tracks.length} tracks)
                </option>
              ))
            )}
          </select>
        </div>

        <div className="flex gap-2">
          <input
            type="text"
            value={newProjectName}
            onChange={(e) => setNewProjectName(e.target.value)}
            placeholder="New project name..."
            className="flex-1 rounded-md border border-studio-border bg-studio-bg px-2.5 py-1.5 text-xs text-slate-200 placeholder-slate-500 focus:border-indigo-500 focus:outline-none"
          />
          <button
            type="button"
            disabled={!connected || busy || !newProjectName.trim()}
            onClick={() => onCreateProject(newProjectName.trim())}
            className="flex items-center gap-1 rounded-md border border-indigo-500/40 bg-indigo-500/15 px-2.5 py-1.5 text-xs font-medium text-indigo-300 hover:bg-indigo-500/25 disabled:opacity-40 transition"
          >
            <FolderPlus className="h-3.5 w-3.5" />
            <span>New</span>
          </button>
        </div>
      </div>

      {/* Zero-VRAM AI Producer Form */}
      <form onSubmit={handleGenerate} className="flex flex-1 flex-col gap-3.5 p-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-1.5">
            <Sparkles className="h-4 w-4 text-cyan-400" />
            <h2 className="text-xs font-bold uppercase tracking-wider text-slate-200">
              AI Producer Spec
            </h2>
          </div>
          {providerBadge && (
            <span className="rounded bg-cyan-500/15 px-1.5 py-0.5 font-mono text-[10px] text-cyan-300">
              {providerBadge}
            </span>
          )}
        </div>

        <div>
          <label className="mb-1 block text-[11px] font-medium text-slate-400">
            Song Idea / Vibe Prompt
          </label>
          <textarea
            rows={3}
            value={rawPrompt}
            onChange={(e) => setRawPrompt(e.target.value)}
            className="w-full rounded-md border border-studio-border bg-studio-bg p-2 text-xs text-slate-200 placeholder-slate-500 focus:border-indigo-500 focus:outline-none"
            placeholder="Describe genre, mood, instruments, vocal style..."
          />
          <button
            type="button"
            disabled={!connected || enhancing || !rawPrompt.trim()}
            onClick={handleEnhance}
            className="mt-1.5 flex w-full items-center justify-center gap-1.5 rounded-md border border-cyan-500/40 bg-cyan-500/10 py-1.5 text-xs font-medium text-cyan-300 hover:bg-cyan-500/20 disabled:opacity-40 transition"
          >
            <Wand2 className={`h-3.5 w-3.5 ${enhancing ? "animate-spin" : ""}`} />
            <span>{enhancing ? "Structuring via LLM..." : "Enhance Lyrics & Tags (Zero-VRAM)"}</span>
          </button>
        </div>

        <div className="grid grid-cols-2 gap-2.5">
          <div>
            <label className="mb-1 block text-[11px] text-slate-400">BPM</label>
            <input
              type="number"
              min={50}
              max={220}
              step={1}
              value={bpm}
              onChange={(e) => setBpm(Number(e.target.value))}
              className="w-full rounded-md border border-studio-border bg-studio-bg px-2.5 py-1.5 font-mono text-xs text-slate-200 focus:border-indigo-500 focus:outline-none"
            />
          </div>
          <div>
            <label className="mb-1 block text-[11px] text-slate-400">Musical Key</label>
            <input
              type="text"
              value={musicalKey}
              onChange={(e) => setMusicalKey(e.target.value)}
              className="w-full rounded-md border border-studio-border bg-studio-bg px-2.5 py-1.5 font-mono text-xs text-slate-200 focus:border-indigo-500 focus:outline-none"
            />
          </div>
          <div>
            <label className="mb-1 block text-[11px] text-slate-400">Duration (sec)</label>
            <input
              type="number"
              min={8}
              max={180}
              step={2}
              value={durationSec}
              onChange={(e) => setDurationSec(Number(e.target.value))}
              className="w-full rounded-md border border-studio-border bg-studio-bg px-2.5 py-1.5 font-mono text-xs text-slate-200 focus:border-indigo-500 focus:outline-none"
            />
          </div>
          <div>
            <label className="mb-1 block text-[11px] text-slate-400">Seed</label>
            <input
              type="number"
              value={seed}
              onChange={(e) => setSeed(Number(e.target.value))}
              className="w-full rounded-md border border-studio-border bg-studio-bg px-2.5 py-1.5 font-mono text-xs text-slate-200 focus:border-indigo-500 focus:outline-none"
            />
          </div>
        </div>

        <div>
          <label className="mb-1 block text-[11px] text-slate-400">Style Tags</label>
          <input
            type="text"
            value={styleTagsText}
            onChange={(e) => setStyleTagsText(e.target.value)}
            className="w-full rounded-md border border-studio-border bg-studio-bg px-2.5 py-1.5 text-xs text-slate-200 focus:border-indigo-500 focus:outline-none"
          />
        </div>

        <div className="flex-1 flex flex-col">
          <label className="mb-1 block text-[11px] text-slate-400">
            Structured Lyrics ([Intro], [Verse], [Chorus])
          </label>
          <textarea
            rows={7}
            value={structuredLyrics}
            onChange={(e) => setStructuredLyrics(e.target.value)}
            className="w-full flex-1 rounded-md border border-studio-border bg-studio-bg p-2 font-mono text-[11px] leading-relaxed text-slate-200 focus:border-indigo-500 focus:outline-none"
          />
        </div>

        <label className="flex items-center gap-2 text-xs text-slate-300 cursor-pointer">
          <input
            type="checkbox"
            checked={separateStems}
            onChange={(e) => setSeparateStems(e.target.checked)}
            className="rounded border-studio-border bg-studio-bg text-indigo-500"
          />
          <span>Auto-split 4 Stems via Demucs (htdemucs)</span>
        </label>

        <button
          type="submit"
          disabled={!connected || busy}
          className="flex items-center justify-center gap-2 rounded-lg bg-gradient-to-r from-indigo-600 to-cyan-600 py-2.5 text-xs font-bold uppercase tracking-wide text-white shadow-lg shadow-indigo-500/20 hover:from-indigo-500 hover:to-cyan-500 disabled:opacity-40 transition"
        >
          <Music2 className="h-4 w-4" />
          <span>{busy ? "GPU Pipeline Running..." : "Generate Song & Stems"}</span>
        </button>
      </form>
    </aside>
  );
};
