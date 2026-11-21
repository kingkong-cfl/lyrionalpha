import React, { useEffect, useState } from "react";
import { FileText, FolderPlus, Music2, Sparkles, Tags, Wand2 } from "lucide-react";
import type { EnhanceMode, ProducerEnhanceResponse, ProjectSchema } from "../../types/project";

interface ProducerPanelProps {
  projects: ProjectSchema[];
  activeProject: ProjectSchema | null;
  onSelectProject: (projectId: string) => void;
  onCreateProject: (name: string) => Promise<void>;
  onEnhancePrompt: (
    rawPrompt: string,
    options?: {
      preferredDurationSec?: number | null;
      mode?: EnhanceMode;
      existingStyleTags?: string[];
      existingLyrics?: string;
    }
  ) => Promise<ProducerEnhanceResponse>;
  onGenerateSong: (params: {
    raw_prompt: string;
    style_tags: string[];
    structured_lyrics: string;
    vocal_language?: string | null;
    bpm: number | null;
    musical_key: string | null;
    duration_sec: number | null;
    seed: number | null;
    separate_stems: boolean;
  }) => Promise<void>;
  busy: boolean;
  connected: boolean;
}

function sanitizeDecimalDraft(raw: string): string {
  const cleaned = raw.replace(/[^\d.]/g, "");
  const dotIdx = cleaned.indexOf(".");
  const singleDot =
    dotIdx === -1
      ? cleaned
      : cleaned.slice(0, dotIdx + 1) + cleaned.slice(dotIdx + 1).replace(/\./g, "");
  return singleDot.replace(/^0+(?=\d)/, "");
}

function sanitizeIntegerDraft(raw: string): string {
  return raw.replace(/\D/g, "").replace(/^0+(?=\d)/, "");
}

function parseOptionalNumber(
  raw: string,
  min: number,
  max: number,
  integer = false
): number | null {
  const trimmed = raw.trim();
  if (!trimmed) return null;
  const parsed = integer ? parseInt(trimmed, 10) : parseFloat(trimmed);
  if (!Number.isFinite(parsed)) return null;
  const clamped = Math.max(min, Math.min(max, parsed));
  return integer ? Math.round(clamped) : Math.round(clamped * 10) / 10;
}

function countLyricWords(lyrics: string): number {
  const withoutTags = lyrics.replace(/\[[^\]]+\]/g, " ").trim();
  if (!withoutTags) return 0;
  return withoutTags.split(/\s+/).filter(Boolean).length;
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
    "[Verse 1]\nStreetlights blur across the midnight glass\nCounting down the miles as shadows pass\n\n[Chorus]\nOut on the neon horizon we run\nChasing the pulse till the rising sun"
  );
  const [bpmInput, setBpmInput] = useState("");
  const [musicalKeyInput, setMusicalKeyInput] = useState("");
  const [durationInput, setDurationInput] = useState("");
  const [seedInput, setSeedInput] = useState("");
  const [vocalLanguage, setVocalLanguage] = useState("");
  const [separateStems, setSeparateStems] = useState(true);
  const [providerBadge, setProviderBadge] = useState<string | null>(null);
  const [enhancingMode, setEnhancingMode] = useState<EnhanceMode | null>(null);

  useEffect(() => {
    if (!activeProject) return;
    if (activeProject.tracks.length > 0) {
      setBpmInput(
        activeProject.grid.bpm > 0 ? String(Math.round(activeProject.grid.bpm * 10) / 10) : ""
      );
      const rawKey = (activeProject.grid.musical_key || "").trim();
      setMusicalKeyInput(
        rawKey && rawKey.toLowerCase() !== "auto" && rawKey.toLowerCase() !== "unknown"
          ? rawKey
          : ""
      );
      setDurationInput(
        activeProject.grid.duration_sec > 0 ? String(Math.round(activeProject.grid.duration_sec)) : ""
      );
    } else {
      setBpmInput("");
      setMusicalKeyInput("");
      setDurationInput("");
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
    setSeedInput(
      activeProject.producer_spec.seed !== null && activeProject.producer_spec.seed !== undefined
        ? String(activeProject.producer_spec.seed)
        : ""
    );
    setVocalLanguage(activeProject.producer_spec.vocal_language || "");
  }, [
    activeProject?.project_id,
    activeProject?.tracks.length,
    activeProject?.grid.bpm,
    activeProject?.grid.duration_sec,
    activeProject?.grid.musical_key,
    activeProject?.producer_spec.seed,
    activeProject?.producer_spec.vocal_language,
  ]);

  const currentTagsList = () =>
    styleTagsText
      .split(",")
      .map((s) => s.trim())
      .filter(Boolean);

  const handleEnhance = async (mode: EnhanceMode) => {
    if (!rawPrompt.trim()) return;
    setEnhancingMode(mode);
    try {
      const res = await onEnhancePrompt(rawPrompt, {
        preferredDurationSec: parseOptionalNumber(durationInput, 8, 240),
        mode,
        existingStyleTags: currentTagsList(),
        existingLyrics: structuredLyrics,
      });
      if (mode === "tags") {
        setStyleTagsText(res.style_tags.join(", "));
      } else if (mode === "lyrics") {
        setStructuredLyrics(res.structured_lyrics);
      } else {
        setBpmInput(String(Math.round(res.bpm * 10) / 10));
        setMusicalKeyInput(res.musical_key);
        setDurationInput(String(Math.round(res.duration_sec)));
        setStyleTagsText(res.style_tags.join(", "));
        setStructuredLyrics(res.structured_lyrics);
      }
      setProviderBadge(res.provider);
    } finally {
      setEnhancingMode(null);
    }
  };

  const handleGenerate = async (e: React.FormEvent) => {
    e.preventDefault();
    const bpmVal = parseOptionalNumber(bpmInput, 21, 300);
    const durationVal = parseOptionalNumber(durationInput, 8, 240);
    const seedVal = parseOptionalNumber(seedInput, 0, 2_147_483_647, true);
    const cleanKey = musicalKeyInput.trim();
    await onGenerateSong({
      raw_prompt: rawPrompt,
      style_tags: currentTagsList(),
      structured_lyrics: structuredLyrics,
      vocal_language: vocalLanguage || null,
      bpm: bpmVal,
      musical_key: cleanKey || null,
      duration_sec: durationVal,
      seed: seedVal,
      separate_stems: separateStems,
    });
  };

  const lyricWordCount = countLyricWords(structuredLyrics);

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
          <div className="mt-1.5 flex items-center gap-1.5">
            <button
              type="button"
              disabled={!connected || enhancingMode !== null || !rawPrompt.trim()}
              onClick={() => handleEnhance("tags")}
              title="Generate or refine only Style Tags (keeps Lyrics, BPM, Key & Duration untouched)"
              className="flex flex-1 items-center justify-center gap-1 rounded-md border border-cyan-500/40 bg-cyan-500/10 py-1.5 text-[11px] font-medium text-cyan-300 hover:bg-cyan-500/20 disabled:opacity-40 transition"
            >
              <Tags className={`h-3 w-3 ${enhancingMode === "tags" ? "animate-spin" : ""}`} />
              <span>{enhancingMode === "tags" ? "Tags..." : "Enhance Tags"}</span>
            </button>
            <button
              type="button"
              disabled={!connected || enhancingMode !== null || !rawPrompt.trim()}
              onClick={() => handleEnhance("lyrics")}
              title="Write or structure only Lyrics (keeps Style Tags, BPM, Key & Duration untouched)"
              className="flex flex-1 items-center justify-center gap-1 rounded-md border border-indigo-500/40 bg-indigo-500/10 py-1.5 text-[11px] font-medium text-indigo-300 hover:bg-indigo-500/20 disabled:opacity-40 transition"
            >
              <FileText className={`h-3 w-3 ${enhancingMode === "lyrics" ? "animate-spin" : ""}`} />
              <span>{enhancingMode === "lyrics" ? "Lyrics..." : "Enhance Lyrics"}</span>
            </button>
            <button
              type="button"
              disabled={!connected || enhancingMode !== null || !rawPrompt.trim()}
              onClick={() => handleEnhance("all")}
              title="Auto-fill Tags, Lyrics, BPM, Key & Duration together"
              className="flex items-center justify-center gap-1 rounded-md border border-studio-border bg-studio-bg px-2.5 py-1.5 text-[11px] font-medium text-slate-300 hover:border-cyan-500/40 hover:text-cyan-200 disabled:opacity-40 transition"
            >
              <Wand2 className={`h-3 w-3 ${enhancingMode === "all" ? "animate-spin" : ""}`} />
              <span>All</span>
            </button>
          </div>
        </div>

        <div className="grid grid-cols-2 gap-2.5">
          <div>
            <label className="mb-1 block text-[11px] text-slate-400">BPM (Optional)</label>
            <input
              type="text"
              inputMode="decimal"
              value={bpmInput}
              placeholder="Auto"
              onChange={(e) => setBpmInput(sanitizeDecimalDraft(e.target.value))}
              onBlur={() => {
                const parsed = parseOptionalNumber(bpmInput, 21, 300);
                setBpmInput(parsed !== null ? String(parsed) : "");
              }}
              className="w-full rounded-md border border-studio-border bg-studio-bg px-2.5 py-1.5 font-mono text-xs text-slate-200 placeholder-slate-500 focus:border-indigo-500 focus:outline-none"
            />
          </div>
          <div>
            <label className="mb-1 block text-[11px] text-slate-400">Musical Key (Optional)</label>
            <input
              type="text"
              value={musicalKeyInput}
              placeholder="Auto"
              onChange={(e) => setMusicalKeyInput(e.target.value)}
              onBlur={() => setMusicalKeyInput((v) => v.trim())}
              className="w-full rounded-md border border-studio-border bg-studio-bg px-2.5 py-1.5 font-mono text-xs text-slate-200 placeholder-slate-500 focus:border-indigo-500 focus:outline-none"
            />
          </div>
          <div>
            <label className="mb-1 block text-[11px] text-slate-400">Duration (sec)</label>
            <input
              type="text"
              inputMode="decimal"
              value={durationInput}
              placeholder="Auto"
              onChange={(e) => setDurationInput(sanitizeDecimalDraft(e.target.value))}
              onBlur={() => {
                const parsed = parseOptionalNumber(durationInput, 8, 240);
                setDurationInput(parsed !== null ? String(parsed) : "");
              }}
              className="w-full rounded-md border border-studio-border bg-studio-bg px-2.5 py-1.5 font-mono text-xs text-slate-200 placeholder-slate-500 focus:border-indigo-500 focus:outline-none"
            />
          </div>
          <div>
            <label className="mb-1 block text-[11px] text-slate-400">Seed (Optional)</label>
            <input
              type="text"
              inputMode="numeric"
              value={seedInput}
              placeholder="Auto"
              onChange={(e) => setSeedInput(sanitizeIntegerDraft(e.target.value))}
              onBlur={() => {
                const parsed = parseOptionalNumber(seedInput, 0, 2_147_483_647, true);
                setSeedInput(parsed !== null ? String(parsed) : "");
              }}
              className="w-full rounded-md border border-studio-border bg-studio-bg px-2.5 py-1.5 font-mono text-xs text-slate-200 placeholder-slate-500 focus:border-indigo-500 focus:outline-none"
            />
          </div>
        </div>

        <div className="flex items-center justify-between gap-2">
          <label className="text-[11px] text-slate-400">Vocal Language</label>
          <select
            value={vocalLanguage}
            onChange={(e) => setVocalLanguage(e.target.value)}
            className="rounded-md border border-studio-border bg-studio-bg px-2.5 py-1 font-mono text-xs text-slate-200 focus:border-indigo-500 focus:outline-none"
          >
            <option value="">Auto (TR algıla)</option>
            <option value="tr">Turkish (tr)</option>
            <option value="en">English (en)</option>
            <option value="instrumental">Instrumental / Other</option>
          </select>
        </div>

        <div>
          <label className="mb-1 block text-[11px] text-slate-400">Style Tags</label>
          <input
            type="text"
            value={styleTagsText}
            onChange={(e) => setStyleTagsText(e.target.value)}
            placeholder="Leave blank to use Song Idea prompt directly..."
            className="w-full rounded-md border border-studio-border bg-studio-bg px-2.5 py-1.5 text-xs text-slate-200 placeholder-slate-500 focus:border-indigo-500 focus:outline-none"
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
          <div
            className={`mt-1 text-[10px] leading-snug ${
              lyricWordCount >= 140 && !durationInput.trim()
                ? "text-amber-300"
                : "text-slate-500"
            }`}
          >
            ~{lyricWordCount} kelime — uzun sözlerde Auto 120s keserse Duration kutusuna 180–210 yazabilirsin.
          </div>
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
