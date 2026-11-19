import * as Tone from "tone";
import { buildTrackAudioUrl } from "../api/colabClient";
import type { ProjectSchema, TrackSchema } from "../types/project";

export interface TrackWaveformData {
  url: string;
  peaks: Float32Array[];
  duration: number;
}

interface LoadedTrackNode {
  trackId: string;
  revision: number;
  url: string;
  player: Tone.Player;
  lowFilter: Tone.BiquadFilter;
  midFilter: Tone.BiquadFilter;
  highFilter: Tone.BiquadFilter;
  compressor: Tone.Compressor;
  reverb: Tone.Reverb;
  lastReverbDecay: number;
  channel: Tone.Channel;
  waveform: TrackWaveformData;
}

function extractWaveformPeaks(
  buffer: Tone.ToneAudioBuffer,
  peaksPerSec = 250
): { peaks: Float32Array[]; duration: number } {
  const duration = Math.max(0.01, buffer.duration || 0);
  const channels = Math.min(2, Math.max(1, buffer.numberOfChannels || 1));
  const targetLen = Math.max(64, Math.ceil(duration * peaksPerSec));
  const peaks: Float32Array[] = [];

  for (let ch = 0; ch < channels; ch++) {
    const raw = buffer.getChannelData(ch);
    const out = new Float32Array(targetLen);
    const total = raw.length;
    if (total > 0) {
      const step = total / targetLen;
      for (let i = 0; i < targetLen; i++) {
        const start = Math.floor(i * step);
        const end = Math.min(total, Math.ceil((i + 1) * step));
        let signedMax = 0;
        let absMax = 0;
        for (let j = start; j < end; j++) {
          const val = raw[j];
          const a = val < 0 ? -val : val;
          if (a > absMax) {
            absMax = a;
            signedMax = val;
          }
        }
        out[i] = Math.max(-1, Math.min(1, signedMax));
      }
    }
    peaks.push(out);
  }
  return { peaks, duration };
}

/**
 * Multi-track Tone.Transport audio engine with strict Promise.all preloading,
 * progressive waveform availability, incremental track diffing, race-safe project
 * switching, WYSIWYG 3-band EQ + Compressor + Reverb, and shared peak extraction.
 */
export class StudioAudioEngine {
  private nodes: Map<string, LoadedTrackNode> = new Map();
  private previewWaveforms: Map<string, TrackWaveformData> = new Map();
  private loadedProjectId: string | null = null;
  private isReady = false;
  private loadRequestId = 0;

  public getTrackWaveform(trackId: string): TrackWaveformData | null {
    return this.previewWaveforms.get(trackId) ?? this.nodes.get(trackId)?.waveform ?? null;
  }

  public async loadProjectTracks(
    baseUrl: string,
    project: ProjectSchema,
    onProgress?: (loaded: number, total: number) => void
  ): Promise<void> {
    const reqId = ++this.loadRequestId;
    const audioTracks = project.tracks.filter((t) => Boolean(t.file_rel_path));
    if (audioTracks.length === 0) {
      this.disposeAll();
      this.loadedProjectId = project.project_id;
      this.isReady = false;
      onProgress?.(0, 0);
      return;
    }

    const totalCount = audioTracks.length;
    const sameProject = this.loadedProjectId === project.project_id;
    const desiredUrls = new Map<string, string>();
    for (const track of audioTracks) {
      desiredUrls.set(
        track.track_id,
        buildTrackAudioUrl(baseUrl, project.project_id, track.file_rel_path!, track.revision)
      );
    }

    // Fast path: all URLs already loaded
    const allMatch =
      sameProject &&
      audioTracks.length === this.nodes.size &&
      audioTracks.every((t) => {
        const existing = this.nodes.get(t.track_id);
        return existing && existing.url === desiredUrls.get(t.track_id);
      });

    if (allMatch) {
      onProgress?.(totalCount, totalCount);
      this.syncMixerParams(audioTracks);
      return;
    }

    if (!sameProject) {
      this.disposeNodesOnly();
      this.previewWaveforms.clear();
      this.isReady = false;
    }

    // Incrementally fetch ONLY new or changed track URLs
    const tracksToFetch = audioTracks.filter((track) => {
      if (!sameProject) return true;
      const existing = this.nodes.get(track.track_id);
      return !existing || existing.url !== desiredUrls.get(track.track_id);
    });

    let loadedCount = totalCount - tracksToFetch.length;
    onProgress?.(loadedCount, totalCount);

    const fetchedBuffers = await Promise.all(
      tracksToFetch.map(async (track) => {
        const url = desiredUrls.get(track.track_id)!;
        const buffer = await new Promise<Tone.ToneAudioBuffer>((resolve, reject) => {
          const buf = new Tone.ToneAudioBuffer(
            url,
            () => resolve(buf),
            (err) => reject(new Error(`Failed loading ${track.name}: ${String(err)}`))
          );
        });
        const { peaks, duration } = extractWaveformPeaks(buffer);
        const waveform: TrackWaveformData = { url, peaks, duration };
        if (reqId === this.loadRequestId) {
          this.previewWaveforms.set(track.track_id, waveform);
          loadedCount += 1;
          onProgress?.(loadedCount, totalCount);
        }
        return { track, url, buffer, waveform };
      })
    );

    // Abort if a newer project load / disposeAll superseded this request while downloading
    if (reqId !== this.loadRequestId) {
      for (const { buffer } of fetchedBuffers) {
        try {
          buffer.dispose();
        } catch {
          // Ignore disposal errors
        }
      }
      return;
    }

    const transport = Tone.getTransport();
    const wasPlaying = transport.state === "started";
    const resumeSec = transport.seconds;
    if (wasPlaying) {
      transport.pause();
    }

    // Dispose removed or replaced track nodes without touching unchanged tracks
    if (sameProject) {
      for (const [trackId, node] of Array.from(this.nodes.entries())) {
        const wantedUrl = desiredUrls.get(trackId);
        if (!wantedUrl || wantedUrl !== node.url) {
          this.disposeSingleNode(node);
          this.nodes.delete(trackId);
        }
      }
    }

    transport.bpm.value = project.grid.bpm || 120;

    for (const { track, url, buffer, waveform } of fetchedBuffers) {
      const channel = new Tone.Channel({
        volume: track.volume_db,
        pan: track.pan,
        mute: track.muted,
        solo: track.soloed,
      }).toDestination();

      const reverbDecay = Math.max(
        0.35,
        0.4 + Math.min(1, Math.max(0, track.dsp_chain.reverb.room_size)) * 3.6
      );
      const reverb = new Tone.Reverb({
        decay: reverbDecay,
        wet: track.dsp_chain.reverb.enabled
          ? Math.min(0.85, Math.max(0, track.dsp_chain.reverb.wet_level))
          : 0,
      }).connect(channel);

      const compEnabled = track.dsp_chain.compressor.enabled;
      const compressor = new Tone.Compressor({
        threshold: compEnabled
          ? Math.max(-100, Math.min(0, track.dsp_chain.compressor.threshold_db))
          : 0,
        ratio: compEnabled ? Math.max(1, Math.min(20, track.dsp_chain.compressor.ratio)) : 1,
        attack: Math.max(0.001, Math.min(1, (track.dsp_chain.compressor.attack_ms || 10) / 1000)),
        release: Math.max(0.01, Math.min(1, (track.dsp_chain.compressor.release_ms || 100) / 1000)),
      }).connect(reverb);

      const highFilter = new Tone.BiquadFilter({
        type: "highshelf",
        frequency: 4500,
        gain: track.dsp_chain.eq.high_shelf_db,
      }).connect(compressor);

      const midFilter = new Tone.BiquadFilter({
        type: "peaking",
        frequency: track.dsp_chain.eq.mid_freq_hz || 1000,
        Q: 1.2,
        gain: track.dsp_chain.eq.mid_gain_db,
      }).connect(highFilter);

      const lowFilter = new Tone.BiquadFilter({
        type: "lowshelf",
        frequency: 140,
        gain: track.dsp_chain.eq.low_shelf_db,
      }).connect(midFilter);

      const player = new Tone.Player({
        url: buffer,
        fadeIn: 0.003,
        fadeOut: 0.003,
      }).connect(lowFilter);

      player.sync().start(0);

      this.nodes.set(track.track_id, {
        trackId: track.track_id,
        revision: track.revision,
        url,
        player,
        lowFilter,
        midFilter,
        highFilter,
        compressor,
        reverb,
        lastReverbDecay: reverbDecay,
        channel,
        waveform,
      });
    }

    this.previewWaveforms.clear();
    this.syncMixerParams(audioTracks);
    this.loadedProjectId = project.project_id;
    this.isReady = true;

    if (wasPlaying) {
      transport.seconds = Math.min(resumeSec, project.grid.duration_sec || resumeSec);
      transport.start();
    }
  }

  public syncMixerParams(tracks: TrackSchema[]): void {
    const anySolo = tracks.some((t) => Boolean(t.soloed));
    for (const track of tracks) {
      const node = this.nodes.get(track.track_id);
      if (!node) continue;
      const effectivelyMuted = Boolean(track.muted) || (anySolo && !track.soloed);
      node.player.mute = effectivelyMuted;
      node.channel.mute = effectivelyMuted;
      node.channel.solo = Boolean(track.soloed);
      node.channel.volume.value = effectivelyMuted ? -Infinity : track.volume_db;
      node.channel.pan.value = track.pan;
      node.lowFilter.gain.value = track.dsp_chain.eq.low_shelf_db;
      node.midFilter.frequency.value = track.dsp_chain.eq.mid_freq_hz || 1000;
      node.midFilter.gain.value = track.dsp_chain.eq.mid_gain_db;
      node.highFilter.gain.value = track.dsp_chain.eq.high_shelf_db;

      const compEnabled = Boolean(track.dsp_chain.compressor.enabled);
      node.compressor.threshold.value = compEnabled
        ? Math.max(-100, Math.min(0, track.dsp_chain.compressor.threshold_db))
        : 0;
      node.compressor.ratio.value = compEnabled
        ? Math.max(1, Math.min(20, track.dsp_chain.compressor.ratio))
        : 1;
      node.compressor.attack.value = Math.max(
        0.001,
        Math.min(1, (track.dsp_chain.compressor.attack_ms || 10) / 1000)
      );
      node.compressor.release.value = Math.max(
        0.01,
        Math.min(1, (track.dsp_chain.compressor.release_ms || 100) / 1000)
      );

      const targetDecay = Math.max(
        0.35,
        0.4 + Math.min(1, Math.max(0, track.dsp_chain.reverb.room_size)) * 3.6
      );
      if (Math.abs(targetDecay - node.lastReverbDecay) > 0.05) {
        node.reverb.decay = targetDecay;
        node.lastReverbDecay = targetDecay;
      }
      node.reverb.wet.value = track.dsp_chain.reverb.enabled
        ? Math.min(0.85, Math.max(0, track.dsp_chain.reverb.wet_level))
        : 0;
    }
  }

  public async play(loopRegion?: { startSec: number; endSec: number } | null): Promise<boolean> {
    if (!this.isReady) return false;
    await Tone.start();
    const transport = Tone.getTransport();
    if (loopRegion && loopRegion.endSec > loopRegion.startSec) {
      transport.loop = true;
      transport.loopStart = loopRegion.startSec;
      transport.loopEnd = loopRegion.endSec;
      if (transport.seconds < loopRegion.startSec || transport.seconds >= loopRegion.endSec) {
        transport.seconds = loopRegion.startSec;
      }
    } else {
      transport.loop = false;
    }
    transport.start();
    return true;
  }

  public pause(): void {
    Tone.getTransport().pause();
  }

  public stop(): void {
    const transport = Tone.getTransport();
    transport.stop();
    transport.seconds = 0;
  }

  public seekToSec(sec: number): void {
    Tone.getTransport().seconds = Math.max(0, sec);
  }

  public setLoop(enabled: boolean, startSec = 0, endSec = 0): void {
    const transport = Tone.getTransport();
    if (enabled && endSec > startSec) {
      transport.loop = true;
      transport.loopStart = startSec;
      transport.loopEnd = endSec;
    } else {
      transport.loop = false;
    }
  }

  public getCurrentTimeSec(): number {
    return Tone.getTransport().seconds;
  }

  public getTransportState(): "started" | "paused" | "stopped" {
    return Tone.getTransport().state;
  }

  private disposeSingleNode(node: LoadedTrackNode): void {
    try {
      node.player.unsync();
      node.player.dispose();
      node.lowFilter.dispose();
      node.midFilter.dispose();
      node.highFilter.dispose();
      node.compressor.dispose();
      node.reverb.dispose();
      node.channel.dispose();
    } catch {
      // Ignore cleanup errors on disposed Web Audio nodes
    }
  }

  private disposeNodesOnly(): void {
    for (const node of this.nodes.values()) {
      this.disposeSingleNode(node);
    }
    this.nodes.clear();
  }

  public disposeAll(): void {
    this.loadRequestId++;
    this.previewWaveforms.clear();
    const transport = Tone.getTransport();
    if (transport.state !== "stopped") {
      transport.stop();
    }
    this.disposeNodesOnly();
    this.loadedProjectId = null;
    this.isReady = false;
  }
}

export const audioEngine = new StudioAudioEngine();
