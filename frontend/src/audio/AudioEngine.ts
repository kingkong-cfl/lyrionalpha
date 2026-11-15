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
 * incremental track diffing, race-safe project switching, WYSIWYG 3-band Biquad EQ,
 * and shared decoded peak extraction so WaveSurfer never double-fetches WAVs.
 */
export class StudioAudioEngine {
  private nodes: Map<string, LoadedTrackNode> = new Map();
  private loadedProjectId: string | null = null;
  private isReady = false;
  private loadRequestId = 0;

  public getTrackWaveform(trackId: string): TrackWaveformData | null {
    return this.nodes.get(trackId)?.waveform ?? null;
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
        if (reqId === this.loadRequestId) {
          loadedCount += 1;
          onProgress?.(loadedCount, totalCount);
        }
        return { track, url, buffer };
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
    if (!sameProject) {
      this.disposeNodesOnly();
    } else {
      for (const [trackId, node] of Array.from(this.nodes.entries())) {
        const wantedUrl = desiredUrls.get(trackId);
        if (!wantedUrl || wantedUrl !== node.url) {
          this.disposeSingleNode(node);
          this.nodes.delete(trackId);
        }
      }
    }

    transport.bpm.value = project.grid.bpm || 120;

    for (const { track, url, buffer } of fetchedBuffers) {
      const channel = new Tone.Channel({
        volume: track.volume_db,
        pan: track.pan,
        mute: track.muted,
        solo: track.soloed,
      }).toDestination();

      const highFilter = new Tone.BiquadFilter({
        type: "highshelf",
        frequency: 4500,
        gain: track.dsp_chain.eq.high_shelf_db,
      }).connect(channel);

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
      const { peaks, duration } = extractWaveformPeaks(buffer);
      const waveform: TrackWaveformData = { url, peaks, duration };

      this.nodes.set(track.track_id, {
        trackId: track.track_id,
        revision: track.revision,
        url,
        player,
        lowFilter,
        midFilter,
        highFilter,
        channel,
        waveform,
      });
    }

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
    }
  }

  public async play(loopRegion?: { startSec: number; endSec: number } | null): Promise<void> {
    if (!this.isReady) return;
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
