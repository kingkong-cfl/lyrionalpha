"""Synchronous pipeline verification tests for Lyrion Colab backend (MOCK_GPU=true)."""

from __future__ import annotations

import os
import shutil
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np

# Isolate test Drive storage in a temporary folder before importing app
TEST_TMP_DIR = Path(tempfile.mkdtemp(prefix="lyrion_test_"))
os.environ["MOCK_GPU"] = "true"
os.environ["DRIVE_ROOT"] = str(TEST_TMP_DIR)

from app.config import settings  # noqa: E402
from app.core.model_manager import ModelType, model_manager  # noqa: E402
from app.core.storage import storage  # noqa: E402
from app.main import app, is_request_authorized  # noqa: E402
from app.routers.generation import (  # noqa: E402
    _enforce_reference_mix_muted,
    _resolve_job_seed,
    _unique_wav_rel,
    sum_raw_stems,
)
from app.routers.projects import merge_track_mixer_updates  # noqa: E402
from app.schemas import ProducerSpecSchema, ProjectSchema, TrackSchema  # noqa: E402
from app.services.acestep_service import (  # noqa: E402
    _check_acestep_init_result,
    _invoke_acestep_real_sync,
    _resolve_acestep_project_root,
    _synthesize_musical_mix,
)
from app.services.beat_service import analyze_beat_grid  # noqa: E402
from app.services.demucs_service import _mock_separate_stems  # noqa: E402
from app.services.dsp_service import export_project_audio, process_track_dsp  # noqa: E402
from app.services.llm_producer import _deterministic_fallback  # noqa: E402
from app.services.splice_service import (  # noqa: E402
    apply_region_fade_edges,
    compute_context_window,
    equal_power_crossfade_splice,
)


class BackendPipelineTest(unittest.TestCase):
    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(TEST_TMP_DIR, ignore_errors=True)

    def test_01_routes_config_and_producer(self) -> None:
        self.assertTrue(settings.mock_gpu)
        self.assertTrue(storage.is_writable())

        route_paths = set(app.openapi()["paths"].keys())
        for expected in (
            "/api/v1/health",
            "/api/v1/producer/enhance",
            "/api/v1/projects",
            "/api/v1/projects/{project_id}/generate",
            "/api/v1/projects/{project_id}/repaint",
            "/api/v1/projects/{project_id}/repaint-track",
            "/api/v1/projects/{project_id}/add-instrument",
            "/api/v1/projects/{project_id}/lego",
            "/api/v1/projects/{project_id}/undo-repaint",
            "/api/v1/projects/{project_id}/export",
            "/api/v1/jobs/{job_id}",
        ):
            self.assertIn(expected, route_paths)

        spec = _deterministic_fallback(
            "driving synthwave track 124 bpm at night",
            preferred_duration=12.0,
        )
        self.assertEqual(spec.bpm, 124.0)
        self.assertEqual(spec.duration_sec, 12.0)
        self.assertIn("[Chorus]", spec.structured_lyrics)
        self.assertGreaterEqual(len(spec.style_tags), 3)

    def test_02_model_manager_telemetry_and_unload(self) -> None:
        telemetry = model_manager.get_telemetry()
        self.assertTrue(telemetry.mock_gpu)
        self.assertEqual(telemetry.vram_total_mb, 24576.0)
        self.assertEqual(telemetry.active_model, "none")

        # Verify explicit _unload_current releases model & increments unload_count
        prev_unloads = model_manager.unload_count
        model_manager._active_type = ModelType.ACE_STEP
        model_manager._active_model = {"dummy": True}
        model_manager._unload_current()
        self.assertEqual(model_manager.active_type, ModelType.NONE)
        self.assertIsNone(model_manager._active_model)
        self.assertEqual(model_manager.unload_count, prev_unloads + 1)

    def test_03_generate_stems_splice_and_export_pipeline(self) -> None:
        # 1. Create & persist project atomically
        project = storage.create_project("Test Synth Track", bpm=120.0, musical_key="A minor")
        project_id = project.project_id

        # 2. Generate 8-second stereo mix & extract CPU beat grid
        sr = settings.sample_rate
        duration_sec = 8.0
        total_samples = int(duration_sec * sr)
        master_mix = _synthesize_musical_mix(
            total_samples=total_samples,
            sample_rate=sr,
            bpm=120.0,
            musical_key="A minor",
            seed=12345,
            style_tags=["synthwave", "analog bass"],
        )
        self.assertEqual(master_mix.shape, (total_samples, 2))
        storage.write_wav(project_id, "raw/master_mix_rev1.wav", master_mix, sr)

        grid = analyze_beat_grid(master_mix, sr, target_bpm=120.0, musical_key="A minor", time_signature="4/4")
        self.assertEqual(grid.sample_rate, 44100)
        self.assertEqual(grid.total_samples, total_samples)
        self.assertEqual(len(grid.beat_samples), 16)
        self.assertEqual(grid.downbeat_beat_indices, [0, 4, 8, 12])

        # 3. Separate into 4 stems and verify accurate mix reconstruction
        stems = _mock_separate_stems(master_mix, sr)
        self.assertEqual(set(stems.keys()), {"vocals", "drums", "bass", "other"})
        reconstructed = stems["vocals"] + stems["drums"] + stems["bass"] + stems["other"]
        max_diff = float(np.max(np.abs(reconstructed - master_mix)))
        self.assertLess(max_diff, 1e-5)

        tracks: list[TrackSchema] = [
            TrackSchema(
                track_id="trk_master_01",
                name="Full Mix (Reference)",
                track_type="audio",
                stem_role="full_mix",
                file_rel_path="raw/master_mix_rev1.wav",
                revision=1,
                sample_rate=sr,
                start_sample=0,
                end_sample=total_samples,
                muted=True,
            )
        ]
        for role in ("vocals", "drums", "bass", "other"):
            rel = f"stems/{role}_rev1.wav"
            storage.write_wav(project_id, rel, stems[role], sr)
            tracks.append(
                TrackSchema(
                    track_id=f"trk_{role}_01",
                    name=role.title(),
                    track_type="stem",
                    stem_role=role,  # type: ignore[arg-type]
                    parent_track_id="trk_master_01",
                    file_rel_path=rel,
                    revision=1,
                    sample_rate=sr,
                    start_sample=0,
                    end_sample=total_samples,
                    muted=False,
                )
            )

        project.grid = grid
        project.producer_spec = ProducerSpecSchema(raw_prompt="synthwave night drive", seed=12345)
        project.tracks = tracks
        storage.save_project(project)

        reloaded = storage.load_project(project_id)
        self.assertIsInstance(reloaded, ProjectSchema)
        self.assertEqual(len(reloaded.tracks), 5)

        # 4. Sample-accurate Repaint on Bars 2..3 (samples 88200..176400) + Windowed Demucs Re-Sync
        start_sample = 88200
        end_sample = 176400
        rev1_master, _ = storage.read_wav(project_id, "raw/master_mix_rev1.wav", sr)
        rev1_vocals, _ = storage.read_wav(project_id, "stems/vocals_rev1.wav", sr)

        fresh_mix = _synthesize_musical_mix(
            total_samples=total_samples,
            sample_rate=sr,
            bpm=120.0,
            musical_key="A minor",
            seed=99999,
            style_tags=["soaring chorus brass"],
        )
        spliced_master = equal_power_crossfade_splice(
            rev1_master,
            fresh_mix[start_sample:end_sample],
            start_sample,
            end_sample,
            settings.crossfade_samples,
        )
        storage.write_wav(project_id, "raw/master_mix_rev2.wav", spliced_master, sr)
        rev2_master, _ = storage.read_wav(project_id, "raw/master_mix_rev2.wav", sr)

        win_start, win_end, inner_start, inner_end = compute_context_window(
            start_sample,
            end_sample,
            total_samples,
            settings.demucs_context_samples,
        )
        win_stems = _mock_separate_stems(rev2_master[win_start:win_end], sr)
        spliced_vocals = equal_power_crossfade_splice(
            rev1_vocals,
            win_stems["vocals"][inner_start:inner_end],
            start_sample,
            end_sample,
            settings.crossfade_samples,
        )
        storage.write_wav(project_id, "stems/vocals_rev2.wav", spliced_vocals, sr)
        rev2_vocals, _ = storage.read_wav(project_id, "stems/vocals_rev2.wav", sr)

        # Verify exact sample alignment and bit-identity outside [start_sample, end_sample)
        self.assertEqual(rev2_master.shape, rev1_master.shape)
        self.assertEqual(rev2_vocals.shape, rev1_vocals.shape)
        self.assertTrue(np.array_equal(rev1_master[:start_sample], rev2_master[:start_sample]))
        self.assertTrue(np.array_equal(rev1_master[end_sample:], rev2_master[end_sample:]))
        self.assertTrue(np.array_equal(rev1_vocals[:start_sample], rev2_vocals[:start_sample]))
        self.assertTrue(np.array_equal(rev1_vocals[end_sample:], rev2_vocals[end_sample:]))
        self.assertFalse(
            np.array_equal(
                rev1_master[start_sample + 4000 : end_sample - 4000],
                rev2_master[start_sample + 4000 : end_sample - 4000],
            )
        )

        # 5. Unique cache-busted WAV filenames & cached Drive writability check
        from pathlib import Path

        rel_a = _unique_wav_rel("raw", "master_mix", 2)
        rel_b = _unique_wav_rel("raw", "master_mix", 2)
        self.assertNotEqual(rel_a, rel_b)
        self.assertTrue(rel_a.startswith("raw/master_mix_rev2_") and rel_a.endswith(".wav"))
        self.assertTrue(storage.is_writable())
        self.assertFalse((Path(settings.drive_root) / ".write_probe").exists())

        # 6. DSP Export: Master WAV & Stems ZIP
        wav_res = export_project_audio(reloaded, export_format="wav", normalize_lufs=True)
        wav_path = storage.resolve_project_file(project_id, wav_res["download_rel_path"])
        self.assertTrue(wav_path.is_file())
        self.assertGreater(wav_path.stat().st_size, 10000)

        zip_res = export_project_audio(reloaded, export_format="stems_zip", normalize_lufs=True)
        zip_path = storage.resolve_project_file(project_id, zip_res["download_rel_path"])
        self.assertTrue(zip_path.is_file())
        with zipfile.ZipFile(zip_path, "r") as zf:
            names = set(zf.namelist())
            self.assertEqual(names, {"master_mix.wav", "vocals.wav", "drums.wav", "bass.wav", "other.wav"})

        # 7. New Instrument Layer (silent outside [start_sample, end_sample], 50ms fade edges)
        layer_start = int(2.0 * sr)
        layer_end = int(5.0 * sr)
        raw_guitar = _synthesize_musical_mix(
            total_samples=total_samples,
            sample_rate=sr,
            bpm=120.0,
            musical_key="A minor",
            seed=77777,
            style_tags=["electric guitar"],
            layer_mode="custom",
            instrument_hint="melodic electric guitar riff",
        )
        faded_guitar = apply_region_fade_edges(raw_guitar[layer_start:layer_end], settings.crossfade_samples)
        guitar_full = np.zeros((total_samples, 2), dtype=np.float32)
        guitar_full[layer_start:layer_end] = faded_guitar
        storage.write_wav(project_id, "stems/layer_electric_guitar_rev1.wav", guitar_full, sr)
        guitar_disk, _ = storage.read_wav(project_id, "stems/layer_electric_guitar_rev1.wav", sr)

        # Digital silence strictly outside region
        self.assertTrue(np.all(guitar_disk[:layer_start] == 0.0))
        self.assertTrue(np.all(guitar_disk[layer_end:] == 0.0))
        self.assertGreater(float(np.max(np.abs(guitar_disk[layer_start:layer_end]))), 0.01)

        reloaded.tracks.append(
            TrackSchema(
                track_id="trk_inst_guitar",
                name="Electric Guitar",
                track_type="stem",
                stem_role="custom",
                parent_track_id="trk_master_01",
                file_rel_path="stems/layer_electric_guitar_rev1.wav",
                revision=1,
                sample_rate=sr,
                start_sample=0,
                end_sample=total_samples,
            )
        )
        storage.save_project(reloaded)

        # 8. Single-Track WAV Export + Region-Specific Export
        single_exp = export_project_audio(
            reloaded,
            export_format="single_track_wav",
            track_id="trk_inst_guitar",
            start_sample=layer_start,
            end_sample=layer_end,
            normalize_lufs=True,
        )
        single_buf, _ = storage.read_wav(project_id, single_exp["download_rel_path"], sr)
        self.assertEqual(single_buf.shape[0], layer_end - layer_start)

        region_zip_res = export_project_audio(
            reloaded,
            export_format="stems_zip",
            start_sample=layer_start,
            end_sample=layer_end,
            normalize_lufs=True,
        )
        region_zip_path = storage.resolve_project_file(project_id, region_zip_res["download_rel_path"])
        with zipfile.ZipFile(region_zip_path, "r") as zf:
            znames = set(zf.namelist())
            self.assertIn("electric_guitar.wav", znames)
            self.assertIn("master_mix.wav", znames)

    def test_04_random_repaint_seeds_and_reference_mix_mute_enforcement(self) -> None:
        sr = settings.sample_rate
        total_samples = int(4.0 * sr)

        # 1. Explicit None resolves to varying random seeds
        seeds = {_resolve_job_seed(None) for _ in range(10)}
        self.assertGreater(len(seeds), 1)
        self.assertEqual(_resolve_job_seed(421337), 421337)

        # 2. Every stem role varies distinctly between different seeds
        s1 = _resolve_job_seed(None)
        s2 = _resolve_job_seed(None)
        while s2 == s1:
            s2 = _resolve_job_seed(None)

        for role in ("vocals", "drums", "bass", "other", "custom"):
            buf_a = _synthesize_musical_mix(
                total_samples=total_samples,
                sample_rate=sr,
                bpm=120.0,
                musical_key="A minor",
                seed=s1,
                style_tags=["synthwave"],
                layer_mode=role,
                instrument_hint="electric guitar" if role == "custom" else "",
            )
            buf_b = _synthesize_musical_mix(
                total_samples=total_samples,
                sample_rate=sr,
                bpm=120.0,
                musical_key="A minor",
                seed=s2,
                style_tags=["synthwave"],
                layer_mode=role,
                instrument_hint="electric guitar" if role == "custom" else "",
            )
            self.assertFalse(
                np.allclose(buf_a, buf_b),
                f"Role '{role}' produced identical audio for distinct seeds {s1} and {s2}",
            )
            max_delta = float(np.max(np.abs(buf_a - buf_b)))
            self.assertGreater(max_delta, 0.02, f"Role '{role}' delta too small between distinct seeds")

        # 3. Reference mix mute enforcement when stems exist
        proj = storage.create_project("Mute Check", bpm=120.0, musical_key="A minor")
        proj.tracks = [
            TrackSchema(
                track_id="trk_master_01",
                name="Full Mix (Reference)",
                track_type="audio",
                stem_role="full_mix",
                sample_rate=sr,
                end_sample=total_samples,
                muted=False,
            ),
            TrackSchema(
                track_id="trk_drums_01",
                name="Drums",
                track_type="stem",
                stem_role="drums",
                parent_track_id="trk_master_01",
                sample_rate=sr,
                end_sample=total_samples,
                muted=False,
            ),
        ]
        _enforce_reference_mix_muted(proj)
        master_track = next(t for t in proj.tracks if t.stem_role == "full_mix")
        drums_track = next(t for t in proj.tracks if t.stem_role == "drums")
        self.assertTrue(master_track.muted)
        self.assertFalse(drums_track.muted)

    def test_05_phase1_to_3_integrity_dsp_and_concurrency(self):
        import numpy as np

        sr = 44100
        # 1. Equal-gain crossfade splice preserves unity sum on correlated stereo signals
        t = np.linspace(0.0, 1.0, sr, endpoint=False, dtype=np.float32)
        sine = (0.4 * np.sin(2.0 * np.pi * 220.0 * t)).astype(np.float32)
        stereo_sine = np.stack([sine, sine], axis=1)
        spliced = equal_power_crossfade_splice(
            original_audio=stereo_sine.copy(),
            new_segment=stereo_sine[11025:33075].copy(),
            start_sample=11025,
            end_sample=33075,
            crossfade_samples=2205,
            mode="equal_gain",
        )
        self.assertLess(float(np.max(np.abs(spliced - stereo_sine))), 1e-4)

        # 2. sum_raw_stems sums stem WAVs at unity gain, ignoring muted/volume_db/full_mix
        proj = storage.create_project("Phase 1-3 Suite", bpm=120.0, musical_key="C major")
        proj.grid.sample_rate = sr
        proj.grid.total_samples = sr
        proj.grid.duration_sec = 1.0
        drums_wave = np.stack(
            [0.15 * np.sin(2.0 * np.pi * 110.0 * t), 0.15 * np.sin(2.0 * np.pi * 110.0 * t)],
            axis=1,
        ).astype(np.float32)
        bass_wave = np.stack(
            [0.50 * np.sin(2.0 * np.pi * 220.0 * t), 0.50 * np.sin(2.0 * np.pi * 220.0 * t)],
            axis=1,
        ).astype(np.float32)
        ref_wave = np.full((sr, 2), 0.8, dtype=np.float32)

        drums_rel = _unique_wav_rel("stems", "drums", 1)
        bass_rel = _unique_wav_rel("stems", "bass", 1)
        ref_rel = _unique_wav_rel("raw", "master_mix", 1)
        storage.write_wav(proj.project_id, drums_rel, drums_wave, sample_rate=sr)
        storage.write_wav(proj.project_id, bass_rel, bass_wave, sample_rate=sr)
        storage.write_wav(proj.project_id, ref_rel, ref_wave, sample_rate=sr)

        proj.tracks = [
            TrackSchema(
                track_id="trk_ref",
                name="Full Mix (Reference)",
                track_type="audio",
                stem_role="full_mix",
                file_rel_path=ref_rel,
                revision=1,
                sample_rate=sr,
                end_sample=sr,
                muted=False,
            ),
            TrackSchema(
                track_id="trk_drums",
                name="Drums",
                track_type="stem",
                stem_role="drums",
                file_rel_path=drums_rel,
                revision=1,
                sample_rate=sr,
                end_sample=sr,
                muted=True,
                volume_db=-24.0,
                pan=0.8,
            ),
            TrackSchema(
                track_id="trk_bass",
                name="Bass",
                track_type="stem",
                stem_role="bass",
                file_rel_path=bass_rel,
                revision=1,
                sample_rate=sr,
                end_sample=sr,
                muted=False,
                volume_db=-12.0,
                pan=0.0,
            ),
        ]
        summed = sum_raw_stems(proj.project_id, proj.tracks, sr, sr)
        self.assertIsNotNone(summed)
        self.assertLess(float(np.max(np.abs(summed - (drums_wave + bass_wave)))), 5e-4)

        # 3. merge_track_mixer_updates applies mixer controls without clobbering server paths
        server_drums = next(trk for trk in proj.tracks if trk.track_id == "trk_drums")
        server_drums.file_rel_path = "stems/drums_rev2_server.wav"
        server_drums.revision = 2
        stale_client = TrackSchema(
            track_id="trk_drums",
            name="Drums Custom",
            track_type="stem",
            stem_role="drums",
            file_rel_path="stems/drums_rev1_stale.wav",
            revision=1,
            sample_rate=sr,
            end_sample=sr,
            muted=False,
            volume_db=-3.5,
            pan=-0.5,
        )
        merge_track_mixer_updates(proj, [stale_client])
        merged_drums = next(trk for trk in proj.tracks if trk.track_id == "trk_drums")
        self.assertEqual(merged_drums.file_rel_path, "stems/drums_rev2_server.wav")
        self.assertEqual(merged_drums.revision, 2)
        self.assertEqual(merged_drums.name, "Drums Custom")
        self.assertFalse(merged_drums.muted)
        self.assertAlmostEqual(merged_drums.volume_db, -3.5, places=5)
        self.assertAlmostEqual(merged_drums.pan, -0.5, places=5)
        self.assertTrue(any(trk.track_id == "trk_bass" for trk in proj.tracks))

        # 4. process_track_dsp hard-left pan (-1.0) & single_track_wav -12 dB preservation
        pan_track = TrackSchema(
            track_id="trk_pan",
            name="Pan Test",
            track_type="stem",
            stem_role="other",
            sample_rate=sr,
            end_sample=sr,
            volume_db=0.0,
            pan=-1.0,
        )
        pan_in = np.stack(
            [0.25 * np.sin(2.0 * np.pi * 440.0 * t), 0.25 * np.sin(2.0 * np.pi * 440.0 * t)],
            axis=1,
        ).astype(np.float32)
        pan_out = process_track_dsp(pan_in, pan_track, sr=sr)
        self.assertGreaterEqual(float(np.max(np.abs(pan_out[:, 0]))), 0.24)
        self.assertLess(float(np.max(np.abs(pan_out[:, 1]))), 1e-5)

        storage.save_project(proj)
        export_res = export_project_audio(
            proj,
            export_format="single_track_wav",
            track_id="trk_bass",
        )
        exported_audio, _ = storage.read_wav(
            proj.project_id, export_res["download_rel_path"], target_sr=sr
        )
        exported_peak = float(np.max(np.abs(exported_audio)))
        self.assertLess(exported_peak, 0.18)
        self.assertGreater(exported_peak, 0.09)

    def test_06_float32_wav_headroom_auth_and_acestep_v15_adapter(self) -> None:
        from dataclasses import dataclass
        from scipy.io import wavfile

        sr = 44100
        # 1. Lossless 32-bit float WAV preserves super-unity peaks (>1.0) without 16-bit clipping
        proj = storage.create_project("Float32 Auth & 1.5 Adapter", bpm=122.0, musical_key="F minor")
        n = sr // 4
        t = np.linspace(0.0, 0.25, n, endpoint=False, dtype=np.float32)
        orig = np.stack(
            [1.35 * np.sin(2.0 * np.pi * 220.0 * t), -1.20 * np.cos(2.0 * np.pi * 330.0 * t)],
            axis=1,
        ).astype(np.float32)
        storage.write_wav(proj.project_id, "raw/headroom_test.wav", orig, sample_rate=sr)
        read_back, read_sr = storage.read_wav(proj.project_id, "raw/headroom_test.wav", target_sr=sr)
        self.assertEqual(read_sr, sr)
        self.assertLess(float(np.max(np.abs(read_back - orig))), 1e-6)
        self.assertGreater(float(np.max(read_back)), 1.30)
        self.assertLess(float(np.min(read_back)), -1.15)

        # 2. LYRION_API_TOKEN auth & CORS preflight exemptions
        self.assertTrue(is_request_authorized("GET", "/api/v1/health", {}, {}, expected_token=""))
        self.assertTrue(is_request_authorized("OPTIONS", "/api/v1/projects", {}, {}, expected_token="sec_123"))
        self.assertTrue(is_request_authorized("GET", "/", {}, {}, expected_token="sec_123"))
        self.assertFalse(is_request_authorized("GET", "/api/v1/projects", {}, {}, expected_token="sec_123"))
        self.assertFalse(
            is_request_authorized(
                "GET",
                "/api/v1/projects",
                {"Authorization": "Bearer wrong_tok"},
                {},
                expected_token="sec_123",
            )
        )
        self.assertTrue(
            is_request_authorized(
                "GET",
                "/api/v1/projects",
                {"Authorization": "Bearer sec_123"},
                {},
                expected_token="sec_123",
            )
        )
        self.assertTrue(
            is_request_authorized(
                "GET",
                "/api/v1/projects/p1/audio/stems/drums.wav",
                {},
                {"token": "sec_123"},
                expected_token="sec_123",
            )
        )

        # 3. ACE-Step 1.5 adapter handles text2music and repaint with llm_handler=None
        captured_calls: list[dict[str, object]] = []

        @dataclass
        class FakeParams:
            task_type: str
            caption: str
            lyrics: str
            bpm: int
            keyscale: str
            duration: float
            seed: int
            inference_steps: int
            guidance_scale: float
            thinking: bool = False
            src_audio: str | None = None
            repainting_start: float = 0.0
            repainting_end: float = 0.0

        @dataclass
        class FakeConfig:
            batch_size: int = 1
            audio_format: str = "wav"

        @dataclass
        class FakeResult:
            success: bool
            audios: list[dict[str, str]]
            error: str | None = None

        def fake_generate_music(dit_handler, llm_handler, params, config, save_dir):
            self.assertIsNone(llm_handler)
            self.assertIsNotNone(dit_handler)
            self.assertFalse(params.thinking)
            captured_calls.append(
                {
                    "task_type": params.task_type,
                    "bpm": params.bpm,
                    "keyscale": params.keyscale,
                    "src_exists": bool(params.src_audio and Path(params.src_audio).is_file()),
                    "repainting_start": params.repainting_start,
                    "repainting_end": params.repainting_end,
                }
            )
            out_wav = Path(save_dir) / f"{params.task_type}_{params.seed}.wav"
            fake_audio = np.stack(
                [0.25 * np.sin(2.0 * np.pi * 440.0 * t), 0.25 * np.sin(2.0 * np.pi * 440.0 * t)],
                axis=1,
            ).astype(np.float32)
            wavfile.write(str(out_wav), sr, fake_audio)
            return FakeResult(success=True, audios=[{"path": str(out_wav)}])

        fake_bundle = {
            "version": "1.5",
            "dit_handler": object(),
            "generate_music": fake_generate_music,
            "GenerationParams": FakeParams,
            "GenerationConfig": FakeConfig,
        }

        gen_audio = _invoke_acestep_real_sync(
            fake_bundle,
            task_type="text2music",
            prompt="melodic house, 122 bpm, F minor",
            lyrics="[Verse]\nTest",
            bpm=122.0,
            musical_key="F minor",
            duration_sec=0.25,
            seed=101,
            inference_steps=20,
            guidance_scale=7.0,
            target_sr=sr,
            total_samples=n,
        )
        self.assertEqual(gen_audio.shape, (n, 2))
        self.assertEqual(gen_audio.dtype, np.float32)

        rep_audio = _invoke_acestep_real_sync(
            fake_bundle,
            task_type="repaint",
            prompt="acid synth, 122 bpm, F minor",
            lyrics="[Solo]",
            bpm=122.0,
            musical_key="F minor",
            duration_sec=0.25,
            seed=202,
            inference_steps=20,
            guidance_scale=7.0,
            target_sr=sr,
            total_samples=n,
            src_audio=orig,
            repaint_start_sec=0.05,
            repaint_end_sec=0.20,
        )
        self.assertEqual(rep_audio.shape, (n, 2))
        self.assertEqual(captured_calls[0]["task_type"], "text2music")
        self.assertFalse(captured_calls[0]["src_exists"])
        self.assertEqual(captured_calls[1]["task_type"], "repaint")
        self.assertTrue(captured_calls[1]["src_exists"])
        self.assertAlmostEqual(float(captured_calls[1]["repainting_start"]), 0.05, places=4)
        self.assertAlmostEqual(float(captured_calls[1]["repainting_end"]), 0.20, places=4)

        # 4. ACE-Step 1.5 project_root fallback & (status_msg, success) validation
        prev_root = os.environ.pop("ACESTEP_PROJECT_ROOT", None)
        try:
            resolved_root = _resolve_acestep_project_root()
            self.assertTrue(bool(resolved_root))
            self.assertTrue(Path(resolved_root).is_dir())
        finally:
            if prev_root is not None:
                os.environ["ACESTEP_PROJECT_ROOT"] = prev_root

        _check_acestep_init_result(("Ready", True))
        with self.assertRaises(RuntimeError):
            _check_acestep_init_result(("Missing checkpoint", False))

    def test_07_live_l4_quality_notebook_and_tempo_fixes(self):
        import json
        from dataclasses import dataclass
        from types import SimpleNamespace

        import numpy as np
        from scipy.io import wavfile

        from app.schemas import (
            AddInstrumentLayerRequest,
            GenerateSongRequest,
            LegoLayerRequest,
            RepaintRegionRequest,
            SingleTrackRepaintRequest,
        )
        from app.services.beat_service import _correct_octave_tempo

        # 1. ACE-Step 1.5 turbo defaults (steps=8, guidance=1.0, shift=3.0; bpm/key/duration/seed Auto=None)
        gen_req = GenerateSongRequest()
        self.assertEqual(gen_req.inference_steps, 8)
        self.assertAlmostEqual(gen_req.guidance_scale, 1.0, places=5)
        self.assertAlmostEqual(gen_req.shift, 3.0, places=5)
        self.assertIsNone(gen_req.bpm)
        self.assertIsNone(gen_req.musical_key)
        self.assertIsNone(gen_req.duration_sec)
        self.assertIsNone(gen_req.seed)

        spec = ProducerSpecSchema()
        self.assertEqual(spec.inference_steps, 8)
        self.assertAlmostEqual(spec.guidance_scale, 1.0, places=5)
        self.assertAlmostEqual(spec.shift, 3.0, places=5)
        self.assertIsNone(spec.seed)

        rep_req = RepaintRegionRequest(start_sample=0, end_sample=44100)
        single_req = SingleTrackRepaintRequest(target_track_id="trk_vocals_01", start_sample=0, end_sample=44100)
        add_req = AddInstrumentLayerRequest(
            track_name="Synth Lead", instrument_prompt="analog lead", start_sample=0, end_sample=44100
        )
        lego_req = LegoLayerRequest(reference_track_id="trk_master_01", start_sample=0, end_sample=44100)
        for r in (rep_req, single_req, add_req, lego_req):
            self.assertEqual(r.inference_steps, 8)
            self.assertAlmostEqual(r.guidance_scale, 1.0, places=5)
            self.assertAlmostEqual(r.shift, 3.0, places=5)

        # 2. LLMHandler thinking=True on text2music, thinking=False & llm_handler=None on repaint, shift=3.0 forwarded
        sr = 44100
        n = sr // 4
        t = np.linspace(0.0, 0.25, n, endpoint=False, dtype=np.float32)
        orig = np.zeros((n, 2), dtype=np.float32)
        captured: list[dict[str, object]] = []
        sentinel_llm = object()

        @dataclass
        class FakeTurboParams:
            task_type: str
            caption: str
            lyrics: str
            bpm: int
            keyscale: str
            duration: float
            seed: int
            inference_steps: int
            guidance_scale: float
            shift: float = 1.0
            thinking: bool = False
            src_audio: str | None = None
            repainting_start: float = 0.0
            repainting_end: float = 0.0

        @dataclass
        class FakeTurboConfig:
            batch_size: int = 1
            audio_format: str = "wav"

        @dataclass
        class FakeTurboResult:
            success: bool
            audios: list[dict[str, str]]

        def fake_turbo_generate(dit_handler, llm_handler, params, config, save_dir):
            captured.append(
                {
                    "task_type": params.task_type,
                    "llm_is_sentinel": llm_handler is sentinel_llm,
                    "llm_is_none": llm_handler is None,
                    "thinking": params.thinking,
                    "shift": params.shift,
                    "steps": params.inference_steps,
                    "guidance": params.guidance_scale,
                }
            )
            out_wav = Path(save_dir) / f"{params.task_type}_{params.seed}.wav"
            fake_audio = np.stack(
                [0.2 * np.sin(2.0 * np.pi * 330.0 * t), 0.2 * np.sin(2.0 * np.pi * 330.0 * t)],
                axis=1,
            ).astype(np.float32)
            wavfile.write(str(out_wav), sr, fake_audio)
            return FakeTurboResult(success=True, audios=[{"path": str(out_wav)}])

        turbo_bundle = {
            "version": "1.5",
            "dit_handler": object(),
            "llm_handler": sentinel_llm,
            "generate_music": fake_turbo_generate,
            "GenerationParams": FakeTurboParams,
            "GenerationConfig": FakeTurboConfig,
        }

        _invoke_acestep_real_sync(
            turbo_bundle,
            task_type="text2music",
            prompt="synthwave, 120 bpm, A minor",
            lyrics="[Verse 1]\nNeon",
            bpm=120.0,
            musical_key="A minor",
            duration_sec=0.25,
            seed=42,
            inference_steps=8,
            guidance_scale=1.0,
            shift=3.0,
            target_sr=sr,
            total_samples=n,
        )
        _invoke_acestep_real_sync(
            turbo_bundle,
            task_type="repaint",
            prompt="synthwave, 120 bpm, A minor",
            lyrics="[Chorus]\nGlow",
            bpm=120.0,
            musical_key="A minor",
            duration_sec=0.25,
            seed=43,
            inference_steps=8,
            guidance_scale=1.0,
            shift=3.0,
            target_sr=sr,
            total_samples=n,
            src_audio=orig,
            repaint_start_sec=0.05,
            repaint_end_sec=0.20,
        )

        self.assertEqual(len(captured), 2)
        self.assertTrue(captured[0]["llm_is_sentinel"])
        self.assertTrue(captured[0]["thinking"])
        self.assertAlmostEqual(float(captured[0]["shift"]), 3.0, places=5)
        self.assertEqual(captured[0]["steps"], 8)
        self.assertAlmostEqual(float(captured[0]["guidance"]), 1.0, places=5)

        self.assertTrue(captured[1]["llm_is_none"])
        self.assertFalse(captured[1]["thinking"])
        self.assertAlmostEqual(float(captured[1]["shift"]), 3.0, places=5)

        # 3. model_manager._unload_current clears inner dict references cleanly
        fake_dit = SimpleNamespace(model=object(), vae=object(), text_encoder=object())
        fake_llm = SimpleNamespace(llm=object(), model=object())
        fake_dict_bundle = {"dit_handler": fake_dit, "llm_handler": fake_llm}
        model_manager._active_model = fake_dict_bundle
        model_manager._active_type = ModelType.ACE_STEP
        model_manager._unload_current()
        self.assertIsNone(model_manager._active_model)
        self.assertEqual(model_manager._active_type, ModelType.NONE)
        self.assertEqual(len(fake_dict_bundle), 0)
        self.assertIsNone(fake_dit.model)
        self.assertIsNone(fake_dit.vae)
        self.assertIsNone(fake_llm.llm)

        # 4. _correct_octave_tempo fixes half-tempo & double-tempo detections
        doubled_bpm, doubled_beats = _correct_octave_tempo(
            59.4, [0.0, 1.01, 2.02, 3.03], target_bpm=120.0, duration_sec=4.0
        )
        self.assertAlmostEqual(doubled_bpm, 118.8, places=2)
        self.assertGreaterEqual(len(doubled_beats), 7)
        self.assertAlmostEqual(doubled_beats[1], 0.505, places=3)

        halved_bpm, halved_beats = _correct_octave_tempo(
            180.0, [0.0, 0.333333, 0.666667, 1.0, 1.333333], target_bpm=90.0, duration_sec=2.0
        )
        self.assertAlmostEqual(halved_bpm, 90.0, places=2)
        self.assertEqual(len(halved_beats), 3)

        # 5. Lyrion_Colab_L4.ipynb is valid JSON (nbformat 4) with 3 ordered code cells
        nb_path = Path(__file__).resolve().parents[2] / "Lyrion_Colab_L4.ipynb"
        self.assertTrue(nb_path.is_file())
        nb_data = json.loads(nb_path.read_text(encoding="utf-8"))
        self.assertEqual(nb_data.get("nbformat"), 4)
        code_cells = [c for c in nb_data.get("cells", []) if c.get("cell_type") == "code"]
        self.assertEqual(len(code_cells), 3)
        cell_texts = ["\n".join(c.get("source", [])) for c in code_cells]
        self.assertIn("drive.mount", cell_texts[0])
        self.assertIn("kingkong-cfl/lyrionalpha", cell_texts[1])
        self.assertIn("%run scripts/colab_bootstrap.py", cell_texts[2])

    def test_08_seed_config_unload_parity_preview_and_hygiene(self):
        import importlib.util
        import json
        from dataclasses import dataclass

        import numpy as np
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.responses import FileResponse
        from scipy.io import wavfile

        from app.config import Settings
        from app.routers.projects import _iter_file_range, build_audio_file_response
        from app.services.dsp_service import export_project_audio

        # 1. Seed propagation via GenerationConfig(use_random_seed=False, seeds=[seed])
        sr = 44100
        n = sr // 4
        t = np.linspace(0.0, 0.25, n, endpoint=False, dtype=np.float32)
        captured_cfg: dict[str, object] = {}

        @dataclass
        class FakeParamsNoSeed:
            task_type: str
            caption: str
            lyrics: str
            bpm: int
            keyscale: str
            duration: float
            inference_steps: int
            guidance_scale: float
            shift: float = 3.0
            thinking: bool = False

        @dataclass
        class FakeConfigWithSeeds:
            batch_size: int = 1
            audio_format: str = "wav"
            use_random_seed: bool = True
            seeds: list[int] | None = None

        @dataclass
        class FakeResult:
            success: bool
            audios: list[dict[str, str]]

        def fake_gen_with_cfg(dit_handler, llm_handler, params, config, save_dir):
            captured_cfg["use_random_seed"] = config.use_random_seed
            captured_cfg["seeds"] = list(config.seeds or [])
            out_wav = Path(save_dir) / "seeded.wav"
            fake_audio = np.stack(
                [0.25 * np.sin(2.0 * np.pi * 220.0 * t), 0.25 * np.sin(2.0 * np.pi * 220.0 * t)],
                axis=1,
            ).astype(np.float32)
            wavfile.write(str(out_wav), sr, fake_audio)
            return FakeResult(success=True, audios=[{"path": str(out_wav)}])

        _invoke_acestep_real_sync(
            {
                "version": "1.5",
                "dit_handler": object(),
                "generate_music": fake_gen_with_cfg,
                "GenerationParams": FakeParamsNoSeed,
                "GenerationConfig": FakeConfigWithSeeds,
            },
            task_type="text2music",
            prompt="house groove",
            lyrics="[Verse 1]\nLine",
            bpm=124.0,
            musical_key="C minor",
            duration_sec=0.25,
            seed=777123,
            inference_steps=8,
            guidance_scale=1.0,
            shift=3.0,
            target_sr=sr,
            total_samples=n,
        )
        self.assertIs(captured_cfg["use_random_seed"], False)
        self.assertEqual(captured_cfg["seeds"], [777123])

        # 2. ModelManager._unload_current() deep cleanup (unload hook + arbitrary attrs + count)
        class FakeHandlerWithUnload:
            def __init__(self) -> None:
                self.unload_called = False
                self.dit = object()
                self.lm_model = object()
                self.silence_latent = np.zeros(16, dtype=np.float32)
                self.custom_tensor = np.ones(16, dtype=np.float32)

            def unload(self) -> None:
                self.unload_called = True

        handler_direct = FakeHandlerWithUnload()
        before_count = model_manager.unload_count
        model_manager._active_model = handler_direct
        model_manager._active_type = ModelType.ACE_STEP
        model_manager._unload_current()
        self.assertTrue(handler_direct.unload_called)
        self.assertIsNone(handler_direct.dit)
        self.assertIsNone(handler_direct.lm_model)
        self.assertIsNone(handler_direct.silence_latent)
        self.assertFalse(hasattr(handler_direct, "custom_tensor"))
        self.assertEqual(model_manager.unload_count, before_count + 1)

        handler_in_dict = FakeHandlerWithUnload()
        bundle = {"dit_handler": handler_in_dict, "arbitrary_cache": {"big": np.ones(8)}}
        model_manager._active_model = bundle
        model_manager._active_type = ModelType.ACE_STEP
        model_manager._unload_current()
        self.assertTrue(handler_in_dict.unload_called)
        self.assertEqual(len(bundle), 0)
        self.assertIsNone(handler_in_dict.dit)

        # 3. Solo + Mute & All-Muted Export Parity (no silent Full Mix fallback)
        proj = storage.create_project("Parity Suite", bpm=120.0, musical_key="A minor")
        proj.grid.sample_rate = sr
        proj.grid.total_samples = n
        proj.grid.duration_sec = 0.25

        drums_w = np.stack(
            [0.30 * np.sin(2.0 * np.pi * 110.0 * t), 0.30 * np.sin(2.0 * np.pi * 110.0 * t)],
            axis=1,
        ).astype(np.float32)
        bass_w = np.stack(
            [0.40 * np.sin(2.0 * np.pi * 220.0 * t), 0.40 * np.sin(2.0 * np.pi * 220.0 * t)],
            axis=1,
        ).astype(np.float32)
        full_w = np.full((n, 2), 0.75, dtype=np.float32)

        drums_rel = _unique_wav_rel("stems", "drums", 1)
        bass_rel = _unique_wav_rel("stems", "bass", 1)
        full_rel = _unique_wav_rel("raw", "master_mix", 1)
        storage.write_wav(proj.project_id, drums_rel, drums_w, sample_rate=sr)
        storage.write_wav(proj.project_id, bass_rel, bass_w, sample_rate=sr)
        storage.write_wav(proj.project_id, full_rel, full_w, sample_rate=sr)

        def make_tracks(drums_muted: bool, drums_solo: bool, bass_muted: bool, bass_solo: bool, full_muted: bool):
            return [
                TrackSchema(
                    track_id="trk_full",
                    name="Full Mix (Reference)",
                    track_type="audio",
                    stem_role="full_mix",
                    file_rel_path=full_rel,
                    revision=1,
                    sample_rate=sr,
                    end_sample=n,
                    muted=full_muted,
                    soloed=False,
                ),
                TrackSchema(
                    track_id="trk_drums",
                    name="Drums",
                    track_type="stem",
                    stem_role="drums",
                    file_rel_path=drums_rel,
                    revision=1,
                    sample_rate=sr,
                    end_sample=n,
                    muted=drums_muted,
                    soloed=drums_solo,
                ),
                TrackSchema(
                    track_id="trk_bass",
                    name="Bass",
                    track_type="stem",
                    stem_role="bass",
                    file_rel_path=bass_rel,
                    revision=1,
                    sample_rate=sr,
                    end_sample=n,
                    muted=bass_muted,
                    soloed=bass_solo,
                ),
            ]

        # Case A: soloed=True AND muted=True -> Mute wins -> silence
        proj.tracks = make_tracks(drums_muted=True, drums_solo=True, bass_muted=False, bass_solo=False, full_muted=True)
        res_a = export_project_audio(proj, export_format="wav", normalize_lufs=False)
        aud_a, _ = storage.read_wav(proj.project_id, res_a["download_rel_path"], target_sr=sr)
        self.assertEqual(float(np.max(np.abs(aud_a))), 0.0)

        # Case B: soloed=True AND muted=False on Drums -> only Drums mixed (matches non-zero bass exclusion)
        proj.tracks = make_tracks(drums_muted=False, drums_solo=True, bass_muted=False, bass_solo=False, full_muted=True)
        res_b = export_project_audio(proj, export_format="wav", normalize_lufs=False)
        aud_b, _ = storage.read_wav(proj.project_id, res_b["download_rel_path"], target_sr=sr)
        self.assertLess(float(np.max(np.abs(aud_b - drums_w))), 1e-3)

        # Case C: ALL tracks muted -> strict silence (never falls back to unmuted Full Mix)
        proj.tracks = make_tracks(drums_muted=True, drums_solo=False, bass_muted=True, bass_solo=False, full_muted=True)
        res_c = export_project_audio(proj, export_format="wav", normalize_lufs=False)
        aud_c, _ = storage.read_wav(proj.project_id, res_c["download_rel_path"], target_sr=sr)
        self.assertEqual(float(np.max(np.abs(aud_c))), 0.0)

        # 4. Preview Audio Streaming (?preview=1) + Range 206 + Cache-Control: private + CORS allow_credentials=False
        orig_path = storage.resolve_project_file(proj.project_id, drums_rel)
        orig_size = orig_path.stat().st_size
        preview_path, preview_mime = storage.get_or_create_preview_audio(proj.project_id, drums_rel)
        self.assertTrue(preview_path.is_file())
        self.assertIn(preview_mime, ("audio/flac", "audio/wav"))
        self.assertLess(preview_path.stat().st_size, orig_size * 0.65)
        preview_mtime = preview_path.stat().st_mtime_ns

        # Second call hits cached preview file without rewriting
        preview_path_2, _ = storage.get_or_create_preview_audio(proj.project_id, drums_rel)
        self.assertEqual(preview_path_2, preview_path)
        self.assertEqual(preview_path_2.stat().st_mtime_ns, preview_mtime)

        decoded_preview, _ = storage.read_wav(
            proj.project_id,
            preview_path.relative_to(storage.project_dir(proj.project_id)).as_posix(),
            target_sr=sr,
        )
        self.assertEqual(decoded_preview.shape, drums_w.shape)
        self.assertLess(float(np.max(np.abs(decoded_preview - drums_w))), 2e-4)

        preview_url = f"https://testserver/api/v1/projects/{proj.project_id}/audio/{drums_rel}?rev=1&preview=1"
        range_resp = build_audio_file_response(
            proj.project_id,
            drums_rel,
            preview_url,
            "bytes=0-1023",
            preview=True,
        )
        self.assertEqual(range_resp.status_code, 206)
        self.assertIn("private", range_resp.headers.get("cache-control", "").lower())
        self.assertNotIn("public", range_resp.headers.get("cache-control", "").lower())
        self.assertIn("immutable", range_resp.headers.get("cache-control", "").lower())
        self.assertEqual(range_resp.headers.get("content-length"), "1024")
        self.assertEqual(len(b"".join(_iter_file_range(preview_path, 0, 1023))), 1024)

        full_resp = build_audio_file_response(
            proj.project_id,
            drums_rel,
            preview_url,
            None,
            preview=True,
        )
        self.assertIsInstance(full_resp, FileResponse)
        self.assertEqual(Path(full_resp.path), preview_path)
        self.assertIn("private", full_resp.headers.get("cache-control", "").lower())

        cors_entries = [m for m in app.user_middleware if m.cls is CORSMiddleware]
        self.assertEqual(len(cors_entries), 1)
        self.assertIs(cors_entries[0].kwargs.get("allow_credentials"), False)

        # 5. Gemini defaults, Bootstrap Version Gate & Shared Schema Shift Field
        prev_gm = os.environ.pop("GEMINI_MODEL", None)
        prev_gf = os.environ.pop("GEMINI_FALLBACK_MODEL", None)
        try:
            fresh_settings = Settings()
            self.assertEqual(fresh_settings.gemini_model, "gemini-2.5-flash")
            self.assertEqual(fresh_settings.gemini_fallback_model, "gemini-2.5-flash-lite")
        finally:
            if prev_gm is not None:
                os.environ["GEMINI_MODEL"] = prev_gm
            if prev_gf is not None:
                os.environ["GEMINI_FALLBACK_MODEL"] = prev_gf

        bootstrap_path = Path(__file__).resolve().parents[1] / "scripts" / "colab_bootstrap.py"
        spec = importlib.util.spec_from_file_location("colab_bootstrap_under_test", bootstrap_path)
        self.assertIsNotNone(spec)
        bootstrap_mod = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(bootstrap_mod)
        self.assertTrue(bootstrap_mod.check_acestep_versions("4.57.1", "0.35.2", "0.36.0"))
        self.assertFalse(bootstrap_mod.check_acestep_versions("5.0.0", "0.35.2", "0.36.0"))
        self.assertFalse(bootstrap_mod.check_acestep_versions("4.57.1", "0.35.2", "1.0.0"))
        self.assertFalse(bootstrap_mod.check_acestep_versions("4.57.1", "0.34.0", "0.36.0"))

        shared_schema_path = Path(__file__).resolve().parents[2] / "shared" / "schemas" / "project.schema.json"
        shared_schema = json.loads(shared_schema_path.read_text(encoding="utf-8"))
        self.assertIn("shift", shared_schema["properties"]["producer_spec"]["properties"])

    def test_09_auto_params_split_enhance_and_1_7b_lm_default(self):
        import json
        import sys
        import types
        from dataclasses import dataclass

        import numpy as np
        from pydantic import ValidationError
        from scipy.io import wavfile

        from app.schemas import GenerateSongRequest, ProducerEnhanceRequest
        from app.services import acestep_service, beat_service
        from app.services.acestep_service import (
            _build_acestep_prompt,
            _load_acestep_model,
            _read_and_fit_stereo,
            resolve_mock_generation_params,
        )
        from app.services.llm_producer import _normalize_response

        # 1. GenerateSongRequest Auto (None) & bounds validation
        auto_req = GenerateSongRequest()
        self.assertIsNone(auto_req.bpm)
        self.assertIsNone(auto_req.musical_key)
        self.assertIsNone(auto_req.duration_sec)
        self.assertIsNone(auto_req.seed)
        with self.assertRaises(ValidationError):
            GenerateSongRequest(duration_sec=5.0)
        explicit_req = GenerateSongRequest(bpm=128.0, musical_key="F# minor", duration_sec=30.0, seed=99)
        self.assertEqual(explicit_req.bpm, 128.0)
        self.assertEqual(explicit_req.duration_sec, 30.0)
        self.assertEqual(explicit_req.seed, 99)

        # 2. _build_acestep_prompt omits Auto fields cleanly
        self.assertEqual(_build_acestep_prompt(["synthwave", "punchy drums"], None, None), "synthwave, punchy drums")
        self.assertEqual(
            _build_acestep_prompt(["synthwave"], 120.0, "A minor"),
            "synthwave, 120 bpm, A minor",
        )

        # 3. _invoke_acestep_real_sync with all Auto (None) args + natural output duration
        sr = 44100
        natural_samples = 16317  # ~0.37s non-standard length preserved when total_samples=None
        captured: dict[str, object] = {}

        @dataclass
        class FakeAutoParams:
            task_type: str
            caption: str
            lyrics: str
            bpm: int | None = None
            keyscale: str = ""
            duration: float = -1.0
            inference_steps: int = 8
            guidance_scale: float = 1.0
            shift: float = 3.0
            thinking: bool = False

        @dataclass
        class FakeAutoConfig:
            batch_size: int = 1
            audio_format: str = "wav"
            use_random_seed: bool = False
            seeds: list[int] | None = None

        @dataclass
        class FakeResult:
            success: bool
            audios: list[dict[str, str]]

        def fake_auto_gen(dit_handler, llm_handler, params, config, save_dir):
            captured["bpm"] = params.bpm
            captured["keyscale"] = params.keyscale
            captured["duration"] = params.duration
            captured["thinking"] = params.thinking
            captured["llm_present"] = llm_handler is not None
            captured["use_random_seed"] = config.use_random_seed
            captured["seeds"] = config.seeds

            out_wav = Path(save_dir) / "natural_auto.wav"
            t = np.linspace(0.0, natural_samples / float(sr), natural_samples, endpoint=False, dtype=np.float32)
            audio = np.stack([0.2 * np.sin(2 * np.pi * 220 * t), 0.2 * np.sin(2 * np.pi * 220 * t)], axis=1).astype(
                np.float32
            )
            wavfile.write(str(out_wav), sr, audio)
            return FakeResult(success=True, audios=[{"path": str(out_wav)}])

        out_audio = _invoke_acestep_real_sync(
            {
                "version": "1.5",
                "dit_handler": object(),
                "llm_handler": object(),
                "generate_music": fake_auto_gen,
                "GenerationParams": FakeAutoParams,
                "GenerationConfig": FakeAutoConfig,
            },
            task_type="text2music",
            prompt=_build_acestep_prompt(["synthwave"], None, None),
            lyrics="[Verse 1]\nNeon highway",
            bpm=None,
            musical_key=None,
            duration_sec=None,
            seed=None,
            inference_steps=8,
            guidance_scale=1.0,
            target_sr=sr,
            total_samples=None,
            shift=3.0,
        )
        self.assertIsNone(captured["bpm"])
        self.assertEqual(captured["keyscale"], "")
        self.assertEqual(captured["duration"], -1.0)
        self.assertIs(captured["thinking"], True)
        self.assertIs(captured["llm_present"], True)
        self.assertIs(captured["use_random_seed"], True)
        self.assertIsNone(captured["seeds"])
        self.assertEqual(out_audio.shape, (natural_samples, 2))

        # Verify max_duration_sec safety cap when total_samples=None
        with tempfile.TemporaryDirectory() as tmpd:
            long_wav = Path(tmpd) / "long.wav"
            long_buf = np.zeros((sr, 2), dtype=np.float32)
            wavfile.write(str(long_wav), sr, long_buf)
            capped = _read_and_fit_stereo(long_wav, sr, total_samples=None, max_duration_sec=0.2)
            self.assertEqual(capped.shape, (int(round(0.2 * sr)), 2))

        # 4. MOCK_GPU fallback defaults for Auto mode (28s, 116 BPM, A minor, 421337)
        mbpm, mkey, mdur, mseed = resolve_mock_generation_params(
            bpm=None, musical_key=None, duration_sec=None, seed=None
        )
        self.assertEqual(mbpm, 116.0)
        self.assertEqual(mkey, "A minor")
        self.assertEqual(mdur, 28.0)
        self.assertEqual(mseed, 421337)

        # 5. Beat grid handles target_bpm=None without octave doubling, vs target_bpm=120.0 doubling 59.4 -> 118.8
        orig_mock_flag = beat_service.settings.mock_gpu
        orig_librosa = beat_service.librosa
        try:
            object.__setattr__(beat_service.settings, "mock_gpu", False)
            fake_librosa = types.SimpleNamespace(
                beat=types.SimpleNamespace(beat_track=lambda y, sr, start_bpm, units: (59.4, np.array([0, 44, 88]))),
                frames_to_time=lambda frames, sr: np.array([0.0, 1.01, 2.02]),
            )
            beat_service.librosa = fake_librosa
            probe_audio = np.zeros((sr * 3, 2), dtype=np.float32)

            grid_auto = analyze_beat_grid(probe_audio, sr, target_bpm=None, musical_key=None)
            self.assertAlmostEqual(grid_auto.bpm, 59.4, places=2)
            self.assertEqual(grid_auto.musical_key, "Auto")

            grid_target = analyze_beat_grid(probe_audio, sr, target_bpm=120.0, musical_key="")
            self.assertAlmostEqual(grid_target.bpm, 118.8, places=2)
            self.assertEqual(grid_target.musical_key, "Auto")
        finally:
            object.__setattr__(beat_service.settings, "mock_gpu", orig_mock_flag)
            beat_service.librosa = orig_librosa

        # 6. Split Enhance modes ("tags", "lyrics", "all")
        req_tags = ProducerEnhanceRequest(raw_prompt="dark synthwave", mode="tags", existing_lyrics="[Verse 1]\nKeep me")
        self.assertEqual(req_tags.mode, "tags")

        res_tags = _deterministic_fallback(
            "dark synthwave drive",
            preferred_duration=None,
            mode="tags",
            existing_lyrics="[Verse 1]\nUser custom line",
        )
        self.assertEqual(res_tags.mode, "tags")
        self.assertGreaterEqual(len(res_tags.style_tags), 3)
        self.assertEqual(res_tags.structured_lyrics, "[Verse 1]\nUser custom line")

        res_lyrics = _deterministic_fallback(
            "melancholic night walk",
            preferred_duration=None,
            mode="lyrics",
            existing_style_tags=["ambient", "warm tape"],
        )
        self.assertEqual(res_lyrics.mode, "lyrics")
        self.assertEqual(res_lyrics.style_tags, ["ambient", "warm tape"])
        self.assertIn("[Verse 1]", res_lyrics.structured_lyrics)
        self.assertIn("[Chorus]", res_lyrics.structured_lyrics)

        norm_tags = _normalize_response(
            {"style_tags": ["cyberpunk", "heavy bass"]},
            provider="test-llm",
            preferred_duration=None,
            mode="tags",
            existing_lyrics="[Chorus]\nFixed chorus",
        )
        self.assertEqual(norm_tags.style_tags, ["cyberpunk", "heavy bass"])
        self.assertEqual(norm_tags.structured_lyrics, "[Chorus]\nFixed chorus")

        # 7. Default 5Hz LM is acestep-5Hz-lm-1.7B in both _load_acestep_model and Lyrion_Colab_L4.ipynb
        captured_lm_init: dict[str, object] = {}

        class FakeDitHandler:
            def initialize_service(self, project_root, config_path, device):
                return ("Ready", True)

        class FakeLlmHandler:
            def initialize(self, checkpoint_dir, lm_model_path, backend, device, offload_to_cpu=False):
                captured_lm_init["lm_model_path"] = lm_model_path
                captured_lm_init["backend"] = backend
                return ("LM Ready", True)

        fake_torch_mod = types.ModuleType("torch")
        fake_torch_mod.cuda = types.SimpleNamespace(is_available=lambda: False)
        fake_acestep_pkg = types.ModuleType("acestep")
        fake_handler_mod = types.ModuleType("acestep.handler")
        fake_handler_mod.AceStepHandler = FakeDitHandler
        fake_inf_mod = types.ModuleType("acestep.inference")
        fake_inf_mod.GenerationConfig = FakeAutoConfig
        fake_inf_mod.GenerationParams = FakeAutoParams
        fake_inf_mod.generate_music = fake_auto_gen
        fake_llm_mod = types.ModuleType("acestep.llm_inference")
        fake_llm_mod.LLMHandler = FakeLlmHandler

        prev_lm_env = os.environ.pop("ACESTEP_LM_MODEL_PATH", None)
        orig_ace_mock = acestep_service.settings.mock_gpu
        saved_mods = {
            k: sys.modules.get(k)
            for k in ("torch", "acestep", "acestep.handler", "acestep.inference", "acestep.llm_inference")
        }
        try:
            object.__setattr__(acestep_service.settings, "mock_gpu", False)
            sys.modules["torch"] = fake_torch_mod
            sys.modules["acestep"] = fake_acestep_pkg
            sys.modules["acestep.handler"] = fake_handler_mod
            sys.modules["acestep.inference"] = fake_inf_mod
            sys.modules["acestep.llm_inference"] = fake_llm_mod
            bundle = _load_acestep_model()
            self.assertEqual(bundle["version"], "1.5")
            self.assertEqual(captured_lm_init.get("lm_model_path"), "acestep-5Hz-lm-1.7B")
            self.assertEqual(captured_lm_init.get("backend"), "pt")
        finally:
            object.__setattr__(acestep_service.settings, "mock_gpu", orig_ace_mock)
            if prev_lm_env is not None:
                os.environ["ACESTEP_LM_MODEL_PATH"] = prev_lm_env
            for k, v in saved_mods.items():
                if v is None:
                    sys.modules.pop(k, None)
                else:
                    sys.modules[k] = v

        nb_path = Path(__file__).resolve().parents[2] / "Lyrion_Colab_L4.ipynb"
        nb_text = json.dumps(json.loads(nb_path.read_text(encoding="utf-8")))
        self.assertIn("acestep-5Hz-lm-1.7B", nb_text)

    def test_10_quality_turkish_cot_profiles_ab_and_gemini(self):
        import inspect
        import json
        import sys
        import types
        from dataclasses import dataclass

        import numpy as np
        from scipy.io import wavfile

        from app.routers.generation import resolve_effective_style_tags
        from app.services import acestep_service, llm_producer
        from app.services.acestep_service import (
            _load_acestep_model,
            _read_and_fit_stereo,
            resolve_model_inference_profile,
            resolve_vocal_language,
        )

        # 1. resolve_vocal_language auto-detection (Turkish diacritics, ASCII Turkish words, English, explicit)
        self.assertEqual(resolve_vocal_language(None, "[Verse 1]\nYağmur yağıyor şehirde gece"), ("tr", False))
        self.assertEqual(resolve_vocal_language("auto", "[Verse 1]\nBu gece kalp ve zaman susar"), ("tr", False))
        self.assertEqual(resolve_vocal_language(None, "[Verse 1]\nMidnight lights fade into the horizon"), ("unknown", True))
        self.assertEqual(resolve_vocal_language("en", "[Verse 1]\nTürkçe kelime olsa bile"), ("en", False))
        self.assertEqual(resolve_vocal_language("instrumental", ""), ("unknown", False))

        # 2. _invoke_acestep_real_sync passes vocal_language & CoT params on text2music, False on repaint
        sr = 44100
        n = sr // 4
        t = np.linspace(0.0, 0.25, n, endpoint=False, dtype=np.float32)
        orig = np.zeros((n, 2), dtype=np.float32)
        cot_calls: list[dict[str, object]] = []

        @dataclass
        class FakeCotParams:
            task_type: str
            caption: str
            lyrics: str
            bpm: int | None = None
            keyscale: str = ""
            duration: float = -1.0
            vocal_language: str = "unknown"
            inference_steps: int = 8
            guidance_scale: float = 1.0
            shift: float = 3.0
            thinking: bool = False
            use_cot_metas: bool = False
            use_cot_caption: bool = False
            use_cot_language: bool = False
            use_constrained_decoding: bool = False
            lm_cfg_scale: float = 2.0
            src_audio: str | None = None
            repainting_start: float = 0.0
            repainting_end: float = 0.0

        @dataclass
        class FakeCotConfig:
            batch_size: int = 1
            audio_format: str = "wav"
            use_random_seed: bool = False
            seeds: list[int] | None = None

        @dataclass
        class FakeCotResult:
            success: bool
            audios: list[dict[str, str]]

        def fake_cot_generate(dit_handler, llm_handler, params, config, save_dir):
            cot_calls.append(
                {
                    "task_type": params.task_type,
                    "vocal_language": params.vocal_language,
                    "use_cot_metas": params.use_cot_metas,
                    "use_cot_caption": params.use_cot_caption,
                    "use_cot_language": params.use_cot_language,
                    "use_constrained_decoding": params.use_constrained_decoding,
                    "lm_cfg_scale": params.lm_cfg_scale,
                    "steps": params.inference_steps,
                    "guidance": params.guidance_scale,
                }
            )
            out_wav = Path(save_dir) / f"cot_{params.task_type}.wav"
            audio = np.stack([0.2 * np.sin(2 * np.pi * 220 * t), 0.2 * np.sin(2 * np.pi * 220 * t)], axis=1).astype(
                np.float32
            )
            wavfile.write(str(out_wav), sr, audio)
            return FakeCotResult(success=True, audios=[{"path": str(out_wav)}])

        cot_bundle = {
            "version": "1.5",
            "config_path": "acestep-v15-xl-turbo",
            "display_name": "acestep-v15-xl-turbo + lm-4B",
            "dit_handler": object(),
            "llm_handler": object(),
            "generate_music": fake_cot_generate,
            "GenerationParams": FakeCotParams,
            "GenerationConfig": FakeCotConfig,
        }
        _invoke_acestep_real_sync(
            cot_bundle,
            task_type="text2music",
            prompt="indie rock",
            lyrics="[Verse 1]\nKaranlık sokaklar ve yağmur",
            bpm=None,
            musical_key=None,
            duration_sec=None,
            seed=None,
            inference_steps=8,
            guidance_scale=1.0,
            target_sr=sr,
            total_samples=n,
            shift=3.0,
            vocal_language=None,
        )
        _invoke_acestep_real_sync(
            cot_bundle,
            task_type="repaint",
            prompt="indie rock",
            lyrics="[Chorus]\nRun away",
            bpm=120.0,
            musical_key="A minor",
            duration_sec=0.25,
            seed=55,
            inference_steps=8,
            guidance_scale=1.0,
            target_sr=sr,
            total_samples=n,
            shift=3.0,
            vocal_language=None,
            src_audio=orig,
            repaint_start_sec=0.05,
            repaint_end_sec=0.20,
        )
        self.assertEqual(cot_calls[0]["vocal_language"], "tr")
        self.assertIs(cot_calls[0]["use_cot_metas"], True)
        self.assertIs(cot_calls[0]["use_cot_caption"], True)
        self.assertIs(cot_calls[0]["use_cot_language"], False)
        self.assertIs(cot_calls[0]["use_constrained_decoding"], True)
        self.assertAlmostEqual(float(cot_calls[0]["lm_cfg_scale"]), 2.0, places=4)

        self.assertIs(cot_calls[1]["use_cot_metas"], False)
        self.assertIs(cot_calls[1]["use_cot_caption"], False)
        self.assertIs(cot_calls[1]["use_cot_language"], False)
        self.assertIs(cot_calls[1]["use_constrained_decoding"], False)

        # 3. _read_and_fit_stereo max_duration_sec default is 240.0
        sig_read = inspect.signature(_read_and_fit_stereo)
        self.assertEqual(sig_read.parameters["max_duration_sec"].default, 240.0)

        # 4. resolve_model_inference_profile upgrades sft/base turbo defaults -> (32, 7.0), preserves turbo & overrides
        self.assertEqual(resolve_model_inference_profile("acestep-v15-xl-sft", 8, 1.0, 3.0), (32, 7.0, 3.0))
        self.assertEqual(resolve_model_inference_profile("acestep-v15-xl-sft", 50, 6.5, 3.0), (50, 6.5, 3.0))
        self.assertEqual(resolve_model_inference_profile("acestep-v15-xl-turbo", 8, 1.0, 3.0), (8, 1.0, 3.0))

        # 5. _load_acestep_model forwards offload_to_cpu & exposes active_model in model_manager telemetry
        init_capture: dict[str, object] = {}

        class FakeDitOffload:
            def initialize_service(self, project_root, config_path, device, offload_to_cpu=False):
                init_capture["dit_cfg"] = config_path
                init_capture["dit_offload"] = offload_to_cpu
                return ("Ready", True)

        class FakeLlmOffload:
            def initialize(self, checkpoint_dir, lm_model_path, backend, device, offload_to_cpu=False):
                init_capture["lm_path"] = lm_model_path
                init_capture["lm_offload"] = offload_to_cpu
                return ("LM Ready", True)

        class FakeLlmFail:
            def initialize(self, checkpoint_dir, lm_model_path, backend, device, offload_to_cpu=False):
                raise RuntimeError("Out of disk")

        fake_torch_mod = types.ModuleType("torch")
        fake_torch_mod.cuda = types.SimpleNamespace(is_available=lambda: False)
        fake_acestep_pkg = types.ModuleType("acestep")
        fake_handler_mod = types.ModuleType("acestep.handler")
        fake_handler_mod.AceStepHandler = FakeDitOffload
        fake_inf_mod = types.ModuleType("acestep.inference")
        fake_inf_mod.GenerationConfig = FakeCotConfig
        fake_inf_mod.GenerationParams = FakeCotParams
        fake_inf_mod.generate_music = fake_cot_generate
        fake_llm_mod = types.ModuleType("acestep.llm_inference")
        fake_llm_mod.LLMHandler = FakeLlmOffload

        saved_env = {k: os.environ.get(k) for k in ("ACESTEP_CONFIG_PATH", "ACESTEP_LM_MODEL_PATH", "ACESTEP_OFFLOAD_TO_CPU")}
        saved_mods = {
            k: sys.modules.get(k)
            for k in ("torch", "acestep", "acestep.handler", "acestep.inference", "acestep.llm_inference")
        }
        orig_ace_mock = acestep_service.settings.mock_gpu
        try:
            os.environ["ACESTEP_CONFIG_PATH"] = "acestep-v15-xl-turbo"
            os.environ["ACESTEP_LM_MODEL_PATH"] = "acestep-5Hz-lm-4B"
            os.environ["ACESTEP_OFFLOAD_TO_CPU"] = "true"
            object.__setattr__(acestep_service.settings, "mock_gpu", False)
            sys.modules["torch"] = fake_torch_mod
            sys.modules["acestep"] = fake_acestep_pkg
            sys.modules["acestep.handler"] = fake_handler_mod
            sys.modules["acestep.inference"] = fake_inf_mod
            sys.modules["acestep.llm_inference"] = fake_llm_mod

            loaded_bundle = _load_acestep_model()
            self.assertIs(init_capture["dit_offload"], True)
            self.assertIs(init_capture["lm_offload"], True)
            self.assertEqual(loaded_bundle["display_name"], "acestep-v15-xl-turbo + lm-4B")
            model_manager._active_model = loaded_bundle
            model_manager._active_type = ModelType.ACE_STEP
            self.assertEqual(model_manager.get_telemetry().active_model, "acestep-v15-xl-turbo + lm-4B")

            fake_llm_mod.LLMHandler = FakeLlmFail
            fallback_bundle = _load_acestep_model()
            self.assertEqual(fallback_bundle["display_name"], "acestep-v15-xl-turbo + lm=OFF")
        finally:
            model_manager._active_model = None
            model_manager._active_type = ModelType.NONE
            object.__setattr__(acestep_service.settings, "mock_gpu", orig_ace_mock)
            for k, v in saved_env.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
            for k, v in saved_mods.items():
                if v is None:
                    sys.modules.pop(k, None)
                else:
                    sys.modules[k] = v

        # 6. resolve_effective_style_tags fallback
        self.assertEqual(resolve_effective_style_tags([], "  dark Turkish rock  "), ["dark Turkish rock"])
        self.assertEqual(resolve_effective_style_tags(["synthwave", " ", "retrowave"], "ignored"), ["synthwave", "retrowave"])
        self.assertEqual(resolve_effective_style_tags([" "], "   "), [])

        # 7. Preview soft-peak scaling preserves waveform shape when peak > 0.99
        proj = storage.create_project("Peak Soft Clip Test", bpm=120.0, musical_key="A minor")
        hot_t = np.linspace(0.0, 0.25, n, endpoint=False, dtype=np.float32)
        hot_sine = np.stack(
            [1.5 * np.sin(2.0 * np.pi * 220.0 * hot_t), 1.5 * np.sin(2.0 * np.pi * 220.0 * hot_t)],
            axis=1,
        ).astype(np.float32)
        hot_rel = _unique_wav_rel("raw", "hot_mix", 1)
        storage.write_wav(proj.project_id, hot_rel, hot_sine, sample_rate=sr)
        prev_path, _ = storage.get_or_create_preview_audio(proj.project_id, hot_rel)
        decoded_hot, _ = storage.read_wav(
            proj.project_id,
            prev_path.relative_to(storage.project_dir(proj.project_id)).as_posix(),
            target_sr=sr,
        )
        # Soft-scaled sine matches hot_sine * (0.99 / 1.5) everywhere instead of flat clipping at 1.0
        expected_hot = hot_sine * (0.99 / float(np.max(np.abs(hot_sine))))
        self.assertLess(float(np.max(np.abs(decoded_hot - expected_hot))), 5e-4)

        # 8. llm_producer 25s timeout & Turkish conversational + English tag prompts
        sig_post = inspect.signature(llm_producer._http_post_json)
        self.assertEqual(sig_post.parameters["timeout"].default, 25.0)
        self.assertIn("English", llm_producer.SYSTEM_PROMPT_TAGS)
        self.assertIn("Turkish", llm_producer.SYSTEM_PROMPT_LYRICS)
        self.assertIn("English", llm_producer.SYSTEM_PROMPT)

        # 9. Lyrion_Colab_L4.ipynb contains #@param dropdown & all 4 L4 profiles
        nb_path = Path(__file__).resolve().parents[2] / "Lyrion_Colab_L4.ipynb"
        nb_data = json.loads(nb_path.read_text(encoding="utf-8"))
        cell2_src = "".join(nb_data["cells"][1]["source"])
        self.assertIn("#@param", cell2_src)
        for profile in ("fast_2b_1.7b", "balanced_2b_4b_lm", "xl_turbo_4b", "xl_sft_4b"):
            self.assertIn(profile, cell2_src)


if __name__ == "__main__":
    unittest.main()
