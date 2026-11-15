# Lyrion — Zero-Cost Open-Source AI Music Studio

**Lyrion** is a split-architecture AI Music Studio (Suno + multi-track DAW hybrid) engineered to run at **$0 cost**:
- **Frontend (`/frontend`)**: Local or web React + TypeScript + Vite + Tailwind DAW using `Tone.js` (`Tone.Transport` with strict `Promise.all` buffer preloading) and `Wavesurfer.js` v7 (beat-snapped waveform region selection).
- **Backend (`/colab-backend`)**: FastAPI server designed for a free Google Colab L4 GPU (24GB VRAM, `bfloat16`), with strict `ModelManager` VRAM mutual exclusion (`del model; gc.collect(); torch.cuda.empty_cache()`), persistent Google Drive project storage, Zero-VRAM AI Producer (`gemini-3.8-flash` / `gemini-3.5-flash-lite` + Groq `llama-3.3-70b-versatile` fallback), ACE-Step 1.5, Demucs (`htdemucs`), CPU `librosa` beat grid analysis, sample-accurate equal-power crossfade repainting, and Spotify `pedalboard` + `ffmpeg` mastering export.

## Quickstart: Local Development (`MOCK_GPU=true`)

Run the backend locally without a GPU using synthetic multi-stem synthesis (`MOCK_GPU=true`):

```bash
cd colab-backend
cp .env.example .env
uvicorn app.main:app --reload --port 8000
```

Then start the React DAW frontend:

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`. The top bar connects to `http://localhost:8000` by default (stored in `localStorage`).

## Quickstart: Google Colab L4 GPU

1. Upload or clone this repository into your Google Drive or Colab runtime.
2. Run `colab-backend/scripts/colab_bootstrap.py` in a Colab L4 cell (ensure Runtime -> Change runtime type -> **L4 GPU**).
3. Copy the printed `https://*.trycloudflare.com` URL and paste it into the **GPU Server URL** input at the top of the Lyrion DAW frontend.
4. All projects, `project.json` manifests, raw mixes, stems, and exports are saved directly to `/content/drive/MyDrive/AIMusicStudio`.
