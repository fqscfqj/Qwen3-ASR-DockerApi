import gc
import os
import tempfile
import threading
import time
from contextlib import asynccontextmanager
from typing import Optional, Tuple

import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse
from pydub import AudioSegment
from starlette.concurrency import run_in_threadpool
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

WAV_SAMPLE_RATE = 16000
MODEL_ID = os.getenv("MODEL_ID", "Qwen/Qwen3-ASR-1.7B")
MODEL_CACHE_DIR = os.getenv("MODEL_CACHE_DIR")
MODEL_IDLE_TIMEOUT = int(os.getenv("MODEL_IDLE_TIMEOUT", "600"))
MODEL_DEVICE = os.getenv("MODEL_DEVICE", "auto")
MODEL_ALIASES = {MODEL_ID, "whisper-1"}


def should_use_cuda() -> bool:
    if MODEL_DEVICE.lower() == "cuda":
        return True
    if MODEL_DEVICE.lower() == "cpu":
        return False
    return torch.cuda.is_available()


def decode_audio(data: bytes, filename: Optional[str]) -> Tuple[np.ndarray, float]:
    allowed_suffixes = {
        ".wav",
        ".mp3",
        ".m4a",
        ".flac",
        ".ogg",
        ".opus",
        ".webm",
        ".mp4",
        ".mkv",
        ".aac",
    }
    suffix = os.path.splitext(filename or "")[1].lower()
    if suffix not in allowed_suffixes:
        suffix = ".tmp"
    with tempfile.NamedTemporaryFile(suffix=suffix) as tmp_file:
        tmp_file.write(data)
        tmp_file.flush()
        segment = AudioSegment.from_file(tmp_file.name)
    segment = segment.set_frame_rate(WAV_SAMPLE_RATE).set_channels(1)
    samples = np.array(segment.get_array_of_samples()).astype(np.float32)
    max_value = float(1 << (8 * segment.sample_width - 1))
    if max_value:
        samples /= max_value
    duration = len(segment) / 1000.0
    return samples, duration


class ModelManager:
    def __init__(self, model_id: str, cache_dir: Optional[str], idle_timeout: int) -> None:
        self._model_id = model_id
        self._cache_dir = cache_dir
        self._idle_timeout = idle_timeout
        self._use_cuda = should_use_cuda()
        self._pipeline = None
        self._last_used = 0.0
        self._active_requests = 0
        self._lock = threading.Lock()
        self._shutdown = threading.Event()
        self._cleanup_thread: Optional[threading.Thread] = None

    def _build_pipeline(self):
        torch_dtype = torch.float16 if self._use_cuda else torch.float32
        processor = AutoProcessor.from_pretrained(self._model_id, cache_dir=self._cache_dir)
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self._model_id,
            cache_dir=self._cache_dir,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
        )
        if self._use_cuda:
            model.to("cuda")
        return pipeline(
            task="automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            device=0 if self._use_cuda else -1,
        )

    def start_cleanup(self) -> None:
        if self._cleanup_thread is None:
            self._cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
            self._cleanup_thread.start()

    def shutdown(self) -> None:
        self._shutdown.set()
        if self._cleanup_thread:
            self._cleanup_thread.join(timeout=5)

    def acquire(self):
        with self._lock:
            self._active_requests += 1
            if self._pipeline is None:
                self._pipeline = self._build_pipeline()
            self._last_used = time.time()
            return self._pipeline

    def release(self) -> None:
        with self._lock:
            self._active_requests = max(0, self._active_requests - 1)
            self._last_used = time.time()

    @property
    def is_loaded(self) -> bool:
        return self._pipeline is not None

    def _cleanup_loop(self) -> None:
        while not self._shutdown.is_set():
            time.sleep(5)
            if self._idle_timeout <= 0:
                continue
            with self._lock:
                if (
                    self._pipeline is not None
                    and self._active_requests == 0
                    and (time.time() - self._last_used) >= self._idle_timeout
                ):
                    self._pipeline = None
                    if self._use_cuda:
                        torch.cuda.empty_cache()
                    gc.collect()


model_manager = ModelManager(MODEL_ID, MODEL_CACHE_DIR, MODEL_IDLE_TIMEOUT)


def run_transcription(samples: np.ndarray) -> dict:
    pipe = model_manager.acquire()
    try:
        return pipe(samples, sampling_rate=WAV_SAMPLE_RATE)
    finally:
        model_manager.release()


@asynccontextmanager
async def lifespan(_: FastAPI):
    model_manager.start_cleanup()
    yield
    model_manager.shutdown()


app = FastAPI(title="Qwen3-ASR OpenAI-Compatible API", lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": model_manager.is_loaded}


@app.post("/v1/audio/transcriptions")
async def transcriptions(
    file: UploadFile = File(...),
    model: Optional[str] = Form(None),
    language: Optional[str] = Form(None),
    prompt: Optional[str] = Form(None),
    response_format: str = Form("json"),
):
    if model and model not in MODEL_ALIASES:
        raise HTTPException(status_code=400, detail=f"Unsupported model '{model}'.")
    if not file:
        raise HTTPException(status_code=400, detail="Missing audio file.")
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty audio file.")
    try:
        samples, duration = await run_in_threadpool(decode_audio, data, file.filename)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to decode audio: {exc}") from exc
    try:
        result = await run_in_threadpool(run_transcription, samples)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"ASR inference failed: {exc}") from exc
    text = result.get("text") if isinstance(result, dict) else str(result)
    if response_format == "text":
        return PlainTextResponse(text)
    if response_format == "verbose_json":
        return JSONResponse({"text": text, "language": language or "unknown", "duration": duration})
    if response_format != "json":
        raise HTTPException(status_code=400, detail="Unsupported response_format.")
    return {"text": text}


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))


if __name__ == "__main__":
    main()
