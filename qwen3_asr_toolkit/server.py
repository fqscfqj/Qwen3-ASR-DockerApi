import gc
import logging
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
MODEL_NAME = os.getenv("MODEL_NAME", "qwen3-asr-1.7b")
MODEL_CACHE_DIR = os.getenv("MODEL_CACHE_DIR")
MODEL_IDLE_TIMEOUT = int(os.getenv("MODEL_IDLE_TIMEOUT", "600"))
MODEL_DEVICE = os.getenv("MODEL_DEVICE", "auto")
MODEL_ALIASES = {MODEL_ID.lower(), MODEL_NAME.lower(), "whisper-1"}
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "100"))
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
MAX_CONCURRENT_INFERENCES = max(1, int(os.getenv("MAX_CONCURRENT_INFERENCES", "1")))
INFERENCE_SEMAPHORE = threading.Semaphore(MAX_CONCURRENT_INFERENCES)
UPLOAD_CHUNK_SIZE = 1024 * 1024

logger = logging.getLogger("qwen3_asr_toolkit.server")


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


async def read_upload(file: UploadFile) -> bytes:
    size = 0
    chunks = []
    while True:
        chunk = await file.read(UPLOAD_CHUNK_SIZE)
        if not chunk:
            break
        size += len(chunk)
        if size > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="Audio file is too large.")
        chunks.append(chunk)
    return b"".join(chunks)


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
                try:
                    self._pipeline = self._build_pipeline()
                except Exception:
                    self._active_requests = max(0, self._active_requests - 1)
                    raise
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
    INFERENCE_SEMAPHORE.acquire()
    try:
        pipe = model_manager.acquire()
        try:
            return pipe(samples, sampling_rate=WAV_SAMPLE_RATE)
        finally:
            model_manager.release()
    finally:
        INFERENCE_SEMAPHORE.release()


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
    """OpenAI-compatible transcription endpoint (model/language/prompt accepted for compatibility)."""
    if model and model.lower() not in MODEL_ALIASES:
        raise HTTPException(status_code=400, detail="Unsupported model.")
    if not file:
        raise HTTPException(status_code=400, detail="Missing audio file.")
    data = await read_upload(file)
    if not data:
        raise HTTPException(status_code=400, detail="Empty audio file.")
    try:
        samples, duration = await run_in_threadpool(decode_audio, data, file.filename)
    except Exception:
        logger.exception("Failed to decode audio upload.")
        raise HTTPException(status_code=400, detail="Failed to decode audio.") from None
    try:
        result = await run_in_threadpool(run_transcription, samples)
    except Exception:
        logger.exception("ASR inference failed.")
        raise HTTPException(status_code=500, detail="ASR inference failed.") from None
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
