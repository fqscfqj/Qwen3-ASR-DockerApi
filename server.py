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
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from pydub import AudioSegment
from qwen_asr import Qwen3ASRModel
from starlette.concurrency import run_in_threadpool

WAV_SAMPLE_RATE = 16000
MODEL_ID = os.getenv("MODEL_ID", "Qwen/Qwen3-ASR-1.7B")
MODEL_NAME = os.getenv("MODEL_NAME")
MODEL_CACHE_DIR = os.getenv("MODEL_CACHE_DIR")
MODEL_IDLE_TIMEOUT = int(os.getenv("MODEL_IDLE_TIMEOUT", "600"))
MODEL_DEVICE = os.getenv("MODEL_DEVICE", "auto")
MODEL_ALIASES = {"whisper-1", "qwen-asr"}
if MODEL_NAME:
    MODEL_ALIASES.add(MODEL_NAME.strip().lower())
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "100"))
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
MAX_CONCURRENT_INFERENCES = max(1, int(os.getenv("MAX_CONCURRENT_INFERENCES", "1")))
INFERENCE_SEMAPHORE = threading.Semaphore(MAX_CONCURRENT_INFERENCES)
UPLOAD_CHUNK_SIZE = 1024 * 1024
API_KEY = os.getenv("API_KEY")

logger = logging.getLogger("server")


def get_segment_field(seg, field: str, default=None):
    """Extract field from segment, handling both dict and object types."""
    if isinstance(seg, dict):
        return seg.get(field, default)
    return getattr(seg, field, default)


def create_error_response(status_code: int, message: str, error_type: str, code: str):
    """Create a standardized error response."""
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "message": message,
                "type": error_type,
                "code": code,
            }
        },
    )


def format_timestamp_srt(seconds: float) -> str:
    """Convert seconds to SRT timestamp format (HH:mm:ss,ms)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def format_timestamp_vtt(seconds: float) -> str:
    """Convert seconds to WebVTT timestamp format (HH:mm:ss.ms)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def convert_to_srt(segments: list) -> str:
    """Convert segments to SRT format."""
    if not segments:
        return ""
    lines = []
    for i, seg in enumerate(segments, start=1):
        start = get_segment_field(seg, "start", 0)
        end = get_segment_field(seg, "end", 0)
        text = get_segment_field(seg, "text", "")
        text = text.strip()
        if text:
            lines.append(f"{i}\n{format_timestamp_srt(start)} --> {format_timestamp_srt(end)}\n{text}")
    return "\n\n".join(lines)


def convert_to_vtt(segments: list) -> str:
    """Convert segments to WebVTT format."""
    if not segments:
        return "WEBVTT\n\n"
    lines = ["WEBVTT\n"]
    for seg in segments:
        start = get_segment_field(seg, "start", 0)
        end = get_segment_field(seg, "end", 0)
        text = get_segment_field(seg, "text", "")
        text = text.strip()
        if text:
            lines.append(f"\n{format_timestamp_vtt(start)} --> {format_timestamp_vtt(end)}\n{text}\n")
    return "".join(lines)


def convert_to_text(segments: list) -> str:
    """Convert segments to plain text."""
    if not segments:
        return ""
    texts = []
    for seg in segments:
        text = get_segment_field(seg, "text", "")
        text = text.strip()
        if text:
            texts.append(text)
    return " ".join(texts)


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
            raise HTTPException(
                status_code=413,
                detail=f"Audio file is too large. Max size is {MAX_UPLOAD_MB} MB.",
            )
        chunks.append(chunk)
    return b"".join(chunks)


class ModelManager:
    def __init__(self, model_id: str, cache_dir: Optional[str], idle_timeout: int) -> None:
        self._model_id = model_id
        self._cache_dir = cache_dir
        self._idle_timeout = idle_timeout
        self._use_cuda = should_use_cuda()
        self._model = None
        self._last_used = 0.0
        self._active_requests = 0
        self._lock = threading.Lock()
        self._shutdown = threading.Event()
        self._cleanup_thread: Optional[threading.Thread] = None

    def _build_model(self):
        torch_dtype = torch.float16 if self._use_cuda else torch.float32
        device_map = "cuda:0" if self._use_cuda else "cpu"
        return Qwen3ASRModel.from_pretrained(
            self._model_id,
            cache_dir=self._cache_dir,
            dtype=torch_dtype,
            device_map=device_map,
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
            if self._model is None:
                try:
                    self._model = self._build_model()
                except Exception:
                    self._active_requests = max(0, self._active_requests - 1)
                    raise
            self._last_used = time.time()
            return self._model

    def release(self) -> None:
        with self._lock:
            self._active_requests = max(0, self._active_requests - 1)
            self._last_used = time.time()

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def _cleanup_loop(self) -> None:
        while not self._shutdown.is_set():
            time.sleep(5)
            if self._idle_timeout <= 0:
                continue
            with self._lock:
                if (
                    self._model is not None
                    and self._active_requests == 0
                    and (time.time() - self._last_used) >= self._idle_timeout
                ):
                    self._model = None
                    if self._use_cuda:
                        torch.cuda.empty_cache()
                    gc.collect()


model_manager = ModelManager(MODEL_ID, MODEL_CACHE_DIR, MODEL_IDLE_TIMEOUT)


def run_transcription(samples: np.ndarray, language: Optional[str] = None) -> object:
    INFERENCE_SEMAPHORE.acquire()
    try:
        model = model_manager.acquire()
        try:
            results = model.transcribe(
                audio=(samples, WAV_SAMPLE_RATE),
                language=language,
            )
            return results[0]
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

# CORS configuration (allow cross-origin requests from browsers)
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*")
if CORS_ORIGINS.strip() == "*":
    _cors_origins = ["*"]
    _cors_allow_credentials = False
else:
    _cors_origins = [o.strip() for o in CORS_ORIGINS.split(",") if o.strip()]
    _cors_allow_credentials = True
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=_cors_allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)


def require_api_key(
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    authorization: Optional[str] = Header(None, alias="Authorization"),
) -> None:
    if not API_KEY:
        return
    if x_api_key and x_api_key == API_KEY:
        return
    if authorization and authorization.startswith("Bearer "):
        token = authorization.removeprefix("Bearer ").strip()
        if token == API_KEY:
            return
    raise HTTPException(status_code=401, detail="Unauthorized")


@app.get("/health")
async def health(
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    authorization: Optional[str] = Header(None, alias="Authorization"),
):
    require_api_key(x_api_key=x_api_key, authorization=authorization)
    return {"status": "ok", "model_loaded": model_manager.is_loaded}


@app.get("/v1/models")
async def list_models(
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    authorization: Optional[str] = Header(None, alias="Authorization"),
):
    require_api_key(x_api_key=x_api_key, authorization=authorization)
    return {
        "object": "list",
        "data": [
            {
                "id": model_id,
                "object": "model",
                "owned_by": "qwen",
            }
            for model_id in sorted(MODEL_ALIASES)
        ],
    }


@app.post("/v1/audio/transcriptions")
async def transcriptions(
    file: UploadFile = File(...),
    model: Optional[str] = Form(None),
    language: Optional[str] = Form(None),
    prompt: Optional[str] = Form(None),
    response_format: str = Form("json"),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    authorization: Optional[str] = Header(None, alias="Authorization"),
):
    """OpenAI-compatible transcription endpoint (model/language/prompt accepted for compatibility)."""
    require_api_key(x_api_key=x_api_key, authorization=authorization)
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
        raise HTTPException(
            status_code=400,
            detail=(
                "Failed to decode audio. Supported formats: wav, mp3, m4a, flac, ogg, "
                "opus, webm, mp4, mkv, aac."
            ),
        ) from None
    try:
        result = await run_in_threadpool(run_transcription, samples, language)
    except Exception:
        logger.exception("ASR inference failed.")
        return create_error_response(
            status_code=500,
            message="ASR inference failed.",
            error_type="server_error",
            code="inference_error",
        )
    
    # Extract text and segments from result
    text = result.text if hasattr(result, "text") else str(result)
    detected_language = result.language if hasattr(result, "language") else language

    # Extract segments if available
    # Note: Checking both 'segments' (primary) and 'chunks' (potential alternative name) for robustness
    segments = []
    if hasattr(result, "segments") and result.segments:
        segments = result.segments
    elif hasattr(result, "chunks") and result.chunks:
        segments = result.chunks
    elif isinstance(result, dict):
        segments = result.get("segments", result.get("chunks", []))

    # Convert segments to serializable format (list of dicts)
    serializable_segments = []
    for seg in segments:
        seg_dict = {
            "start": get_segment_field(seg, "start", 0),
            "end": get_segment_field(seg, "end", 0),
            "text": get_segment_field(seg, "text", ""),
        }
        serializable_segments.append(seg_dict)

    # Validate that we have data
    if not text and not serializable_segments:
        return create_error_response(
            status_code=422,
            message="No transcription data was generated from the audio.",
            error_type="processing_error",
            code="empty_result",
        )

    # Handle different response formats
    if response_format == "json":
        return JSONResponse({"text": text})
    elif response_format == "text":
        # For text format, prefer segments if available, otherwise use text
        if serializable_segments:
            output_text = convert_to_text(serializable_segments)
        else:
            output_text = text
        return PlainTextResponse(output_text, media_type="text/plain")
    elif response_format == "verbose_json":
        return JSONResponse({
            "text": text,
            "language": detected_language or "unknown",
            "duration": duration,
            "segments": serializable_segments,
        })
    elif response_format == "srt":
        if not serializable_segments:
            return create_error_response(
                status_code=422,
                message="SRT format requires segment timing data, but the model did not generate segments for this audio.",
                error_type="processing_error",
                code="segments_unavailable",
            )
        srt_content = convert_to_srt(serializable_segments)
        return PlainTextResponse(srt_content, media_type="text/plain")
    elif response_format == "vtt":
        if not serializable_segments:
            return create_error_response(
                status_code=422,
                message="VTT format requires segment timing data, but the model did not generate segments for this audio.",
                error_type="processing_error",
                code="segments_unavailable",
            )
        vtt_content = convert_to_vtt(serializable_segments)
        return PlainTextResponse(vtt_content, media_type="text/plain")
    else:
        # Unsupported format
        return create_error_response(
            status_code=400,
            message=f"Unsupported response_format: {response_format}. Supported formats: json, verbose_json, text, srt, vtt.",
            error_type="invalid_request_error",
            code="unsupported_format",
        )


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))


if __name__ == "__main__":
    main()
