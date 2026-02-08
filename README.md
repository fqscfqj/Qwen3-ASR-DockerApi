# Qwen3-ASR Docker API

基于 **Qwen/Qwen3-ASR-1.7B** 的 OpenAI 兼容语音转写服务，提供 `/v1/audio/transcriptions` 接口。镜像默认在首次请求时自动下载模型，并支持 GPU/CPU 自动切换与空闲释放显存。

## 🚀 快速开始

### 使用预构建镜像

#### GPU（需要 NVIDIA Container Toolkit）

```bash
docker run --gpus all -p 8000:8000 \
  -e MODEL_DEVICE=auto \
  -e MODEL_IDLE_TIMEOUT=600 \
  ghcr.io/fqscfqj/qwen3-asr-dockerapi:latest
```

#### CPU

```bash
docker run -p 8000:8000 \
  -e MODEL_DEVICE=cpu \
  -e MODEL_IDLE_TIMEOUT=600 \
  ghcr.io/fqscfqj/qwen3-asr-dockerapi:latest
```

### Docker Compose

```bash
docker compose up
```

建议先拉取最新镜像：

```bash
docker compose pull
```

### 本地构建

```bash
docker build -t qwen3-asr-dockerapi .
```

### 本地运行（非容器）

```bash
pip install -r requirements.txt
uvicorn server:app --host 0.0.0.0 --port 8000
```

## 📡 接口示例

```bash
curl -X POST "http://localhost:8000/v1/audio/transcriptions" \
  -F file=@/path/to/audio.wav \
  -F model=qwen-asr \
  -F response_format=json
```

支持 OpenAI 的 `model`/`language`/`prompt` 字段，`model` 仅支持 `qwen-asr` 与 `whisper-1`。

## ⚙️ 环境变量

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `MODEL_ID` | `Qwen/Qwen3-ASR-1.7B` | Hugging Face 模型 ID |
| `MODEL_NAME` | `qwen-asr` | OpenAI 兼容的模型名 |
| `MODEL_CACHE_DIR` | `/models` | 模型缓存目录 |
| `MODEL_DEVICE` | `auto` | `auto`/`cuda`/`cpu` |
| `CORS_ORIGINS` | `*` | 允许的 CORS 来源，逗号分隔。默认 `*` 表示允许所有来源（若需允许带凭据请求，请设置为具体 origin 列表） |
| `MODEL_IDLE_TIMEOUT` | `600` | 空闲释放模型时间（秒） |
| `MAX_UPLOAD_MB` | `100` | 最大上传文件大小（MB） |
| `MAX_CONCURRENT_INFERENCES` | `1` | 并发推理数 |
| `PORT` | `8000` | 服务端口 |

## ✅ 健康检查

```bash
curl http://localhost:8000/health
```

返回示例：

```json
{"status":"ok","model_loaded":false}
```

`model_loaded` 表示当前进程是否已加载模型，首次推理完成后会变为 `true`，当空闲超时触发卸载时可能恢复为 `false`。

## 📄 License

MIT License，详见 [LICENSE](LICENSE)。
