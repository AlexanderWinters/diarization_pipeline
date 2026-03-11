# Whisper Transcription & Diarization Pipeline

A high-performance pipeline for audio transcription and speaker diarization using `faster-whisper` and `pyannote.audio`.

## Features
- **Fast Transcription**: Powered by `faster-whisper`.
- **Accurate Diarization**: Uses `pyannote.audio` for speaker identification.
- **Batch Processing**: Process entire directories of audio files in one go.
- **Configurable Models**: Easily switch between Whisper model sizes (`tiny`, `base`, `small`, etc.).

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. (Optional) Set up Hugging Face token for diarization:
   - Accept terms for [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
   - Accept terms for [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)
   - Create an access token at [hf.co/settings/tokens](https://huggingface.co/settings/tokens).

## Usage

```bash
# Standard usage with Hugging Face token
python pipeline/main.py --input path/to/audio --output results --model base --hf_token YOUR_TOKEN

# Local usage (if model is already cached or path to config.yaml is provided)
python pipeline/main.py --input path/to/audio --output results --diar_model /path/to/local/config.yaml
```

For more technical details, see [tech.md](tech.md).
