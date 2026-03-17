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
# General installation
pip install -r requirements.txt

# For WSL/Linux with NVIDIA GPU (CUDA), ensure you have the correct version of torch:
# pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
```

2. (Optional) Set up Hugging Face token for diarization:
   - Accept terms for [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
   - Accept terms for [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)
   - Create an access token at [hf.co/settings/tokens](https://huggingface.co/settings/tokens).

## Usage

```bash
# Standard usage (will auto-detect GPU if available)
python pipeline/main.py --input path/to/audio --output results --hf_token YOUR_TOKEN

# Explicitly specify device and compute type
python pipeline/main.py --input path/to/audio --device cuda --compute_type float16
```

### Running on WSL (Windows Subsystem for Linux)
To use NVIDIA GPUs on WSL:
1. Ensure you have the [NVIDIA Windows Driver](https://www.nvidia.com/Download/index.aspx) installed on the host machine.
2. Install the [NVIDIA Container Toolkit](https://docs.nvidia.com/cuda/wsl-user-guide/index.html) if you plan to use Docker, or just ensure CUDA is visible in WSL by running `nvidia-smi` in the WSL terminal.
3. The pipeline will automatically detect the GPU if `torch.cuda.is_available()` is true.

For more technical details, see [tech.md](tech.md).
