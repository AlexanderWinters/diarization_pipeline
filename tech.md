### Whisper Transcription & Diarization Pipeline Technical Document

#### Introduction
This document outlines the architecture and implementation details for a Whisper-based transcription and diarization pipeline, designed for efficiency, flexibility, and batch processing.

#### Core Components
- **Transcription**: Utilizing `faster-whisper` for high-performance audio-to-text conversion.
- **Diarization**: Leveraging `pyannote.audio` for speaker identification and segmentation.
- **Batch Processing**: A pipeline designed to handle multiple audio files concurrently or sequentially, with configurable models and parameters.

#### Implementation Strategy
1. **Module: Transcription**
   - Supports various Whisper models (tiny, base, small, medium, large-v2, large-v3).
   - Utilizes `faster-whisper` for optimized inference on CPU and GPU.
2. **Module: Diarization**
   - Integrates `pyannote.audio`'s pre-trained diarization models.
   - **Local Execution**: Supports loading models from local paths (e.g., a `config.yaml` file) to bypass mandatory Hugging Face downloads if the model is already available locally.
   - Requires an HF (Hugging Face) access token for the first download of restricted models.
3. **Module: Alignment**
   - Aligns diarization segments with transcription segments to assign speaker IDs to transcribed text.
4. **Batch Processing**
   - Scans a directory for audio files (e.g., .wav, .mp3, .m4a) and processes them through the pipeline.
   - Results are saved in structured formats (e.g., JSON, TXT).

#### Usage & Configuration
The pipeline will be configurable through a Python script or CLI, allowing users to specify:
- Input directory
- Output directory
- Whisper model type
- Hugging Face token for diarization
- Compute device (cpu, cuda)

#### Detailed Implementation Details

##### Transcription Module
We use `faster-whisper`, which is a reimplementation of OpenAI's Whisper model using CTranslate2. It is significantly faster and uses less memory than the original implementation.
- **Model Selection**: Users can choose from `tiny`, `base`, `small`, `medium`, `large-v2`, `large-v3`.
- **Inference**: Supports both CPU and GPU (CUDA).

##### Diarization Module
We use `pyannote.audio`, specifically the `speaker-diarization-3.1` model. 
- **Requirements**: Accessing this model typically requires accepting terms on Hugging Face for `pyannote/speaker-diarization-3.1` and `pyannote/segmentation-3.0`.
- **Local Loading**: The pipeline can load models locally by passing a path to a `config.yaml` file to the `--diar_model` argument. This allows for fully offline operation once models are downloaded.
- **Alignment**: Since transcription and diarization produce separate segments, we align them by calculating the temporal overlap. Each transcribed segment is assigned the speaker label that has the maximum overlap with its time interval.

##### Batch Processing
The `process_batch` method in `TranscriptionDiarizationPipeline` handles multiple files.
- It iterates through all supported audio files in the input directory.
- For each file, it generates a JSON result containing the transcript with speaker labels and a human-readable TXT file.

#### Example Commands
```bash
# Process a single file using a specific Whisper model
python pipeline/main.py --input audio.mp3 --output results --model small

# Process a directory (Batch) with speaker diarization
python pipeline/main.py --input ./audio_dir --output ./results --hf_token YOUR_HF_TOKEN

# Run diarization using a local model configuration (Fully Local/Offline)
python pipeline/main.py --input audio.mp3 --output results --diar_model /path/to/local/config.yaml
```
