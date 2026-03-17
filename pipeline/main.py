import os
import torch
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline
from typing import List, Dict, Any, Optional
import json
from tqdm import tqdm
from pathlib import Path

class TranscriptionDiarizationPipeline:
    def __init__(
        self,
        whisper_model_name: str = "base",
        hf_token: Optional[str] = None,
        diarization_model: str = "pyannote/speaker-diarization-3.1",
        device: Optional[str] = None,
        compute_type: Optional[str] = None
    ):
        """
        Initialize the transcription and diarization pipeline.
        
        Args:
            whisper_model_name (str): Whisper model to use (e.g., 'base', 'small', 'medium', 'large-v3').
            hf_token (str, optional): Hugging Face token for pyannote.audio models.
            diarization_model (str): Path to local config.yaml or HF repo ID.
            device (str, optional): Device to use for computation ('cpu' or 'cuda').
            compute_type (str, optional): Type of computation ('float16', 'int8', 'float32').
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        if compute_type is None:
            self.compute_type = "float16" if self.device == "cuda" else "float32"
        else:
            self.compute_type = compute_type
            
        self.whisper_model_name = whisper_model_name
        self.hf_token = hf_token
        self.diarization_model = diarization_model
        
        # Load Whisper model
        print(f"Loading Whisper model: {whisper_model_name} on {self.device} with {self.compute_type}...")
        self.whisper_model = WhisperModel(whisper_model_name, device=self.device, compute_type=self.compute_type)
        
        # Load Diarization model
        self.diarization_pipeline = None
        print(f"Loading Diarization pipeline: {diarization_model} on {self.device}...")
        try:
            # Try to load without token first (in case of local path or cached model)
            self.diarization_pipeline = Pipeline.from_pretrained(
                diarization_model,
                token=hf_token
            )
            
            if self.diarization_pipeline:
                self.diarization_pipeline.to(torch.device(self.device))
                print("Diarization pipeline loaded successfully.")
            else:
                print("Warning: Diarization pipeline failed to load. Diarization will be skipped.")
        except Exception as e:
            print(f"Error loading Diarization pipeline: {e}")
            print("Diarization will be skipped.")

    def transcribe(self, audio_path: str) -> List[Dict[str, Any]]:
        """Transcribe an audio file using Whisper."""
        segments, info = self.whisper_model.transcribe(audio_path, beam_size=5)
        transcribed_segments = []
        for segment in segments:
            transcribed_segments.append({
                "start": segment.start,
                "end": segment.end,
                "text": segment.text.strip()
            })
        return transcribed_segments

    def diarize(self, audio_path: str) -> List[Dict[str, Any]]:
        """Perform speaker diarization on an audio file."""
        if not self.diarization_pipeline:
            return []
        
        diarization_output = self.diarization_pipeline(audio_path)
        diarization = getattr(diarization_output, "speaker_diarization", diarization_output)

        diarization_segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            diarization_segments.append({
                "start": turn.start,
                "end": turn.end,
                "speaker": speaker
            })
        return diarization_segments

    def process(self, audio_path: str) -> List[Dict[str, Any]]:
        """Run the full pipeline: transcribe, diarize, and align."""
        print(f"Processing: {audio_path}")
        
        # 1. Transcription
        transcribed_segments = self.transcribe(audio_path)
        
        # 2. Diarization
        diarization_segments = self.diarize(audio_path)
        
        # 3. Alignment
        if not diarization_segments:
            return transcribed_segments
            
        final_segments = []
        for segment in transcribed_segments:
            # Simple alignment: find the speaker that overlaps the most with the transcription segment
            segment_start = segment["start"]
            segment_end = segment["end"]
            
            best_speaker = "UNKNOWN"
            max_overlap = 0
            
            for dia_seg in diarization_segments:
                overlap = min(segment_end, dia_seg["end"]) - max(segment_start, dia_seg["start"])
                if overlap > max_overlap:
                    max_overlap = overlap
                    best_speaker = dia_seg["speaker"]
            
            segment["speaker"] = best_speaker
            final_segments.append(segment)
            
        return final_segments

    def process_batch(self, input_dir: str, output_dir: str, extensions: List[str] = [".wav", ".mp3", ".m4a"]):
        """Process a directory of audio files."""
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        audio_files = [f for f in input_path.iterdir() if f.suffix.lower() in extensions]
        print(f"Found {len(audio_files)} audio files in {input_dir}")
        
        for audio_file in tqdm(audio_files, desc="Batch Processing"):
            try:
                result = self.process(str(audio_file))
                
                output_file = output_path / f"{audio_file.stem}_result.json"
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(result, f, ensure_ascii=False, indent=4)
                
                # Also save as readable TXT
                txt_output_file = output_path / f"{audio_file.stem}_result.txt"
                with open(txt_output_file, 'w', encoding='utf-8') as f:
                    for seg in result:
                        f.write(f"[{seg['start']:>7.2f}s - {seg['end']:>7.2f}s] {seg.get('speaker', 'N/A')}: {seg['text']}\n")
                        
            except Exception as e:
                print(f"Error processing {audio_file.name}: {e}")

if __name__ == "__main__":
    # Example usage
    import argparse
    parser = argparse.ArgumentParser(description="Whisper Transcription & Diarization Pipeline")
    parser.add_argument("--input", type=str, required=True, help="Input audio file or directory")
    parser.add_argument("--output", type=str, default="output", help="Output directory")
    parser.add_argument("--model", type=str, default="base", help="Whisper model (tiny, base, small, medium, large-v3)")
    parser.add_argument("--diar_model", type=str, default="pyannote/speaker-diarization-3.1", help="Diarization model (HF repo ID or local path to config.yaml)")
    parser.add_argument("--hf_token", type=str, help="Hugging Face token for diarization")
    parser.add_argument("--device", type=str, help="Compute device (cpu, cuda). Defaults to cuda if available.")
    parser.add_argument("--compute_type", type=str, help="Compute type (float16, int8, float32). Defaults to float16 on cuda, float32 on cpu.")
    
    args = parser.parse_args()
    
    # Check if input is a file or a directory
    pipeline = TranscriptionDiarizationPipeline(
        whisper_model_name=args.model,
        diarization_model=args.diar_model,
        hf_token=args.hf_token,
        device=args.device,
        compute_type=args.compute_type
    )
    
    if os.path.isdir(args.input):
        pipeline.process_batch(args.input, args.output)
    else:
        result = pipeline.process(args.input)
        os.makedirs(args.output, exist_ok=True)
        output_file = Path(args.output) / f"{Path(args.input).stem}_result.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=4)
        print(f"Result saved to {output_file}")
