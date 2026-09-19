import os

# Keep CPU memory usage under control before importing CTranslate2.
os.environ["CT2_INTER_THREADS"] = "1"
os.environ["CT2_INTRA_THREADS"] = "1"

from pathlib import Path
from faster_whisper import WhisperModel


class ASREngine:
    def __init__(
        self,
        model_size: str = "tiny",
        device: str = "cpu",
        compute_type: str = "int8",
    ):
        print(f"Loading Whisper model: {model_size}")
        print(f"Device: {device}")
        print(f"Compute type: {compute_type}")

        self.model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
            cpu_threads=1,
            num_workers=1,
        )

        print("Whisper model loaded successfully.")

    def transcribe(self, audio_path: str) -> dict:
        audio_file = Path(audio_path)

        if not audio_file.exists():
            raise FileNotFoundError(
                f"Audio file not found: {audio_file}"
            )

        print(f"Transcribing: {audio_file}")

        segments, info = self.model.transcribe(
        str(audio_file),
        language="hi",
        beam_size=1,
        vad_filter=True,
        )   
        

        segment_list = []

        for segment in segments:
            segment_list.append({
                "start": segment.start,
                "end": segment.end,
                "text": segment.text.strip(),
            })

        full_text = " ".join(
            item["text"]
            for item in segment_list
            if item["text"]
        ).strip()

        return {
            "text": full_text,
            "language": info.language,
            "language_probability": info.language_probability,
            "segments": segment_list,
        }


if __name__ == "__main__":
    print("ASR module loaded successfully.")