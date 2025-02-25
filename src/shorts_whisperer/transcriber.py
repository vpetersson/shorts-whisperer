"""
Video transcription module using Whisper
"""

import os
import json
import tempfile
import warnings
import contextlib
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, IO

import whisper
from moviepy import VideoFileClip


# Suppress specific warnings
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")


@dataclass
class Segment:
    """A segment of transcribed text with timing information."""
    start: float
    end: float
    text: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert segment to dictionary."""
        return {
            "start": self.start,
            "end": self.end,
            "text": self.text
        }


@dataclass
class Transcript:
    """A complete transcript with segments and metadata."""
    segments: List[Segment] = field(default_factory=list)
    language: str = ""

    @property
    def full_text(self) -> str:
        """Get the full transcript text."""
        return " ".join(segment.text for segment in self.segments)

    def to_dict(self) -> Dict[str, Any]:
        """Convert transcript to dictionary."""
        return {
            "language": self.language,
            "segments": [segment.to_dict() for segment in self.segments],
            "text": self.full_text
        }

    def save_json(self, path: Path) -> None:
        """Save transcript to JSON file."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Transcript":
        """Create transcript from dictionary."""
        transcript = cls(language=data.get("language", ""))
        for segment_data in data.get("segments", []):
            transcript.segments.append(
                Segment(
                    start=segment_data["start"],
                    end=segment_data["end"],
                    text=segment_data["text"]
                )
            )
        return transcript

    @classmethod
    def from_json(cls, path: Path) -> "Transcript":
        """Load transcript from JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Check if the data is a list (alternative format)
        if isinstance(data, list):
            transcript = cls(language="en")  # Default to English

            # Convert the alternative format to segments
            for item in data:
                # Parse the start time (format: "00:00")
                start_time_str = item.get("startTime", "00:00")
                time_parts = start_time_str.split(":")
                start_time = float(time_parts[0]) * 60 + float(time_parts[1])

                # Use a default end time 5 seconds after start time
                end_time = start_time + 5.0

                # Get the text
                text = item.get("sentence", "")

                # Add the segment
                if text:
                    transcript.segments.append(
                        Segment(
                            start=start_time,
                            end=end_time,
                            text=text
                        )
                    )

            return transcript
        else:
            # Original format
            return cls.from_dict(data)


@contextlib.contextmanager
def suppress_stdout_stderr():
    """Context manager to suppress stdout and stderr."""
    with open(os.devnull, "w") as devnull:
        old_stdout, old_stderr = os.dup(1), os.dup(2)
        os.dup2(devnull.fileno(), 1)
        os.dup2(devnull.fileno(), 2)
        try:
            yield
        finally:
            os.dup2(old_stdout, 1)
            os.dup2(old_stderr, 2)


def extract_audio(video_path: str) -> str:
    """Extract audio from video file."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
        temp_path = temp_file.name

    # Suppress output during audio extraction
    with suppress_stdout_stderr():
        video = VideoFileClip(video_path)
        video.audio.write_audiofile(temp_path, logger=None)
        video.close()

    return temp_path


def transcribe_video(video_path: str, model_name: str = "base") -> Transcript:
    """
    Transcribe a video file using Whisper.

    Args:
        video_path: Path to the video file
        model_name: Whisper model name (tiny, base, small, medium, large)

    Returns:
        A Transcript object with the transcription results
    """
    # Extract audio from video
    audio_path = extract_audio(video_path)

    try:
        # Load Whisper model and transcribe with suppressed output
        with suppress_stdout_stderr():
            # Load Whisper model
            model = whisper.load_model(model_name)

            # Transcribe audio
            result = model.transcribe(audio_path)

        # Create transcript
        transcript = Transcript(language=result.get("language", ""))

        for segment in result.get("segments", []):
            transcript.segments.append(
                Segment(
                    start=segment["start"],
                    end=segment["end"],
                    text=segment["text"].strip()
                )
            )

        return transcript

    finally:
        # Clean up temporary audio file
        if os.path.exists(audio_path):
            os.unlink(audio_path)