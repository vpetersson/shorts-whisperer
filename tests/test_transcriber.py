"""
Tests for the transcriber module
"""

import os
import json
import tempfile
from pathlib import Path
from unittest import mock

import pytest

# Mock the imports that are causing issues
mock.patch("whisper.load_model", return_value=mock.MagicMock()).start()
mock.patch("moviepy.VideoFileClip", return_value=mock.MagicMock()).start()

from shorts_whisperer.transcriber import (
    Segment,
    Transcript,
    extract_audio,
    transcribe_video
)


class TestSegment:
    def test_segment_creation(self):
        segment = Segment(start=0.0, end=1.0, text="Hello world")
        assert segment.start == 0.0
        assert segment.end == 1.0
        assert segment.text == "Hello world"

    def test_segment_to_dict(self):
        segment = Segment(start=0.0, end=1.0, text="Hello world")
        data = segment.to_dict()
        assert data == {
            "start": 0.0,
            "end": 1.0,
            "text": "Hello world"
        }


class TestTranscript:
    def test_transcript_creation(self):
        transcript = Transcript()
        assert transcript.segments == []
        assert transcript.language == ""

    def test_transcript_with_segments(self):
        segments = [
            Segment(start=0.0, end=1.0, text="Hello"),
            Segment(start=1.0, end=2.0, text="world")
        ]
        transcript = Transcript(segments=segments, language="en")
        assert len(transcript.segments) == 2
        assert transcript.language == "en"

    def test_full_text(self):
        segments = [
            Segment(start=0.0, end=1.0, text="Hello"),
            Segment(start=1.0, end=2.0, text="world")
        ]
        transcript = Transcript(segments=segments)
        assert transcript.full_text == "Hello world"

    def test_to_dict(self):
        segments = [
            Segment(start=0.0, end=1.0, text="Hello"),
            Segment(start=1.0, end=2.0, text="world")
        ]
        transcript = Transcript(segments=segments, language="en")
        data = transcript.to_dict()
        assert data["language"] == "en"
        assert len(data["segments"]) == 2
        assert data["text"] == "Hello world"

    def test_save_and_load_json(self):
        segments = [
            Segment(start=0.0, end=1.0, text="Hello"),
            Segment(start=1.0, end=2.0, text="world")
        ]
        transcript = Transcript(segments=segments, language="en")

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as temp_file:
            temp_path = Path(temp_file.name)

        try:
            # Save to JSON
            transcript.save_json(temp_path)

            # Load from JSON
            loaded = Transcript.from_json(temp_path)

            # Verify
            assert loaded.language == "en"
            assert len(loaded.segments) == 2
            assert loaded.full_text == "Hello world"
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


@mock.patch("shorts_whisperer.transcriber.VideoFileClip")
def test_extract_audio(mock_video_clip):
    # Mock the video clip
    mock_video = mock.MagicMock()
    mock_audio = mock.MagicMock()
    mock_video.audio = mock_audio
    mock_video_clip.return_value = mock_video

    # Call the function
    result = extract_audio("test_video.mp4")

    # Verify
    assert mock_video_clip.called
    assert mock_audio.write_audiofile.called
    assert os.path.exists(result)

    # Clean up
    os.unlink(result)


@mock.patch("shorts_whisperer.transcriber.whisper")
@mock.patch("shorts_whisperer.transcriber.extract_audio")
def test_transcribe_video(mock_extract_audio, mock_whisper):
    # Mock the audio extraction
    mock_extract_audio.return_value = "test_audio.wav"

    # Mock the whisper model
    mock_model = mock.MagicMock()
    mock_whisper.load_model.return_value = mock_model

    # Mock the transcription result
    mock_model.transcribe.return_value = {
        "language": "en",
        "segments": [
            {"start": 0.0, "end": 1.0, "text": "Hello"},
            {"start": 1.0, "end": 2.0, "text": "world"}
        ]
    }

    # Call the function
    transcript = transcribe_video("test_video.mp4")

    # Verify
    assert mock_extract_audio.called
    assert mock_whisper.load_model.called
    assert mock_model.transcribe.called

    assert transcript.language == "en"
    assert len(transcript.segments) == 2
    assert transcript.full_text == "Hello world"