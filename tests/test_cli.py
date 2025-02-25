"""
Tests for the CLI module
"""

import os
import tempfile
from pathlib import Path
from unittest import mock

import pytest
from click.testing import CliRunner

# Mock the imports that are causing issues
mock.patch("shorts_whisperer.transcriber.whisper", return_value=mock.MagicMock()).start()
mock.patch("shorts_whisperer.transcriber.VideoFileClip", return_value=mock.MagicMock()).start()
mock.patch("shorts_whisperer.generator.ollama", return_value=mock.MagicMock()).start()

from shorts_whisperer.cli import main
from shorts_whisperer.transcriber import Transcript, Segment


@pytest.fixture
def mock_transcribe():
    with mock.patch("shorts_whisperer.cli.transcribe_video") as mock_func:
        # Create a test transcript
        segments = [
            Segment(start=0.0, end=1.0, text="Hello"),
            Segment(start=1.0, end=2.0, text="world")
        ]
        transcript = Transcript(segments=segments, language="en")
        mock_func.return_value = transcript
        yield mock_func


@pytest.fixture
def mock_generate():
    with mock.patch("shorts_whisperer.cli.generate_title_description") as mock_func:
        mock_func.return_value = ("Test Title", "Test Description")
        yield mock_func


def test_cli_basic(mock_transcribe, mock_generate):
    runner = CliRunner()
    with tempfile.NamedTemporaryFile(suffix=".mp4") as temp_file:
        result = runner.invoke(main, ["--input", temp_file.name])

        assert result.exit_code == 0
        assert mock_transcribe.called
        assert mock_generate.called
        assert "Test Title" in result.output
        assert "Test Description" in result.output


def test_cli_with_output(mock_transcribe, mock_generate):
    runner = CliRunner()
    with tempfile.NamedTemporaryFile(suffix=".mp4") as video_file, \
         tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as output_file:
        output_path = Path(output_file.name)

        try:
            result = runner.invoke(main, [
                "--input", video_file.name,
                "--output", output_path
            ])

            assert result.exit_code == 0
            assert mock_transcribe.called
            assert mock_generate.called
            assert os.path.exists(output_path)

            content = output_path.read_text()
            assert "Test Title" in content
            assert "Test Description" in content
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)


def test_cli_with_transcript(mock_transcribe, mock_generate):
    runner = CliRunner()
    with tempfile.NamedTemporaryFile(suffix=".mp4") as video_file, \
         tempfile.NamedTemporaryFile(suffix=".json", delete=False) as transcript_file:
        transcript_path = Path(transcript_file.name)

        # Write valid JSON to the transcript file
        transcript_file.write(b'{"language": "en", "segments": [], "text": ""}')
        transcript_file.flush()

        try:
            result = runner.invoke(main, [
                "--input", video_file.name,
                "--full-transcript", transcript_path
            ])

            assert result.exit_code == 0
            assert mock_transcribe.called
            assert mock_generate.called
            assert os.path.exists(transcript_path)
        finally:
            if os.path.exists(transcript_path):
                os.unlink(transcript_path)


def test_cli_with_custom_model(mock_transcribe, mock_generate):
    runner = CliRunner()
    with tempfile.NamedTemporaryFile(suffix=".mp4") as temp_file:
        result = runner.invoke(main, [
            "--input", temp_file.name,
            "--model", "custom-model"
        ])

        assert result.exit_code == 0
        assert mock_transcribe.called
        # Updated assertion to match the new function signature
        mock_generate.assert_called_with(
            mock_transcribe.return_value,
            "custom-model",
            None,
            filename=os.path.basename(temp_file.name),
            full_transcript=None
        )
        assert "Test Title" in result.output
        assert "Test Description" in result.output