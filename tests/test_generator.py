"""
Tests for the generator module
"""

from unittest import mock

import pytest

# Mock the imports that are causing issues
mock.patch("ollama.chat", return_value={"message": {"content": ""}}).start()

from shorts_whisperer.transcriber import Segment, Transcript
from shorts_whisperer.generator import generate_title_description


@mock.patch("shorts_whisperer.generator.ollama")
def test_generate_title_description_success(mock_ollama):
    # Create a test transcript
    segments = [
        Segment(start=0.0, end=1.0, text="Hello"),
        Segment(start=1.0, end=2.0, text="world")
    ]
    transcript = Transcript(segments=segments, language="en")

    # Mock the Ollama response
    mock_ollama.chat.return_value = {
        "message": {
            "content": "TITLE: Hello World\nDESCRIPTION: A simple greeting to the world."
        }
    }

    # Call the function
    title, description = generate_title_description(transcript, model="test-model")

    # Verify
    assert mock_ollama.chat.called
    assert title == "Hello World"
    assert description == "A simple greeting to the world."


@mock.patch("shorts_whisperer.generator.ollama")
def test_generate_title_description_alternative_format(mock_ollama):
    # Create a test transcript
    segments = [
        Segment(start=0.0, end=1.0, text="Hello"),
        Segment(start=1.0, end=2.0, text="world")
    ]
    transcript = Transcript(segments=segments, language="en")

    # Mock the Ollama response with a different format
    mock_ollama.chat.return_value = {
        "message": {
            "content": "Hello World\n\nA simple greeting to the world."
        }
    }

    # Call the function
    title, description = generate_title_description(transcript, model="test-model")

    # Verify
    assert mock_ollama.chat.called
    assert title == "Hello World"
    assert description == "A simple greeting to the world."


@mock.patch("shorts_whisperer.generator.ollama")
def test_generate_title_description_error(mock_ollama):
    # Create a test transcript
    segments = [
        Segment(start=0.0, end=1.0, text="Hello"),
        Segment(start=1.0, end=2.0, text="world")
    ]
    transcript = Transcript(segments=segments, language="en")

    # Mock the Ollama response to raise an exception
    mock_ollama.chat.side_effect = Exception("API error")

    # Call the function
    title, description = generate_title_description(transcript, model="test-model")

    # Verify we get a fallback response
    assert title == "Video Transcript"
    assert description.startswith("This video contains the following content:")
    assert "Hello world" in description


@mock.patch("shorts_whisperer.generator.ollama")
def test_generate_title_description_with_full_transcript(mock_ollama):
    # Create a test clip transcript
    clip_segments = [
        Segment(start=0.0, end=1.0, text="Hello"),
        Segment(start=1.0, end=2.0, text="world")
    ]
    clip_transcript = Transcript(segments=clip_segments, language="en")

    # Create a test full episode transcript
    full_segments = [
        Segment(start=0.0, end=1.0, text="Hello"),
        Segment(start=1.0, end=2.0, text="world"),
        Segment(start=2.0, end=3.0, text="This"),
        Segment(start=3.0, end=4.0, text="is"),
        Segment(start=4.0, end=5.0, text="a"),
        Segment(start=5.0, end=6.0, text="full"),
        Segment(start=6.0, end=7.0, text="episode")
    ]
    full_transcript = Transcript(segments=full_segments, language="en")

    # Mock the Ollama response
    mock_ollama.chat.return_value = {
        "message": {
            "content": "# Hello World\n\nA simple greeting to the world with context from the full episode."
        }
    }

    # Call the function with both transcripts
    title, description = generate_title_description(
        clip_transcript,
        model="test-model",
        full_transcript=full_transcript,
        filename="test.mp4"
    )

    # Verify
    assert mock_ollama.chat.called
    assert title == "Hello World"
    assert description == "A simple greeting to the world with context from the full episode."

    # Verify that the prompt contains both transcripts
    call_args = mock_ollama.chat.call_args[1]
    messages = call_args["messages"]
    prompt = messages[0]["content"]

    assert "FULL EPISODE TRANSCRIPT" in prompt
    assert "SHORT CLIP TRANSCRIPT" in prompt
    assert "Hello world This is a full episode" in prompt.replace("\n", " ")
    assert "Hello world" in prompt