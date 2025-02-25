# shorts-whisperer

A tool to transcribe videos and generate titles and descriptions based on the transcription.

## Features

- Transcribe video files using Whisper
- Generate catchy titles and detailed descriptions using Ollama
- Save full transcripts for reference
- Support for custom prompt templates
- Flexible transcript format handling
- Simple command-line interface

## Installation

### Prerequisites

- Python 3.9 or higher
- [FFmpeg](https://ffmpeg.org/download.html) (required for audio extraction)
- [Ollama](https://ollama.ai/) running locally

### Install from source

```bash
# Clone the repository
git clone https://github.com/vpetersson/shorts-whisperer.git
cd shorts-whisperer

# Install with Poetry
poetry install

# Activate the virtual environment
poetry shell
```

## Usage

```bash
# Basic usage
shorts-whisperer --input /path/to/video.mp4

# Save the full transcript
shorts-whisperer --input /path/to/video.mp4 --full-transcript /path/to/transcript.json

# Save the generated title and description
shorts-whisperer --input /path/to/video.mp4 --output /path/to/output.txt

# Use a different Ollama model
shorts-whisperer --input /path/to/video.mp4 --model llama3:8b

# Use a custom prompt template
shorts-whisperer --input /path/to/video.mp4 --prompt-template /path/to/prompt.txt

# Load an existing transcript instead of generating a new one
shorts-whisperer --input /path/to/video.mp4 --load-transcript /path/to/transcript.json
```

### Options

- `--input`, `-i`: Path to the input video file (required)
- `--full-transcript`, `-f`: Path to save the full transcript JSON
- `--output`, `-o`: Path to save the generated title and description
- `--model`, `-m`: Ollama model to use for generating title and description (default: llama3)
- `--prompt-template`, `-p`: Path to a custom prompt template file
- `--transcript-format`, `-t`: Format for saving/loading the transcript (json, txt, srt, vtt)
- `--load-transcript`, `-l`: Path to load an existing transcript instead of generating a new one

### Custom Prompt Templates

You can create your own prompt templates to customize how titles and descriptions are generated. Create a text file with your prompt and use the `{transcript}` placeholder where you want the transcript text to be inserted.

Example custom prompt template:

```
You are a YouTube content creator. Based on this video transcript, create an engaging title and description that will maximize views and engagement.

TRANSCRIPT:
{transcript}

The title should be attention-grabbing and under 60 characters.
The description should include relevant keywords and be between 200-300 characters.

Format:
TITLE: [title here]
DESCRIPTION: [description here]
```

## Development

### Running Tests

```bash
# Run tests
poetry run pytest

# Run tests with coverage
poetry run pytest --cov=shorts_whisperer
```

### Project Structure

- `src/shorts_whisperer/`: Main package
  - `cli.py`: Command-line interface
  - `transcriber.py`: Video transcription using Whisper
  - `generator.py`: Title and description generation using Ollama
- `tests/`: Test suite

## License

MIT

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
