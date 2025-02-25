"""
Command-line interface for shorts-whisperer
"""

import os
import sys
import click
from pathlib import Path
import json
import logging

from shorts_whisperer.transcriber import transcribe_video, Transcript, Segment
from shorts_whisperer.generator import generate_title_description


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger('shorts-whisperer')


@click.command()
@click.option(
    "--input",
    "-i",
    required=True,
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True),
    help="Path to the input video file",
)
@click.option(
    "--full-transcript",
    "-f",
    required=False,
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True),
    help="Path to the full episode transcript JSON for reference",
)
@click.option(
    "--output",
    "-o",
    required=False,
    type=click.Path(file_okay=True, dir_okay=False, writable=True),
    help="Path to save the generated title and description",
)
@click.option(
    "--model",
    "-m",
    default="llama3.1:latest",
    help="Ollama model to use for generating title and description",
)
@click.option(
    "--prompt-template",
    "-p",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True),
    help="Path to a custom prompt template file",
)
@click.option(
    "--transcript-format",
    "-t",
    type=click.Choice(["json", "txt", "srt", "vtt"]),
    default="json",
    help="Format for saving/loading the transcript",
)
@click.option(
    "--load-transcript",
    "-l",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True),
    help="Path to load an existing transcript instead of generating one",
)
@click.option(
    "--whisper-model",
    "-w",
    default="medium.en",
    type=click.Choice(["tiny", "base", "small", "medium", "large", "tiny.en", "base.en", "small.en", "medium.en", "large.en"]),
    help="Whisper model to use for transcription",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable verbose output",
)
def main(input, full_transcript, output, model, prompt_template, transcript_format, load_transcript, whisper_model, verbose):
    """
    Transcribe a video and generate a title and description based on the transcription.
    """
    # Set logging level based on verbose flag
    if not verbose:
        logger.setLevel(logging.WARNING)
        # Suppress other loggers
        logging.getLogger("moviepy").setLevel(logging.ERROR)
        logging.getLogger("whisper").setLevel(logging.ERROR)
        logging.getLogger("ollama").setLevel(logging.ERROR)

    logger.info(f"Processing video: {input}")

    # Variables to hold both transcripts
    video_transcript = None
    full_episode_transcript = None

    # Either load an existing transcript or generate a new one
    if load_transcript:
        logger.info(f"Loading video transcript from: {load_transcript}")
        transcript_path = Path(load_transcript)
        if transcript_format == "json":
            video_transcript = Transcript.from_json(transcript_path)
        else:
            # For future implementation of other formats
            logger.warning(f"Format {transcript_format} not yet supported for loading. Using JSON format.")
            video_transcript = Transcript.from_json(transcript_path)
    else:
        # Generate transcript from the video
        logger.info("Generating transcript from video...")
        video_transcript = transcribe_video(input, model_name=whisper_model)
        logger.info("Transcript generation complete.")

    # Load the full episode transcript if provided
    if full_transcript:
        full_transcript_path = Path(full_transcript)
        logger.info(f"Loading full episode transcript from: {full_transcript_path}")

        # Load the full transcript as plain text
        try:
            with open(full_transcript_path, 'r', encoding='utf-8') as f:
                full_text = f.read()

            # Create a simple transcript with the full text
            full_episode_transcript = Transcript()
            full_episode_transcript.segments = [Segment(start=0.0, end=1.0, text=full_text)]

        except Exception as e:
            logger.error(f"Error loading full transcript: {str(e)}")
            logger.warning("Continuing without full transcript reference.")
            full_episode_transcript = None

    # Load custom prompt template if provided
    custom_prompt = None
    if prompt_template:
        with open(prompt_template, 'r', encoding='utf-8') as f:
            custom_prompt = f.read()
        logger.info(f"Using custom prompt template from: {prompt_template}")

    # Generate title and description
    logger.info(f"Generating title and description using model: {model}")
    title, description = generate_title_description(
        video_transcript,
        model,
        custom_prompt,
        filename=os.path.basename(input),
        full_transcript=full_episode_transcript
    )

    # Output the results
    result = f"Title: {title}\n\nDescription:\n{description}"

    if output:
        output_path = Path(output)
        # Create the directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(result)
        logger.info(f"Title and description saved to: {output_path}")
    else:
        # Always print the result to stdout regardless of verbose setting
        click.echo("\n" + result)

    return 0


if __name__ == "__main__":
    sys.exit(main())