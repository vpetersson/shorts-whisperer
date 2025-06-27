"""
Title and description generator using Ollama
"""

import ollama
from typing import Tuple, Optional
import sys
import re
import logging

from shorts_whisperer.transcriber import Transcript

# Get logger
logger = logging.getLogger('shorts-whisperer')


def check_ollama_availability(model: str) -> bool:
    """
    Check if Ollama is running and the specified model is available.

    Args:
        model: The model name to check

    Returns:
        True if Ollama is available and model exists, False otherwise
    """
    try:
        # Try to list models to see if Ollama is running
        models = ollama.list()

        # Check if the specific model is available
        # Handle both old dict format and new object format
        if hasattr(models, 'models'):
            # New format: models is an object with .models attribute
            model_names = [m.model for m in models.models]
        else:
            # Old format: models is a dict with 'models' key
            model_names = [m['name'] for m in models.get('models', [])]

        # Handle both exact matches and base model names
        model_base = model.split(':')[0]  # Remove tag if present
        available = any(model in name or model_base in name for name in model_names)

        if not available:
            logger.error(f"Model '{model}' not found. Available models: {', '.join(model_names)}")
            return False

        logger.info(f"Ollama is running and model '{model}' is available")
        return True

    except ConnectionError:
        logger.error("Ollama is not running. Please start Ollama first.")
        return False
    except Exception as e:
        logger.error(f"Failed to connect to Ollama: {str(e)}")
        return False


def validate_output_format(title: str, description: str) -> Tuple[str, str, list]:
    """
    Basic validation to ensure output meets minimum quality standards.
    Returns: (cleaned_title, cleaned_description, issues_found)
    """
    issues = []

    # Clean up title
    title = title.strip()
    if len(title) > 100:
        issues.append(f"Title too long ({len(title)} chars) - YouTube recommends <100")
    if len(title) < 20:
        issues.append(f"Title too short ({len(title)} chars) - may not be descriptive enough")

    # Clean up description
    description = description.strip()
    if len(description) < 50:
        issues.append(f"Description too short ({len(description)} chars) - needs more context")
    if len(description) > 1000:
        issues.append(f"Description too long ({len(description)} chars) - may be cut off")

    return title, description, issues


def generate_title_description(
    transcript: Transcript,
    model: str = "llama3.2:latest",
    custom_prompt: Optional[str] = None,
    filename: Optional[str] = None,
    full_transcript: Optional[Transcript] = None
) -> Tuple[str, str]:
    """
    Generate a title and description based on a transcript using Ollama.

    Args:
        transcript: The transcript from the video clip to generate from
        model: The Ollama model to use
        custom_prompt: Optional custom prompt template
        filename: Optional filename for display purposes
        full_transcript: Optional full episode transcript for context

    Returns:
        A tuple of (title, description)
    """
    # Print the filename if provided
    if filename:
        logger.info(f"Processing file: {filename}")

    # Check if Ollama is available before proceeding
    if not check_ollama_availability(model):
        logger.error("Cannot proceed without Ollama. Exiting.")
        sys.exit(1)

    # Calculate transcript metrics for better context
    word_count = len(transcript.full_text.split())
    duration = max(segment.end for segment in transcript.segments) if transcript.segments else 0

    # Prepare the prompt
    if custom_prompt:
        # Use the custom prompt, replacing {transcript} with the actual transcript
        prompt = custom_prompt.replace("{transcript}", transcript.full_text)
    else:
        # Use the enhanced default prompt
        if full_transcript and full_transcript != transcript:
            # If we have both transcripts and they're different, use both
            prompt = f"""
                                    Create a title and description for a video clip based on this transcript.

            FULL EPISODE TRANSCRIPT (for context):
            {full_transcript.full_text}

            SHORT CLIP TRANSCRIPT (main content):
            {transcript.full_text}

            REQUIREMENTS:
            - Title: Summarize the main topic discussed in the clip (under 100 characters)
            - Description: Explain what is discussed in the clip (2-3 sentences)
            - Use only information that is actually mentioned in the transcript
            - Be accurate and direct

            FORMAT:
            # Title

            Description
            """
        else:
                        # If we only have one transcript, use it
            prompt = f"""
            Create a title and description for a video clip based on this transcript.

            TRANSCRIPT:
            {transcript.full_text}

            REQUIREMENTS:
            - Title: Summarize the main topic discussed in the clip (under 100 characters)
            - Description: Explain what is discussed in the clip (2-3 sentences)
            - Use only information that is actually mentioned in the transcript
            - Be accurate and direct

            FORMAT:
            # Title

            Description
            """

    # Call Ollama
    try:
        logger.info(f"Using Ollama model: {model}")
        logger.info("Sending prompt to Ollama...")

        response = ollama.chat(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        logger.info("Received response from Ollama")

        # Extract the response content
        content = response["message"]["content"]

        # Parse the response - simple split by markdown header
        parts = content.split("# ", 1)

        if len(parts) > 1:
            # Found markdown format
            title_and_rest = parts[1].strip()
            title_parts = title_and_rest.split("\n\n", 1)

            if len(title_parts) > 1:
                title = title_parts[0].strip()
                description = title_parts[1].strip()
                # Clean up description
                description = clean_description(description)
            else:
                # Try with single newline
                title_parts = title_and_rest.split("\n", 1)
                if len(title_parts) > 1:
                    title = title_parts[0].strip()
                    description = title_parts[1].strip()
                    # Clean up description
                    description = clean_description(description)
                else:
                    title = title_and_rest
                    description = "No description generated."
        else:
            # Fallback to old format parsing
            title_match = re.search(r'(?:^|\n)(?:TITLE|Title):\s*(.*?)(?:\n|$)', content, re.IGNORECASE)
            desc_match = re.search(r'(?:^|\n)(?:DESCRIPTION|Description):\s*(.*?)(?:\n\n|\Z)', content, re.IGNORECASE | re.DOTALL)

            if title_match:
                title = title_match.group(1).strip()
                if desc_match:
                    description = desc_match.group(1).strip()
                    # Clean up description
                    description = clean_description(description)
                else:
                    description = "No description generated."
            else:
                # Last resort: split by newlines
                lines = content.strip().split("\n")
                if len(lines) > 1:
                    title = lines[0].strip()
                    description = "\n".join(lines[1:]).strip()
                    # Clean up description
                    description = clean_description(description)
                else:
                    title = content.strip()
                    description = "No description generated."

        # Basic validation
        title, description, issues = validate_output_format(title, description)

        if issues:
            logger.warning("Output issues found:")
            for issue in issues:
                logger.warning(f"  - {issue}")

        # Final output
        logger.info(f"Final title: {title}")
        logger.info(f"Final description: {description[:100]}{'...' if len(description) > 100 else ''}")

        return title, description

    except Exception as e:
        logger.error(f"Error calling Ollama: {str(e)}")
        logger.error("Failed to generate title and description. Exiting.")
        sys.exit(1)

def clean_description(description: str) -> str:
    """Clean up the description by removing notes and explanations."""
    # Remove any notes or explanations after the description
    # Look for patterns like "(Note:" or empty lines followed by explanations
    note_patterns = [
        r'\n\s*\(Note:.*',
        r'\n\s*\n.*',
        r'\n\s*The description .*',
    ]

    for pattern in note_patterns:
        description = re.sub(pattern, '', description, flags=re.DOTALL)
    return description.strip()