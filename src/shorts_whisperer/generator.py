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


def generate_title_description(
    transcript: Transcript,
    model: str = "llama3.1:latest",
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

    # Prepare the prompt
    if custom_prompt:
        # Use the custom prompt, replacing {transcript} with the actual transcript
        prompt = custom_prompt.replace("{transcript}", transcript.full_text)
    else:
        # Use the default prompt
        if full_transcript and full_transcript != transcript:
            # If we have both transcripts and they're different, use both
            prompt = f"""
            You are a technical content creator specializing in software development, IT infrastructure, and emerging technologies. Create an informative title and description for a YouTube Short based on this transcript from "Nerding Out with Viktor".

            I'll provide you with two transcripts:
            1. The FULL EPISODE transcript for context
            2. The SHORT CLIP transcript (which is what we're creating a title and description for)

            FULL EPISODE TRANSCRIPT (for context only):
            {full_transcript.full_text}

            SHORT CLIP TRANSCRIPT (focus on this for the title and description):
            {transcript.full_text}

            REQUIREMENTS:
            1. Title: Create a technically accurate title under 100 characters that clearly communicates the technical topic
            2. Description: Write a technically focused description (2-3 sentences) that:
               - Summarizes the key technical information or insight
               - Explains why this is relevant to developers, engineers, or IT professionals
               - Mentions specific technologies, standards, or methodologies where appropriate
            3. Both should use proper technical terminology and avoid oversimplification
            4. Focus on the technical aspects from the SHORT CLIP transcript
            5. Assume an audience with technical knowledge and interest in the subject
            6. Do not include any notes or explanations about your process

            FORMAT YOUR RESPONSE IN MARKDOWN LIKE THIS:
            # Title

            Description text here
            """
        else:
            # If we only have one transcript, use it
            prompt = f"""
            You are a technical content creator specializing in software development, IT infrastructure, and emerging technologies. Create an informative title and description for a YouTube Short based on this transcript from "Nerding Out with Viktor".

            TRANSCRIPT:
            {transcript.full_text}

            REQUIREMENTS:
            1. Title: Create a technically accurate title under 100 characters that clearly communicates the technical topic
            2. Description: Write a technically focused description (2-3 sentences) that:
               - Summarizes the key technical information or insight
               - Explains why this is relevant to developers, engineers, or IT professionals
               - Mentions specific technologies, standards, or methodologies where appropriate
            3. Both should use proper technical terminology and avoid oversimplification
            4. Focus on the technical aspects from the transcript
            5. Assume an audience with technical knowledge and interest in the subject
            6. Do not include any notes or explanations about your process

            FORMAT YOUR RESPONSE IN MARKDOWN LIKE THIS:
            # Title

            Description text here
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

        # Only keep these final print statements for minimal output
        logger.info(f"Final title: {title}")
        logger.info(f"Final description: {description[:100]}{'...' if len(description) > 100 else ''}")

        return title, description

    except Exception as e:
        logger.error(f"Error calling Ollama: {str(e)}")
        # Fallback in case of error
        return (
            f"Video Transcript",
            f"This video contains the following content: {transcript.full_text[:200]}..."
        )

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