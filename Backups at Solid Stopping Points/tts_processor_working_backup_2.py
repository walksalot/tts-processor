#!/usr/bin/env python3
import os
import sys
import argparse
import logging
import time
import json
import hashlib
import re
import concurrent.futures
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import anthropic
import openai
import inquirer
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskID,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.markdown import Markdown
from rich.table import Table
from rich.text import Text
from rich.prompt import Confirm
from rich.layout import Layout
from rich.live import Live
from rich.status import Status
from rich import box
from pydub import AudioSegment

# IMPORTANT: This script REQUIRES Claude 3.7 Sonnet specifically.
# NO OTHER CLAUDE MODEL VERSIONS should be used under any circumstances.
# Claude 3.7 Sonnet offers a context window of 200,000 tokens and supports
# output up to 128,000 tokens per request (64,000 tokens generally available,
# 64,000-128,000 tokens in beta).
CLAUDE_MODEL = "claude-3-7-sonnet-20250219"  # ONLY use Claude 3.7 Sonnet

# Configure rich console for pretty output
console = Console()

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class TextToSpeechProcessor:
    def __init__(self):
        """Initialize the TextToSpeechProcessor with configuration."""
        # Load API keys from environment or files
        with console.status("[bold blue]Loading API keys...[/bold blue]"):
            self.anthropic_api_key = self._load_anthropic_api_key()
            self.openai_api_key = self._load_openai_api_key()

        # Initialize API clients
        self.anthropic_client = anthropic.Anthropic(api_key=self.anthropic_api_key)
        self.openai_client = openai.OpenAI(api_key=self.openai_api_key)

        # Cache directory for storing processed results
        self.cache_dir = Path.home() / ".text2speech_cache"
        self.cache_dir.mkdir(exist_ok=True)

        # OpenAI TTS voice options
        self.voice_options = {
            "Standard voices": ["alloy", "echo", "fable", "onyx", "nova", "shimmer"],
            "HD voices": [
                "alloy-hd",
                "echo-hd",
                "fable-hd",
                "onyx-hd",
                "nova-hd",
                "shimmer-hd",
            ],
        }

        # Voice descriptions to help users choose - more detailed with gender and style information
        self.voice_descriptions = {
            "alloy": "Neutral gender, versatile voice with a balanced, professional tone",
            "echo": "Female-sounding, soft-spoken voice with a gentle, clear delivery",
            "fable": "Female-sounding, whimsical voice with an expressive, animated style for storytelling",
            "onyx": "Male-sounding, deep voice with an authoritative, rich tone for formal narration",
            "nova": "Female-sounding, warm voice with a friendly, approachable tone",
            "shimmer": "Female-sounding, bright voice with a melodic, upbeat quality for energetic content",
            "alloy-hd": "HD: Neutral gender, versatile voice with enhanced clarity and professional tone",
            "echo-hd": "HD: Female-sounding, soft-spoken voice with improved articulation and warmth",
            "fable-hd": "HD: Female-sounding, whimsical voice with rich expressiveness and character",
            "onyx-hd": "HD: Male-sounding, deep voice with premium bass resonance and authority",
            "nova-hd": "HD: Female-sounding, warm voice with enhanced natural intonation and friendliness",
            "shimmer-hd": "HD: Female-sounding, bright voice with crystal-clear high notes and dynamic range",
        }

        # Speech rate options
        self.speech_rate_options = [
            {"label": "Slower (0.8x)", "value": 0.8},
            {"label": "Slightly slower (0.9x)", "value": 0.9},
            {"label": "Normal speed (1.0x)", "value": 1.0},
            {"label": "Slightly faster (1.1x)", "value": 1.1},
            {"label": "Faster (1.2x)", "value": 1.2},
            {"label": "Very fast (1.5x)", "value": 1.5},
        ]

    def _load_anthropic_api_key(self) -> str:
        """Load Anthropic API key from environment or ~/.anthropic file."""
        # First try environment variable
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if api_key:
            logger.info("Loaded Anthropic API key from environment variable")
            return api_key

        # Then try ~/.anthropic file
        anthropic_file = Path.home() / ".anthropic"
        if anthropic_file.exists():
            try:
                content = anthropic_file.read_text().strip()
                # Look for API key pattern or key=value format
                if re.match(r"^sk-ant-api\w+$", content):
                    logger.info("Loaded Anthropic API key from ~/.anthropic file")
                    return content

                # Try to find key in a key=value format
                match = re.search(
                    r"(?:ANTHROPIC_API_KEY|api_key)[=:]\s*([^\s]+)", content
                )
                if match:
                    logger.info("Loaded Anthropic API key from ~/.anthropic file")
                    return match.group(1)

                # If it's a JSON file, try to parse it
                try:
                    data = json.loads(content)
                    if isinstance(data, dict) and "api_key" in data:
                        logger.info(
                            "Loaded Anthropic API key from ~/.anthropic JSON file"
                        )
                        return data["api_key"]
                except json.JSONDecodeError:
                    pass
            except Exception as e:
                logger.warning(f"Error reading ~/.anthropic file: {e}")

        console.print(
            "[bold red]Error:[/bold red] Anthropic API key not found in environment or ~/.anthropic file"
        )
        console.print(
            "Please provide your Anthropic API key when prompted or set the ANTHROPIC_API_KEY environment variable"
        )
        api_key = console.input("[bold green]Enter Anthropic API key: [/bold green]")
        return api_key

    def _load_openai_api_key(self) -> str:
        """Load OpenAI API key from environment, ~/.apikey, or ~/.openai file."""
        # First try environment variable
        api_key = os.environ.get("OPENAI_API_KEY")
        if api_key:
            logger.info("Loaded OpenAI API key from environment variable")
            return api_key

        # Try ~/.apikey file
        apikey_file = Path.home() / ".apikey"
        if apikey_file.exists():
            try:
                content = apikey_file.read_text().strip()

                # Try to find OpenAI key in content
                if re.match(r"^sk-\w+$", content):
                    logger.info("Loaded OpenAI API key from ~/.apikey file")
                    return content

                # Try to find key in a key=value format
                match = re.search(r"(?:OPENAI_API_KEY|api_key)[=:]\s*([^\s]+)", content)
                if match:
                    logger.info("Loaded OpenAI API key from ~/.apikey file")
                    return match.group(1)

                # If it's a JSON file, try to parse it
                try:
                    data = json.loads(content)
                    if isinstance(data, dict) and "api_key" in data:
                        logger.info("Loaded OpenAI API key from ~/.apikey JSON file")
                        return data["api_key"]
                except json.JSONDecodeError:
                    pass
            except Exception as e:
                logger.warning(f"Error reading ~/.apikey file: {e}")

        # Then check ~/.openai file if it exists
        openai_file = Path.home() / ".openai"
        if openai_file.exists():
            try:
                content = openai_file.read_text().strip()

                # Try to find key in content
                if re.match(r"^sk-\w+$", content):
                    logger.info("Loaded OpenAI API key from ~/.openai file")
                    return content

                # Try to find key in a key=value format
                match = re.search(r"(?:OPENAI_API_KEY|api_key)[=:]\s*([^\s]+)", content)
                if match:
                    logger.info("Loaded OpenAI API key from ~/.openai file")
                    return match.group(1)

                # If it's a JSON file, try to parse it
                try:
                    data = json.loads(content)
                    if isinstance(data, dict) and "api_key" in data:
                        logger.info("Loaded OpenAI API key from ~/.openai JSON file")
                        return data["api_key"]
                except json.JSONDecodeError:
                    pass
            except Exception as e:
                logger.warning(f"Error reading ~/.openai file: {e}")

        console.print(
            "[bold red]Error:[/bold red] OpenAI API key not found in environment, ~/.apikey, or ~/.openai file"
        )
        console.print(
            "Please provide your OpenAI API key when prompted or set the OPENAI_API_KEY environment variable"
        )
        api_key = console.input("[bold green]Enter OpenAI API key: [/bold green]")
        return api_key

    def _get_content_hash(self, text: str, voice: str, model: str, format: str) -> str:
        """Generate a hash of the input content and parameters to identify unique processing requests."""
        hash_input = f"{text}|{voice}|{model}|{format}"
        return hashlib.md5(hash_input.encode()).hexdigest()

    def _check_cache(self, content_hash: str) -> Optional[Dict]:
        """Check if the exact content has been processed before."""
        cache_file = self.cache_dir / f"{content_hash}.json"
        if cache_file.exists():
            try:
                return json.loads(cache_file.read_text())
            except Exception as e:
                logger.warning(f"Error reading cache file: {e}")
                return None
        return None

    def _save_to_cache(self, content_hash: str, metadata: Dict) -> None:
        """Save processing metadata to cache."""
        cache_file = self.cache_dir / f"{content_hash}.json"
        try:
            cache_file.write_text(json.dumps(metadata, indent=2))
        except Exception as e:
            logger.warning(f"Error writing to cache: {e}")

    def display_voice_options(self, non_interactive=False) -> Tuple[str, str, float]:
        """Display voice options in a table and let user select or use defaults."""
        if non_interactive:
            # Use default options for non-interactive mode
            logger.info(
                "Running in non-interactive mode, using default voice: onyx-hd, mp3 format, 1.0x speed"
            )
            return "onyx-hd", "mp3", 1.0

        # Create a visually appealing main selection panel
        console.print(
            Panel(
                "[bold]Voice and Audio Settings[/bold]\n"
                "Configure how your text will sound when converted to speech.",
                title="🔊 Audio Configuration",
                border_style="cyan",
                padding=(1, 2),
            )
        )

        # First select voice quality
        console.print("\n[bold cyan]Step 1: Voice Quality Options[/bold cyan]")

        quality_table = Table(
            show_header=True, header_style="bold magenta", box=box.ROUNDED
        )
        quality_table.add_column("Option", style="dim", justify="center", width=8)
        quality_table.add_column("Quality", style="cyan", width=15)
        quality_table.add_column("Description", style="green")

        quality_table.add_row(
            "1", "Standard Voices", "Regular quality, less expensive, faster processing"
        )
        quality_table.add_row(
            "2", "HD Voices", "Higher quality, more natural sounding, premium option"
        )

        console.print(quality_table)

        quality_choice = ""
        while quality_choice not in ["1", "2"]:
            quality_choice = console.input(
                "\n[bold yellow]Select voice quality (1/2): [/bold yellow]"
            )

        voice_category = "Standard voices" if quality_choice == "1" else "HD voices"

        # Then select specific voice with improved visuals
        console.print(f"\n[bold cyan]Step 2: Select Voice Type[/bold cyan]")
        console.print(
            f"[cyan]Choose from the available {voice_category.lower()}:[/cyan]"
        )

        voice_table = Table(
            show_header=True, header_style="bold magenta", box=box.ROUNDED
        )
        voice_table.add_column("Option", style="dim", justify="center", width=8)
        voice_table.add_column("Voice", style="cyan", width=12)
        voice_table.add_column("Description", style="green")
        voice_table.add_column("Best For", style="yellow")

        # Add use case recommendations based on voice characteristics
        voice_use_cases = {
            "alloy": "General purpose, professional content",
            "echo": "Relaxing content, guided meditations",
            "fable": "Stories, children's content, creative narratives",
            "onyx": "Documentaries, formal announcements, business content",
            "nova": "Educational content, explanations, approachable topics",
            "shimmer": "Upbeat content, advertisements, energetic material",
            "alloy-hd": "Professional podcasts, high-quality narration",
            "echo-hd": "Premium meditation, audiobooks with emotive depth",
            "fable-hd": "Professional storytelling, character-rich narratives",
            "onyx-hd": "Premium documentaries, authoritative announcements",
            "nova-hd": "High-quality educational content, premium explainers",
            "shimmer-hd": "Premium marketing, high-energy professional content",
        }

        voices = self.voice_options[voice_category]
        for i, voice in enumerate(voices, 1):
            voice_table.add_row(
                str(i),
                voice,
                self.voice_descriptions.get(voice, "No description available"),
                voice_use_cases.get(voice, "Various content"),
            )

        console.print(voice_table)

        voice_idx = 0
        max_idx = len(voices)
        while voice_idx < 1 or voice_idx > max_idx:
            try:
                voice_choice = console.input(
                    f"\n[bold yellow]Select voice (1-{max_idx}): [/bold yellow]"
                )
                voice_idx = int(voice_choice)
            except ValueError:
                console.print("[bold red]Please enter a valid number[/bold red]")

        selected_voice = voices[voice_idx - 1]

        # Let user select speech rate
        console.print("\n[bold cyan]Step 3: Speech Rate[/bold cyan]")
        console.print("[cyan]How fast should the voice speak?[/cyan]")

        rate_table = Table(
            show_header=True, header_style="bold magenta", box=box.ROUNDED
        )
        rate_table.add_column("Option", style="dim", justify="center", width=8)
        rate_table.add_column("Speed", style="cyan", width=20)
        rate_table.add_column("Best For", style="green")

        rate_use_cases = [
            "Detailed educational content, complex information",
            "Easy comprehension, learning new material",
            "Standard listening, balanced pacing",
            "Efficient consumption of familiar content",
            "Faster review of familiar materials",
            "Quick review, refreshing known content",
        ]

        for i, rate_option in enumerate(self.speech_rate_options, 1):
            rate_table.add_row(str(i), rate_option["label"], rate_use_cases[i - 1])

        console.print(rate_table)

        rate_idx = 0
        max_rate_idx = len(self.speech_rate_options)
        while rate_idx < 1 or rate_idx > max_rate_idx:
            try:
                rate_choice = console.input(
                    f"\n[bold yellow]Select speech rate (1-{max_rate_idx}): [/bold yellow]"
                )
                rate_idx = int(rate_choice)
            except ValueError:
                console.print("[bold red]Please enter a valid number[/bold red]")

        selected_rate = self.speech_rate_options[rate_idx - 1]["value"]

        # Finally select format with enhanced visuals
        console.print("\n[bold cyan]Step 4: Output Format[/bold cyan]")
        console.print("[cyan]Choose the audio file format:[/cyan]")

        format_table = Table(
            show_header=True, header_style="bold magenta", box=box.ROUNDED
        )
        format_table.add_column("Option", style="dim", justify="center", width=8)
        format_table.add_column("Format", style="cyan", width=12)
        format_table.add_column("Description", style="green")
        format_table.add_column("Best For", style="yellow")

        format_table.add_row(
            "1",
            "MP3",
            "Compressed audio format, smaller file size",
            "Sharing, mobile devices, standard use",
        )
        format_table.add_row(
            "2",
            "WAV",
            "Uncompressed audio format, higher quality",
            "Professional use, editing, highest quality needs",
        )

        console.print(format_table)

        format_choice = ""
        while format_choice not in ["1", "2"]:
            format_choice = console.input(
                "\n[bold yellow]Select output format (1/2): [/bold yellow]"
            )

        output_format = "mp3" if format_choice == "1" else "wav"

        # Summary of selections
        console.print(
            Panel(
                f"[bold]Configuration Summary:[/bold]\n\n"
                f"[cyan]• Voice:[/cyan] {selected_voice} - {self.voice_descriptions.get(selected_voice)}\n"
                f"[cyan]• Speech Rate:[/cyan] {selected_rate}x\n"
                f"[cyan]• Format:[/cyan] {output_format.upper()}\n",
                title="✅ Your Selections",
                border_style="green",
                padding=(1, 2),
            )
        )

        return selected_voice, output_format, selected_rate

    def preprocess_text(self, input_text: str) -> str:
        """Use Claude 3.7 to preprocess text for speech readability with enhanced progress updates."""
        # Get input text stats for progress reporting
        char_count = len(input_text)
        word_count = len(input_text.split())
        paragraph_count = len(input_text.split("\n\n"))

        console.print(
            Panel(
                "[cyan]Claude 3.7 will now preprocess the text to optimize it for speech while preserving all important content.[/cyan]",
                title="Step 2: Text Preprocessing",
                border_style="green",
                padding=(1, 2),
            )
        )

        # Create a single progress display that we'll update
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}[/bold blue]"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            # Initialize progress variables
            start_time = time.time()
            received_chars = 0
            total_task = progress.add_task(
                f"Preprocessing Text with Claude 3.7: 0/{char_count:,} characters",
                total=char_count,
            )

            logger.info(
                f"Preprocessing text with Claude 3.7: {char_count} characters, {word_count} words"
            )

            # Create a very specific prompt for Claude
            system_prompt = """
            You are a text preprocessing expert tasked with preparing text for text-to-speech conversion.

            CRITICAL INSTRUCTIONS:
            1. DO NOT remove any substantial content - keep all meaningful information, arguments, data points, and insights.
            2. DO remove formatting that doesn't make sense when read aloud:
               - URLs: Replace with brief descriptions like "[Link to research paper]" only when the URL appears as a raw link
               - Markdown formatting: Remove *, #, -, _, ~, >, etc. symbols but preserve the underlying text and structure
               - Code blocks: Convert to natural language descriptions if brief, or indicate "[Code section omitted]" for lengthy blocks
               - Tables: Convert simple tables to natural language, using phrases like "The data shows the following values..."
            3. Improve text flow and readability:
               - Add natural transitions between sections
               - Convert headers to spoken-word phrases ("Chapter 1: Introduction" or "Moving on to the next section about...")
               - Replace special characters with their spoken equivalents (e.g., % → "percent")
               - Spell out abbreviations on first use when helpful for clarity
            4. Make no other changes - do not summarize, rewrite, or alter the substance of the text
            5. Your output should be a clean, flowing script ready to be converted to audio that reads like a professional audiobook

            Return ONLY the processed text with no explanations, annotations, or meta-commentary.
            """

            try:
                # Create a streaming response for large text processing
                processed_text = ""

                with self.anthropic_client.messages.stream(
                    model=CLAUDE_MODEL,
                    system=system_prompt,
                    messages=[
                        {
                            "role": "user",
                            "content": f"Process the following text for text-to-speech conversion according to the instructions. Here's the text:\n\n{input_text}",
                        }
                    ],
                    max_tokens=64000,
                ) as stream:
                    # Update every second or so to avoid too many refreshes
                    last_update = time.time()

                    for text in stream.text_stream:
                        processed_text += text
                        received_chars = len(processed_text)

                        # Only update the display about once per second to avoid flickering
                        current_time = time.time()
                        if current_time - last_update >= 0.5:
                            progress.update(
                                total_task,
                                completed=min(received_chars, char_count),
                                description=f"Preprocessing Text with Claude 3.7: {received_chars:,}/{char_count:,} characters",
                            )
                            last_update = current_time

                # Complete the progress
                progress.update(total_task, completed=char_count)

                # Final completion metrics
                processed_char_count = len(processed_text)
                processed_word_count = len(processed_text.split())
                processed_paragraph_count = len(processed_text.split("\n\n"))

                # Calculate and display change metrics
                char_diff = processed_char_count - char_count
                char_pct_change = (
                    (char_diff / char_count * 100) if char_count > 0 else 0
                )
                word_diff = processed_word_count - word_count
                word_pct_change = (
                    (word_diff / word_count * 100) if word_count > 0 else 0
                )

                # Add a final stats table
                stats_table = Table(
                    title="Text Preprocessing Results",
                    show_header=True,
                    box=box.ROUNDED,
                )
                stats_table.add_column("Metric", style="cyan")
                stats_table.add_column("Original", style="yellow", justify="right")
                stats_table.add_column("Processed", style="green", justify="right")
                stats_table.add_column("Change", style="magenta", justify="right")

                stats_table.add_row(
                    "Characters",
                    f"{char_count:,}",
                    f"{processed_char_count:,}",
                    f"{char_diff:+,} ({char_pct_change:+.1f}%)",
                )
                stats_table.add_row(
                    "Words",
                    f"{word_count:,}",
                    f"{processed_word_count:,}",
                    f"{word_diff:+,} ({word_pct_change:+.1f}%)",
                )
                stats_table.add_row(
                    "Paragraphs",
                    f"{paragraph_count:,}",
                    f"{processed_paragraph_count:,}",
                    f"{processed_paragraph_count - paragraph_count:+,}",
                )

                console.print(stats_table)

                # Display a summary of what was done
                elapsed = time.time() - start_time
                console.print(
                    f"[dim]• Processing Speed: {received_chars / max(1, elapsed):.1f} chars/sec\n"
                    f"• Time: {elapsed:.1f}s total[/dim]"
                )

                logger.info(
                    f"Text preprocessing complete. Original: {char_count} chars, Processed: {processed_char_count} chars."
                )

                # Compare length to ensure we didn't lose too much content
                retention_rate = (
                    processed_word_count / word_count * 100 if word_count > 0 else 100
                )

                if retention_rate < 85:
                    console.print(
                        f"[bold yellow]Warning:[/bold yellow] Processed text is {retention_rate:.1f}% of the original length. Some content may have been lost."
                    )

                return processed_text

            except Exception as e:
                logger.error(f"Error during text preprocessing: {e}")
                raise

    def generate_filename(self, text: str) -> str:
        """Ask Claude to generate a descriptive filename based on the text content with enhanced progress display."""
        with Status(
            "[bold green]Analyzing content to generate descriptive filename...[/bold green]",
            spinner="dots",
        ) as status:
            logger.info("Asking Claude to suggest a filename...")

            start_time = time.time()
            try:
                filename = ""
                with self.anthropic_client.messages.stream(
                    model=CLAUDE_MODEL,
                    max_tokens=100,
                    messages=[
                        {
                            "role": "user",
                            "content": f"""Based on the following text, suggest a short, descriptive filename (without extension)
                            that captures the main topic or content. The filename should be valid for use in file systems
                            (no special characters other than dashes or underscores, no spaces).
                            Keep it under 50 characters. Respond with ONLY the filename, nothing else.

                            Text content:
                            {text[:2000]}...
                            """,
                        }
                    ],
                ) as stream:
                    for text_chunk in stream.text_stream:
                        filename += text_chunk
                        status.update(
                            f"[bold green]Generating filename: {filename}[/bold green]"
                        )

                # Clean up filename to ensure it's valid
                filename = filename.strip()
                filename = re.sub(r"[^\w\-]", "_", filename)

                elapsed = time.time() - start_time
                logger.info(f"Generated filename: '{filename}' in {elapsed:.2f}s")

                return filename

            except Exception as e:
                logger.error(f"Error generating filename: {e}")
                # Fallback to a timestamp-based filename
                return f"tts_output_{int(time.time())}"

    def split_text_intelligently(
        self, text: str, max_chunk_size: int = 4000
    ) -> List[str]:
        """
        Split text into chunks at natural boundaries like paragraphs and sentences.

        This ensures each chunk makes sense on its own and improves TTS quality.
        """
        # First, try to split by paragraphs (double newlines)
        paragraphs = re.split(r"\n\s*\n", text)

        chunks = []
        current_chunk = ""

        # Try to keep paragraphs together when possible
        for paragraph in paragraphs:
            # If adding this paragraph would exceed the max size
            if len(current_chunk) + len(paragraph) + 2 > max_chunk_size:
                # If the current paragraph alone exceeds the max size, we need to split it by sentences
                if len(paragraph) > max_chunk_size:
                    # If current_chunk has content, add it first
                    if current_chunk:
                        chunks.append(current_chunk)
                        current_chunk = ""

                    # Split the large paragraph by sentences
                    sentences = re.split(r"(?<=[.!?])\s+", paragraph)
                    for sentence in sentences:
                        # If this single sentence is still too large, we'll have to split it
                        if len(sentence) > max_chunk_size:
                            # Split by character chunks as a last resort
                            for i in range(0, len(sentence), max_chunk_size):
                                chunks.append(sentence[i : i + max_chunk_size])
                        else:
                            # Check if adding this sentence would exceed max size
                            if len(current_chunk) + len(sentence) + 1 > max_chunk_size:
                                chunks.append(current_chunk)
                                current_chunk = sentence
                            else:
                                current_chunk = (
                                    f"{current_chunk} {sentence}"
                                    if current_chunk
                                    else sentence
                                )
                else:
                    # The paragraph is too big to add to current chunk but fits in its own chunk
                    chunks.append(current_chunk)
                    current_chunk = paragraph
            else:
                # Add this paragraph to the current chunk
                if current_chunk:
                    current_chunk = f"{current_chunk}\n\n{paragraph}"
                else:
                    current_chunk = paragraph

        # Don't forget the last chunk
        if current_chunk:
            chunks.append(current_chunk)

        # Log the chunking results
        logger.info(f"Split text into {len(chunks)} chunks at natural boundaries")

        return chunks

    def process_chunk(
        self, chunk: str, voice: str, model: str, api_voice: str, temp_file: str
    ) -> str:
        """Process a single chunk of text to speech for parallel processing."""
        try:
            response = self.openai_client.audio.speech.create(
                model=model, voice=api_voice, input=chunk
            )
            response.stream_to_file(temp_file)
            return temp_file
        except Exception as e:
            logger.error(f"Error processing chunk: {e}")
            raise

    def text_to_speech(
        self, text: str, voice: str, output_file: str, speech_rate: float = 1.0
    ) -> Tuple[str, dict]:
        """
        Convert text to speech using OpenAI's TTS API with intelligent chunking and parallel processing.

        Args:
            text: The processed text to convert to speech
            voice: The selected voice
            output_file: Path to the output audio file
            speech_rate: Speed multiplier for speech (0.8-1.5x)

        Returns:
            Tuple containing the output file path and audio metadata
        """
        # Determine if HD or standard voice
        model = "tts-1-hd" if "-hd" in voice else "tts-1"
        # For HD voices, remove the -hd suffix for the actual API call
        api_voice = voice.replace("-hd", "") if "-hd" in voice else voice

        # Use intelligent chunking (4000 chars is OpenAI's approximate limit)
        max_chunk_size = 4000
        char_count = len(text)

        console.print(
            Panel(
                f"[cyan]Converting text to speech using OpenAI's TTS API with {voice} voice at {speech_rate}x speed...[/cyan]",
                title="Step 5: Text-to-Speech Conversion",
                border_style="green",
                padding=(1, 2),
            )
        )

        # Split text with intelligent chunking
        chunks = self.split_text_intelligently(text, max_chunk_size)

        # Display chunking info
        chunks_info = [
            f"[bold]Text-to-Speech Conversion with OpenAI:[/bold]",
            f"",
            f"[cyan]• Voice:[/cyan] {voice}",
            f"[cyan]• Model:[/cyan] {model}",
            f"[cyan]• Speech Rate:[/cyan] {speech_rate}x",
            f"[cyan]• Text Size:[/cyan] {char_count:,} characters",
            f"[cyan]• Chunks:[/cyan] {len(chunks)} total",
            f"[cyan]• Chunking Method:[/cyan] Intelligent (paragraph/sentence boundaries)",
            f"[cyan]• Average Chunk Size:[/cyan] {sum(len(c) for c in chunks) / max(1, len(chunks)):.1f} characters",
        ]

        console.print(
            Panel("\n".join(chunks_info), title="TTS Conversion", border_style="blue")
        )

        # Process chunks with progress bar
        total_chunks = len(chunks)

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}[/bold blue]"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            # Add the main processing task
            process_task = progress.add_task(
                "Processing audio chunks", total=total_chunks
            )
            temp_files = [None] * total_chunks
            processed_chunks = 0
            start_time = time.time()

            try:
                # Use a ThreadPoolExecutor for parallel processing
                # Limit to 3 concurrent chunks to avoid API rate limits
                max_workers = min(3, total_chunks) if total_chunks > 0 else 1

                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=max_workers
                ) as executor:
                    # Map of future to chunk index to track order
                    future_to_index = {}

                    # Prepare the tasks
                    for i, chunk in enumerate(chunks):
                        temp_file = f"{output_file}.part{i}"
                        future = executor.submit(
                            self.process_chunk,
                            chunk,
                            voice,
                            model,
                            api_voice,
                            temp_file,
                        )
                        future_to_index[future] = i

                    # Process as they complete
                    for future in concurrent.futures.as_completed(
                        future_to_index.keys()
                    ):
                        processed_chunks += 1
                        temp_file = future.result()
                        chunk_index = future_to_index[future]

                        # Store in the correct position to maintain order
                        temp_files[chunk_index] = temp_file

                        # Update progress
                        progress.update(
                            process_task,
                            advance=1,
                            description=f"Processing audio chunk {processed_chunks}/{total_chunks}",
                        )

                # Add a merging task
                merge_task = progress.add_task(
                    "Combining audio segments", total=len(temp_files)
                )

                # Combine all chunks using pydub
                combine_start = time.time()
                combined = AudioSegment.empty()

                for i, temp_file in enumerate(temp_files):
                    try:
                        segment = AudioSegment.from_file(temp_file)
                        combined += segment
                        progress.update(
                            merge_task,
                            advance=1,
                            description=f"Merging segment {i+1}/{len(temp_files)}",
                        )
                    except Exception as e:
                        logger.error(f"Error processing audio segment {i}: {e}")
                        console.print(
                            f"[bold red]Warning:[/bold red] Error processing segment {i}, skipping."
                        )

                # Apply speech rate adjustment if not at normal speed (1.0)
                if speech_rate != 1.0:
                    speed_task = progress.add_task(
                        f"Adjusting speech rate to {speech_rate}x", total=1
                    )

                    # Use the speedup function from pydub.effects
                    # This preserves pitch while changing speed
                    if hasattr(combined, "speedup"):
                        # Use native speedup if available (newer pydub versions)
                        combined = combined.speedup(playback_speed=speech_rate)
                    else:
                        # Fallback implementation using frame rate manipulation to preserve pitch
                        # Get the original parameters
                        original_frame_rate = combined.frame_rate

                        # Calculate new frame rate that will result in desired speed change
                        new_frame_rate = int(original_frame_rate * speech_rate)

                        # Export with new frame rate
                        temp_speed_file = f"{output_file}.speed_adjusted"
                        combined.export(
                            temp_speed_file,
                            format="wav",
                            parameters=["-ar", str(new_frame_rate)],
                        )

                        # Reimport at original frame rate
                        combined = AudioSegment.from_file(
                            temp_speed_file, format="wav"
                        ).set_frame_rate(original_frame_rate)

                        # Clean up temp file
                        if os.path.exists(temp_speed_file):
                            os.remove(temp_speed_file)

                    progress.update(speed_task, completed=1)

                # Export task
                export_task = progress.add_task("Exporting final audio file", total=1)

                # Get audio file size before exporting
                temp_size_check = f"{output_file}.size_check"
                combined.export(temp_size_check, format=output_file.split(".")[-1])
                file_size_bytes = os.path.getsize(temp_size_check)
                file_size_mb = file_size_bytes / (1024 * 1024)
                os.remove(temp_size_check)

                # Export the combined audio
                combined.export(output_file, format=output_file.split(".")[-1])
                progress.update(export_task, completed=1)

                # Clean up temp files
                cleanup_task = progress.add_task(
                    "Cleaning up temporary files", total=len(temp_files)
                )
                for temp_file in temp_files:
                    os.remove(temp_file)
                    progress.update(cleanup_task, advance=1)

                # Final metadata
                total_time = time.time() - start_time
                audio_duration = len(combined) / 1000  # in seconds

                # Create metadata dict for summary report
                metadata = {
                    "audio_duration": audio_duration,
                    "audio_duration_formatted": f"{int(audio_duration // 60)}:{int(audio_duration % 60):02d}",
                    "file_size_bytes": file_size_bytes,
                    "file_size_mb": file_size_mb,
                    "char_count": char_count,
                    "chunks": total_chunks,
                    "processing_time": total_time,
                    "chars_per_second": char_count / max(1, total_time),
                }

            except Exception as e:
                logger.error(f"Error during text-to-speech conversion: {e}")
                raise

        # Display a compact summary
        console.print(
            Panel(
                f"[bold green]Audio Generation Complete![/bold green]\n\n"
                f"[cyan]• Output:[/cyan] {output_file}\n"
                f"[cyan]• Audio Length:[/cyan] {metadata['audio_duration_formatted']} ({metadata['audio_duration']:.1f}s)\n"
                f"[cyan]• File Size:[/cyan] {metadata['file_size_mb']:.2f} MB\n"
                f"[cyan]• Speech Rate:[/cyan] {speech_rate}x\n"
                f"[dim]• Processing Time: {total_time:.1f}s ({metadata['chars_per_second']:.1f} chars/sec)[/dim]",
                title="Audio Generated Successfully",
                border_style="green",
            )
        )

        logger.info(f"Audio saved to {output_file}")
        return output_file, metadata

    def process_file(self, input_file: str, non_interactive=False) -> None:
        """Process a text file through the entire pipeline with user interaction."""
        # Show welcome banner
        console.print(
            Panel.fit(
                "[bold green]Text-to-Speech Processor[/bold green]\n"
                "[cyan]Convert text files to professional-sounding speech[/cyan]",
                title="🎙️ TTS Processor",
                border_style="blue",
                padding=(1, 2),
            )
        )

        # Start tracking all process metrics for final report
        process_metrics = {
            "start_time": time.time(),
            "input_file": input_file,
        }

        # Read input file
        input_path = Path(input_file)
        console.print(f"\n[bold]Processing:[/bold] [cyan]{input_path}[/cyan]")

        try:
            with open(input_file, "r", encoding="utf-8") as f:
                input_text = f.read()

            # Show text stats
            char_count = len(input_text)
            word_count = len(input_text.split())
            paragraph_count = len(input_text.split("\n\n"))

            # Save input text metrics
            process_metrics["input_text"] = {
                "char_count": char_count,
                "word_count": word_count,
                "paragraph_count": paragraph_count,
                "file_size_bytes": os.path.getsize(input_file),
                "file_size_mb": os.path.getsize(input_file) / (1024 * 1024),
            }

            console.print(
                f"[dim]Text contains {char_count:,} characters, approximately {word_count:,} words, "
                f"and {paragraph_count:,} paragraphs[/dim]"
            )
        except Exception as e:
            console.print(f"[bold red]Error reading input file:[/bold red] {e}")
            return

        # Voice and format selection
        if not non_interactive:
            console.print(
                Panel(
                    "[cyan]Next, you'll select the voice and output format for your audio.[/cyan]",
                    title="Step 1: Voice and Format Selection",
                    border_style="green",
                    padding=(1, 2),
                )
            )

        voice, output_format, speech_rate = self.display_voice_options(non_interactive)

        # Save voice selection metrics
        process_metrics["voice_settings"] = {
            "voice": voice,
            "output_format": output_format,
            "speech_rate": speech_rate,
        }

        # Check if we've processed this exact content before
        model = "tts-1-hd" if "-hd" in voice else "tts-1"
        content_hash = self._get_content_hash(input_text, voice, model, output_format)
        cache_data = self._check_cache(content_hash)

        if cache_data and not non_interactive:
            console.print(
                Panel(
                    f"[bold yellow]This exact content has been processed before![/bold yellow]\n"
                    f"Previous output directory: [cyan]{cache_data['output_dir']}[/cyan]\n\n"
                    f"Processing again will create a new set of output files.",
                    title="Cache Found",
                    border_style="yellow",
                    padding=(1, 2),
                )
            )

            reprocess = Confirm.ask("Do you want to process it again?", default=False)

            if not reprocess:
                console.print("[bold green]Using cached results. Exiting.[/bold green]")
                return
        elif cache_data and non_interactive:
            # In non-interactive mode, always reprocess
            console.print(
                "[yellow]Cache found, but reprocessing in non-interactive mode.[/yellow]"
            )

        # Preprocess text with Claude
        console.print(
            Panel(
                "[cyan]Claude 3.7 will now preprocess the text to optimize it for speech while preserving all important content.[/cyan]",
                title="Step 2: Text Preprocessing",
                border_style="green",
                padding=(1, 2),
            )
        )

        processed_text = self.preprocess_text(input_text)

        # Save processed text metrics
        process_metrics["processed_text"] = {
            "char_count": len(processed_text),
            "word_count": len(processed_text.split()),
            "paragraph_count": len(processed_text.split("\n\n")),
            "char_diff": len(processed_text) - char_count,
            "char_diff_pct": (len(processed_text) - char_count)
            / max(1, char_count)
            * 100,
        }

        # Generate a descriptive filename/directory name
        console.print(
            Panel(
                "[cyan]Generating a descriptive name for output files based on content...[/cyan]",
                title="Step 3: Filename Generation",
                border_style="green",
                padding=(1, 2),
            )
        )

        base_name = self.generate_filename(processed_text)
        console.print(
            f"[bold green]Generated base name:[/bold green] [cyan]{base_name}[/cyan]"
        )

        # Create output directory
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        language_code = (
            "en"  # Default to English, could be expanded to detect or select language
        )
        output_dir = Path(f"{base_name}_{timestamp}")
        output_dir.mkdir(exist_ok=True)
        console.print(
            f"[bold green]Created output directory:[/bold green] [cyan]{output_dir}[/cyan]"
        )

        # Save directory info
        process_metrics["output_info"] = {
            "base_name": base_name,
            "timestamp": timestamp,
            "output_dir": str(output_dir),
            "language_code": language_code,
        }

        # Save all text versions
        console.print(
            Panel(
                "[cyan]Saving original and processed text files...[/cyan]",
                title="Step 4: Saving Text Files",
                border_style="green",
                padding=(1, 2),
            )
        )

        original_text_path = output_dir / f"{base_name}_original.txt"
        processed_text_path = output_dir / f"{base_name}_processed.txt"

        with open(original_text_path, "w", encoding="utf-8") as f:
            f.write(input_text)

        with open(processed_text_path, "w", encoding="utf-8") as f:
            f.write(processed_text)

        # Convert to speech
        console.print(
            Panel(
                f"[cyan]Converting text to speech using OpenAI's TTS API with {voice} voice at {speech_rate}x speed...[/cyan]",
                title="Step 5: Text-to-Speech Conversion",
                border_style="green",
                padding=(1, 2),
            )
        )

        audio_output_path = (
            output_dir
            / f"{base_name}_{language_code}_{voice}_{speech_rate}x.{output_format}"
        )
        audio_path, audio_metadata = self.text_to_speech(
            processed_text, voice, str(audio_output_path), speech_rate
        )

        # Save audio metadata to process metrics
        process_metrics["audio_metadata"] = audio_metadata

        # Add additional information to process metrics
        process_metrics["total_time"] = time.time() - process_metrics["start_time"]

        # Save metadata to cache
        metadata = {
            "timestamp": timestamp,
            "input_file": str(input_path),
            "output_dir": str(output_dir),
            "voice": voice,
            "format": output_format,
            "language": language_code,
            "speech_rate": speech_rate,
            "metrics": process_metrics,
        }
        self._save_to_cache(content_hash, metadata)

        # Generate comprehensive summary report
        self._generate_summary_report(
            process_metrics,
            output_dir,
            original_text_path,
            processed_text_path,
            audio_output_path,
        )

    def _generate_summary_report(
        self,
        metrics,
        output_dir,
        original_text_path,
        processed_text_path,
        audio_output_path,
    ):
        """Generate a comprehensive summary report of the entire process."""
        # Display header
        console.print("\n")
        console.print(
            Panel(
                "[bold green]Text-to-Speech Processing Complete![/bold green]",
                border_style="green",
                padding=(1, 2),
            )
        )

        # Create a layout for the report
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="stats"),
            Layout(name="files"),
            Layout(name="footer", size=3),
        )

        # Header
        layout["header"].update(
            Panel(
                f"[bold]Process completed in [cyan]{metrics['total_time']:.2f}[/cyan] seconds[/bold]",
                title="⏱️ Processing Time",
                border_style="blue",
            )
        )

        # Stats section with multiple tables
        stats_layout = Layout()
        stats_layout.split_row(Layout(name="text_stats"), Layout(name="audio_stats"))

        # Text statistics table
        text_stats = Table(
            title="📝 Text Processing Statistics", show_header=True, box=box.ROUNDED
        )
        text_stats.add_column("Metric", style="cyan", width=20)
        text_stats.add_column("Original", style="yellow", justify="right")
        text_stats.add_column("Processed", style="green", justify="right")
        text_stats.add_column("Change", style="magenta", justify="right")

        # Add text metrics
        input_text = metrics["input_text"]
        processed_text = metrics["processed_text"]

        text_stats.add_row(
            "Characters",
            f"{input_text['char_count']:,}",
            f"{processed_text['char_count']:,}",
            f"{processed_text['char_diff']:+,} ({processed_text['char_diff_pct']:.1f}%)",
        )

        text_stats.add_row(
            "Words",
            f"{input_text['word_count']:,}",
            f"{processed_text['word_count']:,}",
            f"{processed_text['word_count'] - input_text['word_count']:+,}",
        )

        text_stats.add_row(
            "Paragraphs",
            f"{input_text['paragraph_count']:,}",
            f"{processed_text['paragraph_count']:,}",
            f"{processed_text['paragraph_count'] - input_text['paragraph_count']:+,}",
        )

        # Audio statistics table
        audio_stats = Table(
            title="🔊 Audio Output Statistics", show_header=True, box=box.ROUNDED
        )
        audio_stats.add_column("Metric", style="cyan")
        audio_stats.add_column("Value", style="green")

        # Add audio metrics
        audio_metadata = metrics["audio_metadata"]
        voice_settings = metrics["voice_settings"]

        audio_stats.add_row(
            "Duration",
            f"{audio_metadata['audio_duration_formatted']} ({audio_metadata['audio_duration']:.1f}s)",
        )
        audio_stats.add_row("File Size", f"{audio_metadata['file_size_mb']:.2f} MB")
        audio_stats.add_row("Voice", f"{voice_settings['voice']}")
        audio_stats.add_row("Speech Rate", f"{voice_settings['speech_rate']}x")
        audio_stats.add_row("Format", f"{voice_settings['output_format'].upper()}")
        audio_stats.add_row("Chars/Second", f"{audio_metadata['chars_per_second']:.1f}")
        audio_stats.add_row(
            "Words/Minute",
            f"{(processed_text['word_count'] / audio_metadata['audio_duration']) * 60:.1f}",
        )

        # Update stats layout
        stats_layout["text_stats"].update(text_stats)
        stats_layout["audio_stats"].update(audio_stats)
        layout["stats"].update(stats_layout)

        # Files table
        files_table = Table(
            title="📁 Generated Output Files", show_header=True, box=box.ROUNDED
        )
        files_table.add_column("File Type", style="cyan")
        files_table.add_column("Description", style="yellow")
        files_table.add_column("Path", style="green")
        files_table.add_column("Size", style="magenta", justify="right")

        # Add file information
        original_size = os.path.getsize(original_text_path)
        processed_size = os.path.getsize(processed_text_path)
        audio_size = audio_metadata["file_size_bytes"]

        files_table.add_row(
            "Original Text",
            "Raw input text without modifications",
            str(original_text_path),
            f"{original_size / 1024:.1f} KB",
        )
        files_table.add_row(
            "Processed Text",
            "Text optimized for speech synthesis",
            str(processed_text_path),
            f"{processed_size / 1024:.1f} KB",
        )
        files_table.add_row(
            "Audio Output",
            f"{voice_settings['voice']} voice at {voice_settings['speech_rate']}x speed",
            str(audio_output_path),
            f"{audio_metadata['file_size_mb']:.2f} MB",
        )

        layout["files"].update(files_table)

        # Footer with playback tips
        layout["footer"].update(
            Panel(
                f"[cyan]Output directory:[/cyan] [bold green]{output_dir}[/bold green]\n"
                f"[cyan]Audio file:[/cyan] [bold green]{audio_output_path}[/bold green]",
                title="📋 Summary",
                border_style="green",
            )
        )

        # Print the full report
        console.print(layout)

        # Add a playback tips panel
        console.print(
            Panel(
                "[bold]Playback Tips:[/bold]\n"
                f"• Play the audio file with your default media player: [cyan]open {audio_output_path}[/cyan]\n"
                f"• Compare the original and processed text to see the TTS optimizations\n"
                f"• The processed text is ready for future TTS conversions with different voices",
                title="🎧 Next Steps",
                border_style="blue",
                padding=(1, 2),
            )
        )

        # Save a report file in markdown format
        report_path = output_dir / f"{metrics['output_info']['base_name']}_report.md"
        with open(report_path, "w") as f:
            f.write(f"# TTS Processing Report\n\n")
            f.write(f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Input File:** {metrics['input_file']}\n")
            f.write(
                f"**Total Processing Time:** {metrics['total_time']:.2f} seconds\n\n"
            )

            f.write(f"## Text Processing Statistics\n\n")
            f.write(f"| Metric | Original | Processed | Change |\n")
            f.write(f"|--------|----------|-----------|--------|\n")
            f.write(
                f"| Characters | {input_text['char_count']:,} | {processed_text['char_count']:,} | {processed_text['char_diff']:+,} ({processed_text['char_diff_pct']:.1f}%) |\n"
            )
            f.write(
                f"| Words | {input_text['word_count']:,} | {processed_text['word_count']:,} | {processed_text['word_count'] - input_text['word_count']:+,} |\n"
            )
            f.write(
                f"| Paragraphs | {input_text['paragraph_count']:,} | {processed_text['paragraph_count']:,} | {processed_text['paragraph_count'] - input_text['paragraph_count']:+,} |\n\n"
            )

            f.write(f"## Audio Output Statistics\n\n")
            f.write(f"| Metric | Value |\n")
            f.write(f"|--------|-------|\n")
            f.write(
                f"| Duration | {audio_metadata['audio_duration_formatted']} ({audio_metadata['audio_duration']:.1f}s) |\n"
            )
            f.write(f"| File Size | {audio_metadata['file_size_mb']:.2f} MB |\n")
            f.write(f"| Voice | {voice_settings['voice']} |\n")
            f.write(f"| Speech Rate | {voice_settings['speech_rate']}x |\n")
            f.write(f"| Format | {voice_settings['output_format'].upper()} |\n")
            f.write(f"| Chars/Second | {audio_metadata['chars_per_second']:.1f} |\n")
            f.write(
                f"| Words/Minute | {(processed_text['word_count'] / audio_metadata['audio_duration']) * 60:.1f} |\n\n"
            )

            f.write(f"## Generated Output Files\n\n")
            f.write(f"| File Type | Description | Path | Size |\n")
            f.write(f"|-----------|-------------|------|------|\n")
            f.write(
                f"| Original Text | Raw input text | {original_text_path} | {original_size / 1024:.1f} KB |\n"
            )
            f.write(
                f"| Processed Text | TTS-optimized text | {processed_text_path} | {processed_size / 1024:.1f} KB |\n"
            )
            f.write(
                f"| Audio Output | {voice_settings['voice']} voice at {voice_settings['speech_rate']}x | {audio_output_path} | {audio_metadata['file_size_mb']:.2f} MB |\n"
            )

        # Mention the report file
        console.print(
            f"[dim]A detailed report has been saved to: [cyan]{report_path}[/cyan][/dim]"
        )


def main():
    """Main function to parse arguments and run the processor."""
    # Define console at the start to avoid UnboundLocalError
    console = Console()

    # Create a more sophisticated argument parser
    parser = argparse.ArgumentParser(
        description="Text-to-Speech Processor with AI preprocessing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tts_processor.py myfile.txt
  python tts_processor.py --voice onyx-hd --rate 1.2 --format mp3 myfile.txt
  python tts_processor.py --non-interactive document.txt
        """,
    )

    # Add main input file argument
    parser.add_argument("input_file", help="Path to the input text file")

    # Add option flags
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="Run in non-interactive mode with default settings or specified options",
    )

    # Add voice selection arguments
    voice_group = parser.add_argument_group("Voice Options")
    voice_group.add_argument(
        "--voice",
        choices=[
            "alloy",
            "echo",
            "fable",
            "onyx",
            "nova",
            "shimmer",
            "alloy-hd",
            "echo-hd",
            "fable-hd",
            "onyx-hd",
            "nova-hd",
            "shimmer-hd",
        ],
        help="Specify the voice to use (overrides interactive selection)",
    )

    voice_group.add_argument(
        "--rate",
        type=float,
        choices=[0.8, 0.9, 1.0, 1.1, 1.2, 1.5],
        help="Speech rate multiplier (0.8-1.5x)",
    )

    voice_group.add_argument(
        "--format", choices=["mp3", "wav"], help="Audio output format"
    )

    # Add advanced options
    advanced_group = parser.add_argument_group("Advanced Options")
    advanced_group.add_argument(
        "--skip-preprocessing",
        action="store_true",
        help="Skip Claude preprocessing step and use raw text",
    )
    advanced_group.add_argument(
        "--no-cache",
        action="store_true",
        help="Ignore cached results and force reprocessing",
    )
    advanced_group.add_argument(
        "--output-dir", help="Specify custom output directory (default: auto-generated)"
    )

    try:
        args = parser.parse_args()
    except SystemExit:
        # If argument parsing fails, print help message with colorful formatting
        console.print(
            Panel(
                "[bold]Text-to-Speech Processor[/bold]\n\n"
                "This script converts text files to speech using Claude for preprocessing\n"
                "and OpenAI's Text-to-Speech API for high-quality audio generation.\n\n"
                "[bold]Key Features:[/bold]\n"
                "• AI text preprocessing to optimize for natural speech\n"
                "• Professional voices with speed adjustment\n"
                "• Intelligent text chunking for optimal speech quality\n"
                "• Detailed progress tracking and reporting\n\n"
                "[bold]Basic Usage:[/bold]\n"
                "  python tts_processor.py [input_file]\n\n"
                "[bold]Advanced Usage:[/bold]\n"
                "  python tts_processor.py --voice onyx-hd --rate 1.2 --format mp3 myfile.txt",
                title="🎙️ TTS Processor Help",
                border_style="blue",
                padding=(1, 2),
            )
        )
        return 1

    try:
        # Check if API keys are available in environment for non-interactive mode
        if args.non_interactive:
            if not os.environ.get("ANTHROPIC_API_KEY"):
                console.print(
                    "[bold red]Error:[/bold red] ANTHROPIC_API_KEY must be set in environment for non-interactive mode"
                )
                return 1
            if not os.environ.get("OPENAI_API_KEY"):
                console.print(
                    "[bold red]Error:[/bold red] OPENAI_API_KEY must be set in environment for non-interactive mode"
                )
                return 1

            # If using non-interactive mode with specific voice options, validate them
            if args.voice and args.voice not in [
                "alloy",
                "echo",
                "fable",
                "onyx",
                "nova",
                "shimmer",
                "alloy-hd",
                "echo-hd",
                "fable-hd",
                "onyx-hd",
                "nova-hd",
                "shimmer-hd",
            ]:
                console.print(
                    f"[bold red]Error:[/bold red] Invalid voice option: {args.voice}"
                )
                return 1

            if args.rate and (args.rate < 0.8 or args.rate > 1.5):
                console.print(
                    f"[bold red]Error:[/bold red] Speech rate must be between 0.8 and 1.5"
                )
                return 1

        # Create processor instance
        processor = TextToSpeechProcessor()

        # TODO: Implement advanced options handling (custom output dir, skip preprocessing, etc.)
        # This would require modifying the process_file method to accept these options
        # For now, we'll just pass the non-interactive flag

        processor.process_file(args.input_file, non_interactive=args.non_interactive)
    except KeyboardInterrupt:
        console.print("\n[bold red]Operation cancelled by user[/bold red]")
    except EOFError:
        console.print(
            "\n[bold red]Error:[/bold red] Cannot get input in non-interactive mode. Please set API keys in environment."
        )
        return 1
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        logger.exception("Unexpected error")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
