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
from decimal import Decimal, ROUND_HALF_UP

# API pricing constants (as of March 2025)
# Claude pricing
CLAUDE_INPUT_PRICE_PER_MILLION = 15.00  # USD per 1M input tokens
CLAUDE_OUTPUT_PRICE_PER_MILLION = 75.00  # USD per 1M output tokens

# OpenAI TTS pricing
TTS_STANDARD_PRICE_PER_MILLION = 15.00  # USD per 1M characters
TTS_HD_PRICE_PER_MILLION = 30.00  # USD per 1M characters

# Token estimation constants
CHARS_PER_TOKEN_APPROX = 4  # Approximate number of characters per token
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

    def prompt_for_dummy_mode(self, non_interactive=False, cli_dummy=False) -> bool:
        """Prompt the user for whether to enable dummy mode or use CLI flag."""
        if non_interactive:
            # In non-interactive mode, use the CLI flag
            logger.info(f"Running in non-interactive mode, dummy mode: {cli_dummy}")
            return cli_dummy

        # Create a visually appealing selection panel for dummy mode
        console.print(
            Panel(
                "[bold]Dummy Mode[/bold]\n"
                "When enabled, content will be transformed into super-simple language for absolute beginners\n"
                "while preserving ALL original information.\n\n"
                "[cyan]Here's what Dummy Mode does:[/cyan]\n"
                "• Transforms ALL technical terms into everyday language\n"
                "• Rewrites everything at a middle-school reading level (ages 12-14)\n"
                "• Expands abbreviations like '200k' to '200,000 dollars'\n"
                "• Uses analogies and examples from everyday life\n"
                "• Adds clear explanations for complex concepts\n"
                "• Makes the tone more conversational and engaging\n\n"
                "[bold yellow]Perfect for:[/bold yellow]\n"
                "• Beginners with NO background in the subject\n"
                "• Non-native English speakers\n"
                "• Making complex topics accessible to everyone\n"
                "• People who prefer plain, everyday language",
                title="🧠 Simplification Options",
                border_style="cyan",
                padding=(1, 2),
            )
        )

        # Show a table explaining dummy mode
        dummy_table = Table(
            show_header=True, header_style="bold magenta", box=box.ROUNDED
        )
        dummy_table.add_column("Option", style="dim", justify="center", width=8)
        dummy_table.add_column("Mode", style="cyan", width=15)
        dummy_table.add_column("Description", style="green")
        dummy_table.add_column("Reading Level", style="yellow", justify="center")

        dummy_table.add_row(
            "1",
            "Standard Mode",
            "Maintains original language while optimizing for speech",
            "College/Professional",
        )
        dummy_table.add_row(
            "2",
            "Dummy Mode",
            "Rewrites everything in super-simple language a 13-year-old could understand",
            "Middle School",
        )

        console.print(dummy_table)

        # If the CLI flag was set, use it as the default selection
        default_choice = "2" if cli_dummy else "1"

        dummy_choice = ""
        while dummy_choice not in ["1", "2"]:
            dummy_choice = (
                console.input(
                    f"\n[bold yellow]Select processing mode (1/2) [{default_choice}]: [/bold yellow]"
                )
                or default_choice
            )

        dummy_mode = dummy_choice == "2"

        # Show confirmation of selection
        mode_name = "Dummy Mode" if dummy_mode else "Standard Mode"
        console.print(f"\n[bold green]Selected:[/bold green] [cyan]{mode_name}[/cyan]")

        return dummy_mode

    def preprocess_text(self, input_text: str, dummy_mode=False) -> Tuple[str, Dict]:
        """Preprocess text using Claude to optimize it for speech."""
        # Calculate input text metrics
        char_count = len(input_text)
        word_count = len(input_text.split())
        paragraph_count = len(input_text.split("\n\n"))
        estimated_input_tokens = max(1, char_count // CHARS_PER_TOKEN_APPROX)
        # Set the processing mode name
        mode_name = "Dummy Mode" if dummy_mode else "Standard Mode"
        
        console.print(
            Panel(
                f"[cyan]Claude 3.7 will now preprocess the text in {mode_name} to optimize it for speech while preserving all important content.[/cyan]",
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
                f"Preprocessing text with Claude 3.7 in {mode_name}: {char_count} characters, {word_count} words"
            )

            # Create a very specific prompt for Claude based on the mode
            if dummy_mode:
                system_prompt = """
                You are a text preprocessing expert tasked with preparing text for text-to-speech conversion for beginners with NO specialized knowledge.

                CRITICAL INSTRUCTIONS - READ CAREFULLY:
                1. MAINTAIN COMPLETE CONTENT: Preserve ALL information, arguments, data points, examples, and insights from the original text, but express them in much simpler language.

                2. TARGET LANGUAGE LEVEL: Write at a middle school (7th-8th grade) reading level. Use everyday language that a 13-year-old would understand.

                3. SIMPLIFY VOCABULARY AGGRESSIVELY:
                   - Replace ALL advanced vocabulary with everyday words
                   - EXAMPLES:
                     * "depreciation" → "wear and tear tax deduction"
                     * "amortization" → "spreading costs over time"
                     * "passive income" → "money you earn without actively working for it"
                     * "tax liability" → "the amount of tax you owe"
                     * "qualified intermediary" → "a special third-party person"
                   - ANY word that wouldn't appear in everyday conversation should be replaced or clearly explained

                4. EXPLAIN ALL TECHNICAL TERMS: When you must use a technical term, explain it in parentheses using extremely simple language. For example:
                   - "Section 1031 exchange (a special rule that lets you swap one investment property for another without paying taxes right away)"
                   - "limited partnership (a business where some owners have control but limited responsibility for debts, while others have no control but are only responsible for the money they put in)"

                5. FORMAT NUMBERS FOR SPEECH:
                   - ALWAYS expand abbreviations: "200k" → "200,000 dollars" or "200,000"
                   - Add "dollars" or "percent" when referring to currency or percentages
                   - Write numbers in full: "5M" → "5 million"
                   - For large numbers, use both the number and a relatable comparison: "$500,000" → "500,000 dollars (about half a million dollars)"
                   - For time periods, convert to everyday equivalents: "750 hours a year" → "about 2 hours every day, which adds up to 750 hours a year"

                6. BREAK DOWN COMPLEX CONCEPTS:
                   - Use simple analogies from everyday life
                   - Break multi-step processes into numbered steps
                   - Use "in other words" or "this means" to clarify difficult ideas
                   - Explain cause and effect relationships with "because" and "so"
                   - Use concrete examples for abstract concepts

                7. USE CONVERSATIONAL LANGUAGE:
                   - Write as if you're explaining to a friend who knows nothing about the topic
                   - Use contractions (don't, can't, it's)
                   - Use active voice instead of passive voice
                   - Address the listener directly with "you" and "your"
                   - Add transitional phrases like "Let's talk about..." or "Now let's move on to..."

                8. MAINTAIN STRUCTURE:
                   - Keep all sections and subsections
                   - Keep all examples, just simplify their explanation
                   - Don't skip any details, just express them more simply

                9. ADD HELPFUL CONTEXT AND TRANSITIONS:
                   - Begin sections with context: "Now we're going to talk about X, which is important because..."
                   - Remind listeners of related concepts: "Remember when we talked about X earlier? This is similar because..."
                   - Preview complex ideas: "I'm going to explain X, which might sound complicated, but I'll break it down step by step."

                10. TEST YOUR WORK: For each paragraph, ask "Would a 13-year-old understand this?" If not, simplify further.

                Return ONLY the processed text with no explanations, annotations, or meta-commentary.
                """
            else:
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

                # Check if the processed text is significantly shorter than the original
                processed_char_count = len(processed_text)
                processed_word_count = len(processed_text.split())
                processed_paragraph_count = len(processed_text.split("\n\n"))

                # Calculate percentage change
                char_percent_change = (
                    (processed_char_count - char_count) / char_count * 100
                )

                # If in dummy mode and the text has been shortened by more than 15%, regenerate with stronger preservation instructions
                if dummy_mode and char_percent_change < -15:
                    logger.warning(
                        f"Processed text is {abs(char_percent_change):.1f}% shorter than original. Regenerating with stronger preservation instructions."
                    )

                    # Use an enhanced fallback prompt that forces extreme simplification
                    preservation_system_prompt = """
                    You are a text simplification expert for ABSOLUTE BEGINNERS.

                    EMERGENCY OVERRIDE: Previous processing didn't simplify language enough.

                    YOUR JOB IS TO:
                    1. KEEP 100% OF THE ORIGINAL INFORMATION
                    2. REWRITE EVERYTHING AT A 7TH GRADE LEVEL (age 12-13)

                    SPECIFIC REQUIREMENTS:

                    1. VOCABULARY:
                       - Use ONLY words a middle-school student would know
                       - Any word that wouldn't appear in a children's book needs explanation
                       - Example before: "This depreciation recapture is taxed at 25%"
                       - Example after: "When you sell, you pay back some tax breaks at 25 percent. This is called 'depreciation recapture' (which means paying back the tax savings you got earlier)"

                    2. NUMBERS:
                       - ALWAYS write numbers in full: "40k" → "40,000"
                       - Add "dollars" after money amounts: "$500k" → "500,000 dollars"
                       - Use "percent" not "%": "25%" → "25 percent"
                       - Add comparisons for large numbers: "$2.5M" → "2.5 million dollars (that's 2,500,000 dollars)"

                    3. SENTENCE STRUCTURE:
                       - Use VERY SHORT sentences (15-20 words maximum)
                       - Break up any complex sentence into multiple simpler ones
                       - Use active voice: "You can deduct this" NOT "This can be deducted"
                       - Use conversational phrasing like "Let's talk about..." and "This means..."

                    4. EXPLANATIONS:
                       - For EVERY technical term, add a simple explanation in parentheses
                       - Use everyday analogies: "Depreciation is like how your car loses value as it gets older"
                       - After complex ideas, add "In simpler terms, this means..."
                       - For tax concepts, add real examples with small, relatable numbers

                    5. STRUCTURE:
                       - Keep ALL original content, just express it much more simply
                       - Add transition sentences between topics
                       - Break complex explanations into numbered steps

                    EXAMPLES OF PROPER TRANSFORMATIONS:

                    ORIGINAL: "The marginal tax rate progression affects high-income taxpayers disproportionately."
                    SIMPLIFIED: "As you make more money, your tax rate goes up. This hits people who make a lot of money harder. It's like how a movie theater might charge kids 5 dollars but adults 10 dollars - the people with more money pay a higher rate."

                    ORIGINAL: "REPS qualification requires material participation exceeding 750 hours annually."
                    SIMPLIFIED: "To qualify as a Real Estate Professional (that's a special tax status), you need to work more than 750 hours a year in real estate. That's about 14 hours every week. And you need to be 'materially participating' (which means actively working, not just investing)."

                    DO NOT REMOVE ANY INFORMATION OR EXAMPLES - JUST MAKE THEM SIMPLER.

                    Return ONLY the processed text with no explanations, annotations, or meta-commentary.
                    """

                    logger.info(
                        "Attempting secondary processing with stronger preservation instructions"
                    )

                    try:
                        # Create a secondary processing stream
                        secondary_text = ""

                        with self.anthropic_client.messages.stream(
                            model=CLAUDE_MODEL,
                            system=preservation_system_prompt,
                            messages=[
                                {
                                    "role": "user",
                                    "content": f"Process the following text for text-to-speech conversion according to the instructions, maintaining ALL content and details. Here's the text:\n\n{input_text}",
                                }
                            ],
                            max_tokens=64000,
                        ) as stream:
                            # Update every second or so to avoid too many refreshes
                            last_update = time.time()

                            # Reset progress bar for secondary processing
                            progress.update(
                                total_task,
                                completed=0,
                                description=f"REGENERATING: Ensuring full content preservation",
                            )

                            for text in stream.text_stream:
                                secondary_text += text
                                received_chars = len(secondary_text)

                                # Only update the display about once per second to avoid flickering
                                current_time = time.time()
                                if current_time - last_update >= 0.5:
                                    progress.update(
                                        total_task,
                                        completed=min(received_chars, char_count),
                                        description=f"REGENERATING: {received_chars:,}/{char_count:,} characters",
                                    )
                                    last_update = current_time

                        # Check if secondary processing produced better results
                        secondary_char_count = len(secondary_text)
                        secondary_percent_change = (
                            (secondary_char_count - char_count) / char_count * 100
                        )

                        logger.info(
                            f"Secondary processing produced text {secondary_percent_change:.1f}% of original length"
                        )

                        # If secondary processing resulted in better content preservation, use it
                        if secondary_percent_change > char_percent_change:
                            logger.info(
                                "Using secondary processed text with better content preservation"
                            )
                            processed_text = secondary_text
                            processed_char_count = secondary_char_count
                            processed_word_count = len(processed_text.split())
                            processed_paragraph_count = len(
                                processed_text.split("\n\n")
                            )
                        else:
                            logger.info(
                                "Secondary processing did not improve content preservation, using original result"
                            )

                    except Exception as e:
                        logger.error(f"Error during secondary text preprocessing: {e}")
                        # Continue with the original processed text

                # Calculate and display change metrics
                char_diff = processed_char_count - char_count
                char_pct_change = (
                    (char_diff / char_count * 100) if char_count > 0 else 0
                )
                word_diff = processed_word_count - word_count
                word_pct_change = (
                    (word_diff / word_count * 100) if word_count > 0 else 0
                )

                # Calculate token counts and costs
                estimated_output_tokens = max(
                    1, processed_char_count // CHARS_PER_TOKEN_APPROX
                )

                # Calculate costs
                claude_input_cost = (
                    estimated_input_tokens / 1_000_000
                ) * CLAUDE_INPUT_PRICE_PER_MILLION
                claude_output_cost = (
                    estimated_output_tokens / 1_000_000
                ) * CLAUDE_OUTPUT_PRICE_PER_MILLION
                claude_total_cost = claude_input_cost + claude_output_cost

                # Round to 4 decimal places for display
                claude_input_cost_display = Decimal(str(claude_input_cost)).quantize(
                    Decimal("0.0001"), rounding=ROUND_HALF_UP
                )
                claude_output_cost_display = Decimal(str(claude_output_cost)).quantize(
                    Decimal("0.0001"), rounding=ROUND_HALF_UP
                )
                claude_total_cost_display = Decimal(str(claude_total_cost)).quantize(
                    Decimal("0.0001"), rounding=ROUND_HALF_UP
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
                stats_table.add_row(
                    "Est. Input Tokens",
                    f"{estimated_input_tokens:,}",
                    "",
                    "",
                )
                stats_table.add_row(
                    "Est. Output Tokens",
                    "",
                    f"{estimated_output_tokens:,}",
                    "",
                )

                console.print(stats_table)

                # Create pricing table
                pricing_table = Table(
                    title="Claude API Usage Costs", show_header=True, box=box.ROUNDED
                )
                pricing_table.add_column("Cost Item", style="cyan")
                pricing_table.add_column(
                    "Rate (per 1M)", style="yellow", justify="right"
                )
                pricing_table.add_column(
                    "Estimated Cost", style="green", justify="right"
                )

                pricing_table.add_row(
                    "Input Tokens",
                    f"${CLAUDE_INPUT_PRICE_PER_MILLION:.2f}",
                    f"${claude_input_cost_display}",
                )
                pricing_table.add_row(
                    "Output Tokens",
                    f"${CLAUDE_OUTPUT_PRICE_PER_MILLION:.2f}",
                    f"${claude_output_cost_display}",
                )
                pricing_table.add_row(
                    "Total Claude Cost", "", f"${claude_total_cost_display}"
                )

                console.print(pricing_table)

                # Display a summary of what was done
                elapsed = time.time() - start_time
                console.print(
                    f"[dim]• Processing Speed: {received_chars / max(1, elapsed):.1f} chars/sec\n"
                    f"• Time: {elapsed:.1f}s total[/dim]"
                )

                mode_name = "Dummy Mode" if dummy_mode else "Standard Mode"
                logger.info(
                    f"Text preprocessing complete in {mode_name}. Original: {char_count} chars, Processed: {processed_char_count} chars."
                )

                # Compare length to ensure we didn't lose too much content
                retention_rate = (
                    processed_word_count / word_count * 100 if word_count > 0 else 100
                )

                if retention_rate < 85:
                    console.print(
                        f"[bold yellow]Warning:[/bold yellow] Processed text is {retention_rate:.1f}% of the original length. Some content may have been lost."
                    )

                # Create usage metrics dictionary to return
                usage_metrics = {
                    "input_tokens": estimated_input_tokens,
                    "output_tokens": estimated_output_tokens,
                    "input_cost": float(claude_input_cost_display),
                    "output_cost": float(claude_output_cost_display),
                    "total_cost": float(claude_total_cost_display),
                }

                return processed_text, usage_metrics

            except Exception as e:
                logger.error(f"Error during text preprocessing: {e}")
                # Create empty metrics to avoid unpacking errors
                empty_metrics = {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "input_cost": 0.0,
                    "output_cost": 0.0,
                    "total_cost": 0.0,
                }
                # Return empty result on error to prevent unpacking errors
                return "Error preprocessing text: " + str(e), empty_metrics

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
        is_hd = "-hd" in voice
        model = "tts-1-hd" if is_hd else "tts-1"
        # For HD voices, remove the -hd suffix for the actual API call
        api_voice = voice.replace("-hd", "") if is_hd else voice

        # Determine the price per million characters based on model
        price_per_million = (
            TTS_HD_PRICE_PER_MILLION if is_hd else TTS_STANDARD_PRICE_PER_MILLION
        )

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

                # Calculate TTS API cost
                tts_cost = (char_count / 1_000_000) * price_per_million
                tts_cost_display = Decimal(str(tts_cost)).quantize(
                    Decimal("0.0001"), rounding=ROUND_HALF_UP
                )
                cost_per_minute = (
                    tts_cost / (audio_duration / 60) if audio_duration > 0 else 0
                )
                cost_per_minute_display = Decimal(str(cost_per_minute)).quantize(
                    Decimal("0.0001"), rounding=ROUND_HALF_UP
                )

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
                    "model": model,
                    "is_hd": is_hd,
                    "price_per_million": price_per_million,
                    "tts_cost": float(tts_cost_display),
                    "cost_per_minute": float(cost_per_minute_display),
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
                f"[cyan]• TTS Model:[/cyan] {model}\n"
                f"[cyan]• Estimated Cost:[/cyan] ${metadata['tts_cost']:.4f}\n"
                f"[dim]• Processing Time: {total_time:.1f}s ({metadata['chars_per_second']:.1f} chars/sec)[/dim]",
                title="Audio Generated Successfully",
                border_style="green",
            )
        )

        # Display pricing information
        pricing_table = Table(
            title="OpenAI TTS API Usage Costs", show_header=True, box=box.ROUNDED
        )
        pricing_table.add_column("Cost Item", style="cyan")
        pricing_table.add_column("Rate", style="yellow", justify="right")
        pricing_table.add_column("Usage", style="blue", justify="right")
        pricing_table.add_column("Estimated Cost", style="green", justify="right")

        pricing_table.add_row(
            f"{model} Character Processing",
            f"${price_per_million:.2f} per 1M chars",
            f"{char_count:,} chars",
            f"${metadata['tts_cost']:.4f}",
        )
        pricing_table.add_row(
            "Cost per Minute of Audio",
            "",
            f"{metadata['audio_duration'] / 60:.2f} minutes",
            f"${metadata['cost_per_minute']:.4f}/min",
        )

        console.print(pricing_table)

        logger.info(f"Audio saved to {output_file}")
        return output_file, metadata

    def process_file(
        self, input_file: str, non_interactive=False, dummy_mode=False
    ) -> None:
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

        # Show info about dummy mode
        if dummy_mode:
            console.print(
                Panel(
                    "[bold green]Dummy Mode is ENABLED[/bold green]\n"
                    "[cyan]Text will be completely rewritten at a middle-school reading level.[/cyan]\n"
                    "• ALL technical terms will be simplified or clearly explained\n"
                    "• Number formats will be expanded (e.g., '200k' → '200,000 dollars')\n"
                    "• Complex concepts will be broken down with everyday analogies\n"
                    "• Content will use simple vocabulary a 13-year-old would understand\n"
                    "• All original information will be preserved in simpler language",
                    title="🧠 Dummy Mode Information",
                    border_style="cyan",
                    padding=(1, 2),
                )
            )

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

        # Prompt for dummy mode
        is_dummy_mode = self.prompt_for_dummy_mode(non_interactive, dummy_mode)

        # Save dummy mode setting to metrics
        process_metrics["dummy_mode"] = is_dummy_mode

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
                f"[cyan]Claude 3.7 will now preprocess the text {'in Dummy Mode' if is_dummy_mode else ''} to optimize it for speech while preserving all important content.[/cyan]",
                title="Step 2: Text Preprocessing",
                border_style="green",
                padding=(1, 2),
            )
        )

        processed_text, claude_usage_metrics = self.preprocess_text(
            input_text, is_dummy_mode
        )

        # Save processed text metrics and Claude usage metrics
        process_metrics["processed_text"] = {
            "char_count": len(processed_text),
            "word_count": len(processed_text.split()),
            "paragraph_count": len(processed_text.split("\n\n")),
            "char_diff": len(processed_text) - char_count,
            "char_diff_pct": (len(processed_text) - char_count)
            / max(1, char_count)
            * 100,
        }

        # Add Claude usage metrics
        process_metrics["claude_usage"] = claude_usage_metrics

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
        processed_text_path = (
            output_dir / f"{base_name}_processed{'_dummy' if is_dummy_mode else ''}.txt"
        )

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

        # Add dummy mode indicator to the filename if enabled
        dummy_indicator = "_dummy" if is_dummy_mode else ""
        audio_output_path = (
            output_dir
            / f"{base_name}_{language_code}_{voice}_{speech_rate}x{dummy_indicator}.{output_format}"
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
            "dummy_mode": is_dummy_mode,
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

        # Calculate total cost
        claude_cost = metrics["claude_usage"]["total_cost"]
        tts_cost = metrics["audio_metadata"]["tts_cost"]
        total_cost = claude_cost + tts_cost

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
        dummy_mode = metrics.get("dummy_mode", False)

        audio_stats.add_row(
            "Duration",
            f"{audio_metadata['audio_duration_formatted']} ({audio_metadata['audio_duration']:.1f}s)",
        )
        audio_stats.add_row("File Size", f"{audio_metadata['file_size_mb']:.2f} MB")
        audio_stats.add_row("Voice", f"{voice_settings['voice']}")
        audio_stats.add_row("Speech Rate", f"{voice_settings['speech_rate']}x")
        audio_stats.add_row("Format", f"{voice_settings['output_format'].upper()}")
        audio_stats.add_row("Dummy Mode", f"{'Enabled' if dummy_mode else 'Disabled'}")
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

        # Add dummy mode indicator to the processed text description if enabled
        processed_description = "Text optimized for speech synthesis"
        if dummy_mode:
            processed_description += " (simplified for beginners)"

        files_table.add_row(
            "Processed Text",
            processed_description,
            str(processed_text_path),
            f"{processed_size / 1024:.1f} KB",
        )

        # Add dummy mode indicator to the audio description if enabled
        audio_description = (
            f"{voice_settings['voice']} voice at {voice_settings['speech_rate']}x speed"
        )
        if dummy_mode:
            audio_description += " (simplified content)"

        files_table.add_row(
            "Audio Output",
            audio_description,
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

        # Create pricing summary table
        pricing_summary = Table(
            title="💰 Cost Summary", show_header=True, box=box.ROUNDED
        )
        pricing_summary.add_column("Service", style="cyan")
        pricing_summary.add_column("Usage", style="yellow")
        pricing_summary.add_column("Cost", style="green", justify="right")

        # Claude costs
        claude_usage = metrics["claude_usage"]
        pricing_summary.add_row(
            "Claude 3.7 Sonnet (Input)",
            f"{claude_usage['input_tokens']:,} tokens",
            f"${claude_usage['input_cost']:.4f}",
        )
        pricing_summary.add_row(
            "Claude 3.7 Sonnet (Output)",
            f"{claude_usage['output_tokens']:,} tokens",
            f"${claude_usage['output_cost']:.4f}",
        )

        # TTS costs
        audio_metadata = metrics["audio_metadata"]
        tts_model = audio_metadata["model"]
        pricing_summary.add_row(
            f"OpenAI {tts_model}",
            f"{audio_metadata['char_count']:,} characters",
            f"${audio_metadata['tts_cost']:.4f}",
        )

        # Total cost
        pricing_summary.add_row("Total Cost", "", f"${total_cost:.4f}")

        # Cost per minute of audio
        audio_minutes = audio_metadata["audio_duration"] / 60
        cost_per_minute = total_cost / audio_minutes if audio_minutes > 0 else 0
        pricing_summary.add_row(
            "Cost per Minute of Audio",
            f"{audio_minutes:.2f} minutes",
            f"${cost_per_minute:.4f}/min",
        )

        console.print(pricing_summary)

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
            f.write(f"| Dummy Mode | {'Enabled' if dummy_mode else 'Disabled'} |\n")
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
                f"| Processed Text | {processed_description} | {processed_text_path} | {processed_size / 1024:.1f} KB |\n"
            )
            f.write(
                f"| Audio Output | {audio_description} | {audio_output_path} | {audio_metadata['file_size_mb']:.2f} MB |\n"
            )

            # Add pricing information to the report
            f.write(f"\n## Cost Summary\n\n")
            f.write(f"| Service | Usage | Cost |\n")
            f.write(f"|---------|-------|------|\n")
            f.write(
                f"| Claude 3.7 Sonnet (Input) | {claude_usage['input_tokens']:,} tokens | ${claude_usage['input_cost']:.4f} |\n"
            )
            f.write(
                f"| Claude 3.7 Sonnet (Output) | {claude_usage['output_tokens']:,} tokens | ${claude_usage['output_cost']:.4f} |\n"
            )
            f.write(
                f"| OpenAI {tts_model} | {audio_metadata['char_count']:,} characters | ${audio_metadata['tts_cost']:.4f} |\n"
            )
            f.write(f"| **Total Cost** | | **${total_cost:.4f}** |\n")
            f.write(
                f"| Cost per Minute of Audio | {audio_minutes:.2f} minutes | ${cost_per_minute:.4f}/min |\n"
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
  python tts_processor.py --dummy myfile.txt
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

    # Add dummy mode flag
    parser.add_argument(
        "--dummy",
        action="store_true",
        help="Enable dummy mode to simplify language and explain technical terms for beginners",
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
                "• Detailed progress tracking and reporting\n"
                "• Dummy mode for simplified, beginner-friendly content\n\n"
                "[bold]Basic Usage:[/bold]\n"
                "  python tts_processor.py [input_file]\n\n"
                "[bold]Advanced Usage:[/bold]\n"
                "  python tts_processor.py --voice onyx-hd --rate 1.2 --format mp3 --dummy myfile.txt",
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
        # For now, we'll just pass the non-interactive flag and dummy mode flag

        processor.process_file(
            args.input_file, non_interactive=args.non_interactive, dummy_mode=args.dummy
        )
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
