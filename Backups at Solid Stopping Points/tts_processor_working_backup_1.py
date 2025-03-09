#!/usr/bin/env python3
import os
import sys
import argparse
import logging
import time
import json
import hashlib
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import anthropic
import openai
import inquirer
from rich.console import Console

# IMPORTANT: This script REQUIRES Claude 3.7 Sonnet specifically.
# NO OTHER CLAUDE MODEL VERSIONS should be used under any circumstances.
# Claude 3.7 Sonnet offers a context window of 200,000 tokens and supports
# output up to 128,000 tokens per request (64,000 tokens generally available,
# 64,000-128,000 tokens in beta).
CLAUDE_MODEL = "claude-3-7-sonnet-20250219"  # ONLY use Claude 3.7 Sonnet
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskID
from rich.markdown import Markdown
from rich.table import Table
from rich.text import Text
from rich.prompt import Confirm
from rich.layout import Layout
from rich.live import Live
from pydub import AudioSegment

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

        # Voice descriptions to help users choose
        self.voice_descriptions = {
            "alloy": "Versatile, balanced voice with a neutral tone",
            "echo": "Soft-spoken and clear with a gentle delivery",
            "fable": "Whimsical and expressive, good for storytelling",
            "onyx": "Authoritative and deep, good for narration",
            "nova": "Warm and pleasant with a friendly tone",
            "shimmer": "Bright and melodic with an upbeat quality",
            "alloy-hd": "High-definition version of Alloy - versatile, balanced voice",
            "echo-hd": "High-definition version of Echo - soft-spoken and clear",
            "fable-hd": "High-definition version of Fable - whimsical and expressive",
            "onyx-hd": "High-definition version of Onyx - authoritative and deep",
            "nova-hd": "High-definition version of Nova - warm and pleasant",
            "shimmer-hd": "High-definition version of Shimmer - bright and melodic",
        }

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

    def display_voice_options(self, non_interactive=False) -> Tuple[str, str]:
        """Display voice options in a table and let user select or use defaults."""
        if non_interactive:
            # Use default options for non-interactive mode
            logger.info("Running in non-interactive mode, using default voice: onyx-hd and mp3 format")
            return "onyx-hd", "mp3"
            
        # First select voice quality
        console.print("\n[bold cyan]Voice Quality Options:[/bold cyan]")

        quality_table = Table(show_header=True, header_style="bold magenta")
        quality_table.add_column("Option", style="dim")
        quality_table.add_column("Quality", style="cyan")
        quality_table.add_column("Description", style="green")

        quality_table.add_row("1", "Standard Voices", "Regular quality, less expensive")
        quality_table.add_row("2", "HD Voices", "Higher quality, more natural sounding")

        console.print(quality_table)

        quality_choice = ""
        while quality_choice not in ["1", "2"]:
            quality_choice = console.input(
                "\n[bold yellow]Select voice quality (1/2): [/bold yellow]"
            )

        voice_category = "Standard voices" if quality_choice == "1" else "HD voices"

        # Then select specific voice
        console.print(f"\n[bold cyan]Available {voice_category}:[/bold cyan]")

        voice_table = Table(show_header=True, header_style="bold magenta")
        voice_table.add_column("Option", style="dim")
        voice_table.add_column("Voice", style="cyan")
        voice_table.add_column("Description", style="green")

        voices = self.voice_options[voice_category]
        for i, voice in enumerate(voices, 1):
            voice_table.add_row(
                str(i),
                voice,
                self.voice_descriptions.get(voice, "No description available"),
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

        # Finally select format
        console.print("\n[bold cyan]Output Format Options:[/bold cyan]")

        format_table = Table(show_header=True, header_style="bold magenta")
        format_table.add_column("Option", style="dim")
        format_table.add_column("Format", style="cyan")
        format_table.add_column("Description", style="green")

        format_table.add_row(
            "1", "MP3", "Standard compressed audio format, smaller file size"
        )
        format_table.add_row("2", "WAV", "Uncompressed audio format, higher quality")

        console.print(format_table)

        format_choice = ""
        while format_choice not in ["1", "2"]:
            format_choice = console.input(
                "\n[bold yellow]Select output format (1/2): [/bold yellow]"
            )

        output_format = "mp3" if format_choice == "1" else "wav"

        return selected_voice, output_format

    def preprocess_text(self, input_text: str) -> str:
        """Use Claude 3.7 to preprocess text for speech readability."""
        with Progress(
            SpinnerColumn(),
            TextColumn(
                "[bold green]Preprocessing text with Claude 3.7...[/bold green]"
            ),
            console=console,
        ) as progress:
            task = progress.add_task("", total=None)
            logger.info("Preprocessing text with Claude 3.7...")

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
                    for text in stream.text_stream:
                        processed_text += text
                        progress.update(task, description="[bold green]Receiving Claude's response...[/bold green]")

                logger.info("Text preprocessing complete")

                # Compare length to ensure we didn't lose too much content
                original_length = len(input_text.split())
                processed_length = len(processed_text.split())
                retention_rate = processed_length / original_length * 100

                progress.update(task, completed=1, visible=False)

                if retention_rate < 85:
                    console.print(
                        f"[bold yellow]Warning:[/bold yellow] Processed text is {retention_rate:.1f}% of the original length. Some content may have been lost."
                    )

                return processed_text

            except Exception as e:
                logger.error(f"Error during text preprocessing: {e}")
                raise

    def generate_filename(self, text: str) -> str:
        """Ask Claude to generate a descriptive filename based on the text content."""
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold green]Generating descriptive filename...[/bold green]"),
            console=console,
        ) as progress:
            task = progress.add_task("", total=None)
            logger.info("Asking Claude to suggest a filename...")

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

                # Clean up filename to ensure it's valid
                filename = filename.strip()
                filename = re.sub(r"[^\w\-]", "_", filename)
                logger.info(f"Generated filename: {filename}")

                progress.update(task, completed=1, visible=False)
                return filename

            except Exception as e:
                logger.error(f"Error generating filename: {e}")
                # Fallback to a timestamp-based filename
                return f"tts_output_{int(time.time())}"

    def text_to_speech(self, text: str, voice: str, output_file: str) -> str:
        """Convert text to speech using OpenAI's TTS API."""
        # Determine if HD or standard voice
        model = "tts-1-hd" if "-hd" in voice else "tts-1"
        # For HD voices, remove the -hd suffix for the actual API call
        api_voice = voice.replace("-hd", "") if "-hd" in voice else voice

        # Process in chunks if text is very long
        max_chunk_size = 4000  # OpenAI has a limit on input text size

        # Count total chunks for progress reporting
        sentences = re.split(r"(?<=[.!?])\s+", text)
        chunks = []
        current_chunk = ""

        for sentence in sentences:
            if len(current_chunk) + len(sentence) + 1 <= max_chunk_size:
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
            else:
                chunks.append(current_chunk)
                current_chunk = sentence

        if current_chunk:
            chunks.append(current_chunk)

        total_chunks = len(chunks) if chunks else 1

        with Progress(
            SpinnerColumn(),
            TextColumn(f"[bold green]Converting to speech with {voice}...[/bold green]"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console,
        ) as progress:

            tts_task = progress.add_task(
                f"[cyan]Converting text using {model}...",
                total=total_chunks if chunks else 1,
            )
            logger.info(
                f"Converting text to speech using OpenAI's TTS API with voice: {voice}"
            )

            try:
                if not chunks:
                    # Process in one go if text is small enough
                    response = self.openai_client.audio.speech.create(
                        model=model, voice=api_voice, input=text
                    )

                    # Save the audio to file
                    response.stream_to_file(output_file)
                    progress.update(tts_task, advance=1)

                else:
                    # Process each chunk
                    temp_files = []
                    for i, chunk in enumerate(chunks):
                        progress.update(
                            tts_task,
                            description=f"[cyan]Processing chunk {i+1}/{total_chunks}...",
                        )
                        temp_file = f"{output_file}.part{i}"
                        response = self.openai_client.audio.speech.create(
                            model=model, voice=api_voice, input=chunk
                        )
                        response.stream_to_file(temp_file)
                        temp_files.append(temp_file)
                        progress.update(tts_task, advance=1)

                    # Combine all chunks using pydub
                    progress.update(
                        tts_task, description="[cyan]Combining audio chunks..."
                    )
                    combined = AudioSegment.empty()
                    for temp_file in temp_files:
                        segment = AudioSegment.from_file(temp_file)
                        combined += segment

                    # Export the combined audio
                    combined.export(output_file, format=output_file.split(".")[-1])

                    # Clean up temp files
                    for temp_file in temp_files:
                        os.remove(temp_file)

                logger.info(f"Audio saved to {output_file}")
                return output_file

            except Exception as e:
                logger.error(f"Error during text-to-speech conversion: {e}")
                raise

    def process_file(self, input_file: str, non_interactive=False) -> None:
        """Process a text file through the entire pipeline with user interaction."""
        # Show welcome banner
        console.print(
            Panel.fit(
                "[bold green]Text-to-Speech Processor[/bold green]\n"
                "[cyan]Convert text files to professional-sounding speech[/cyan]",
                title="🎙️ TTS Processor",
                border_style="blue",
            )
        )

        # Read input file
        input_path = Path(input_file)
        console.print(f"\n[bold]Processing:[/bold] [cyan]{input_path}[/cyan]")

        try:
            with open(input_file, "r", encoding="utf-8") as f:
                input_text = f.read()

            # Show text stats
            word_count = len(input_text.split())
            console.print(
                f"[dim]Text contains approximately {word_count:,} words[/dim]"
            )
        except Exception as e:
            console.print(f"[bold red]Error reading input file:[/bold red] {e}")
            return

        # Voice and format selection
        if not non_interactive:
            console.print(
                Panel(
                    "[cyan]Next, you'll select the voice and output format for your audio.[/cyan]",
                    title="Step 1: Voice Selection",
                    border_style="green",
                )
            )

        voice, output_format = self.display_voice_options(non_interactive)
        console.print(
            f"\n[bold green]Selected:[/bold green] [cyan]{voice}[/cyan] voice with [cyan]{output_format}[/cyan] format"
        )

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
            )
        )

        processed_text = self.preprocess_text(input_text)

        # Generate a descriptive filename/directory name
        console.print(
            Panel(
                "[cyan]Generating a descriptive name for output files based on content...[/cyan]",
                title="Step 3: Filename Generation",
                border_style="green",
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

        # Save all text versions
        console.print(
            Panel(
                "[cyan]Saving original and processed text files...[/cyan]",
                title="Step 4: Saving Text Files",
                border_style="green",
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
                f"[cyan]Converting text to speech using OpenAI's TTS API with {voice} voice...[/cyan]",
                title="Step 5: Text-to-Speech Conversion",
                border_style="green",
            )
        )

        audio_output_path = (
            output_dir / f"{base_name}_{language_code}_{voice}.{output_format}"
        )
        self.text_to_speech(processed_text, voice, str(audio_output_path))

        # Save metadata to cache
        metadata = {
            "timestamp": timestamp,
            "input_file": str(input_path),
            "output_dir": str(output_dir),
            "voice": voice,
            "format": output_format,
            "language": language_code,
        }
        self._save_to_cache(content_hash, metadata)

        # Display summary
        console.print(
            Panel(
                "[bold green]Text-to-Speech Processing Complete![/bold green]",
                border_style="green",
            )
        )

        table = Table(
            title="Generated Output Files",
            show_header=True,
            header_style="bold magenta",
        )
        table.add_column("File Type", style="cyan")
        table.add_column("Description", style="yellow")
        table.add_column("Path", style="green")

        table.add_row(
            "Original Text",
            "Raw input text without modifications",
            str(original_text_path),
        )
        table.add_row(
            "Processed Text",
            "Text optimized for speech synthesis",
            str(processed_text_path),
        )
        table.add_row(
            "Audio Output",
            f"{voice} voice in {output_format.upper()} format",
            str(audio_output_path),
        )

        console.print(table)
        console.print(
            f"\n[bold green]Output directory:[/bold green] [cyan]{output_dir}[/cyan]\n"
        )

        console.print(
            "[dim]The preprocessing helped optimize the text for speech while preserving content.[/dim]"
        )
        console.print(
            "[dim]You can compare the original and processed text files to see what was changed.[/dim]"
        )


def main():
    """Main function to parse arguments and run the processor."""
    # Define console at the start to avoid UnboundLocalError
    console = Console()
    
    parser = argparse.ArgumentParser(
        description="Text-to-Speech Processor with AI preprocessing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tts_processor.py myfile.txt
  python tts_processor.py ~/documents/report.txt
        """,
    )
    parser.add_argument("input_file", help="Path to the input text file")
    parser.add_argument("--non-interactive", action="store_true", 
                      help="Run in non-interactive mode (requires API keys to be set in environment)")

    try:
        args = parser.parse_args()
    except SystemExit:
        # If argument parsing fails, print help message with colorful formatting
        console.print(
            Panel(
                "[bold]Text-to-Speech Processor[/bold]\n\n"
                "This script converts text files to speech using Claude for preprocessing\n"
                "and OpenAI's Text-to-Speech API for high-quality audio generation.\n\n"
                "[bold]Usage:[/bold]\n"
                "  python tts_processor.py [input_file]\n\n"
                "[bold]Example:[/bold]\n"
                "  python tts_processor.py myfile.txt",
                title="🎙️ TTS Processor Help",
                border_style="blue",
            )
        )
        return 1

    try:
        # Check if API keys are available in environment for non-interactive mode
        if args.non_interactive:
            if not os.environ.get("ANTHROPIC_API_KEY"):
                console.print("[bold red]Error:[/bold red] ANTHROPIC_API_KEY must be set in environment for non-interactive mode")
                return 1
            if not os.environ.get("OPENAI_API_KEY"):
                console.print("[bold red]Error:[/bold red] OPENAI_API_KEY must be set in environment for non-interactive mode")
                return 1
        
        processor = TextToSpeechProcessor()
        processor.process_file(args.input_file, non_interactive=args.non_interactive)
    except KeyboardInterrupt:
        console.print("\n[bold red]Operation cancelled by user[/bold red]")
    except EOFError:
        console.print("\n[bold red]Error:[/bold red] Cannot get input in non-interactive mode. Please set API keys in environment.")
        return 1
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        logger.exception("Unexpected error")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
