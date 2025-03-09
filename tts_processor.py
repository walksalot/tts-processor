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
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import anthropic
import openai
import inquirer
print("Attempting to import Eleven Labs...")
try:
    import elevenlabs
    # The elevenlabs package structure is different than we thought
    # Don't try to import from elevenlabs.api as that doesn't exist
    ELEVENLABS_AVAILABLE = True
    print("Successfully imported Eleven Labs module!")
except ImportError as e:
    print(f"Error importing Eleven Labs module: {e}")
    ELEVENLABS_AVAILABLE = False
from rich.console import Console
from decimal import Decimal, ROUND_HALF_UP

# ═══════════════════════════════════════════════════════════════════
# ╔═╗╔═╗╔═╗  ╔═╗╦═╗╦╔═╗╦╔╗╔╔═╗  ╔═╗╔═╗╔╗╔╔═╗╔╦╗╔═╗╔╗╔╔╦╗╔═╗
# ╠═╣╠═╝╠═╣  ╠═╝╠╦╝║║  ║║║║║ ╦  ║  ║ ║║║║╚═╗ ║ ╠═╣║║║ ║ ╚═╗
# ╩ ╩╩  ╩ ╩  ╩  ╩╚═╩╚═╝╩╝╚╝╚═╝  ╚═╝╚═╝╝╚╝╚═╝ ╩ ╩ ╩╝╚╝ ╩ ╚═╝
# ═══════════════════════════════════════════════════════════════════

# API pricing constants (as of March 2025)
# Claude pricing
CLAUDE_INPUT_PRICE_PER_MILLION = 15.00  # USD per 1M input tokens
CLAUDE_OUTPUT_PRICE_PER_MILLION = 75.00  # USD per 1M output tokens

# OpenAI TTS pricing
TTS_STANDARD_PRICE_PER_MILLION = 15.00  # USD per 1M characters
TTS_HD_PRICE_PER_MILLION = 30.00  # USD per 1M characters

# Eleven Labs pricing (as of March 2025)
ELEVENLABS_STANDARD_PRICE_PER_MILLION = 15.00  # USD per 1M characters (approximation)
ELEVENLABS_TURBO_PRICE_PER_MILLION = 7.00     # USD per 1M characters (approximation)

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
from rich.group import Group
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
            if ELEVENLABS_AVAILABLE:
                self.elevenlabs_api_key = self._load_elevenlabs_api_key()
                self.elevenlabs_available = self.elevenlabs_api_key is not None
            else:
                self.elevenlabs_available = False
                self.elevenlabs_api_key = None

        # Initialize API clients
        self.anthropic_client = anthropic.Anthropic(api_key=self.anthropic_api_key)
        self.openai_client = openai.OpenAI(api_key=self.openai_api_key)
        
        # Initialize Eleven Labs client if available
        if self.elevenlabs_available:
            # Mask API key for security in logs
            masked_key = "None" if not self.elevenlabs_api_key else f"{'*' * (len(self.elevenlabs_api_key) - 8)}{self.elevenlabs_api_key[-8:]}"
            console.print(f"[blue]Initializing Eleven Labs with API key: {masked_key}[/blue]")
            
            try:
                # Define our configuration paths
                config_dir = Path.home() / ".speech_config"
                config_dir.mkdir(exist_ok=True)
                api_keys_file = config_dir / "api_keys.json"
                
                # Load existing API keys if file exists
                saved_keys = {}
                if api_keys_file.exists():
                    try:
                        saved_keys = json.loads(api_keys_file.read_text())
                        # If we have a saved key but no current key, use the saved one
                        if "elevenlabs" in saved_keys and not self.elevenlabs_api_key:
                            self.elevenlabs_api_key = saved_keys["elevenlabs"]
                            console.print("[blue]Using saved Eleven Labs API key from config[/blue]")
                    except json.JSONDecodeError:
                        console.print("[yellow]Error reading API keys file. Starting with empty configuration.[/yellow]")
                
                # We'll primarily use the direct API approach since the package structure varies widely
                # Set environment variable as a reliable fallback
                os.environ["ELEVEN_API_KEY"] = self.elevenlabs_api_key
                
                # Function to validate the API key
                def validate_eleven_labs_key(api_key):
                    if not api_key:
                        return False, "No API key provided", []
                        
                    try:
                        # Make a simple request to check if the API key works
                        headers = {"xi-api-key": api_key}
                        response = requests.get("https://api.elevenlabs.io/v1/voices", headers=headers)
                        
                        # Check if the request was successful
                        if response.status_code == 200:
                            # Extract voices if available
                            voice_data = response.json()
                            voices = voice_data.get("voices", [])
                            return True, "API key valid", voices
                        else:
                            return False, f"API error: {response.status_code} - {response.text[:100]}", []
                    except Exception as e:
                        return False, f"Connection error: {str(e)}", []
                
                # First try to validate with current key
                import requests
                key_valid, message, voices = validate_eleven_labs_key(self.elevenlabs_api_key)
                
                # If key is invalid, give the user a chance to enter a new one
                max_attempts = 3
                attempts = 0
                while not key_valid and attempts < max_attempts:
                    console.print(f"[bold red]Eleven Labs API key validation failed: {message}[/bold red]")
                    console.print("[bold red]Your Eleven Labs API key is not valid or has expired[/bold red]")
                    
                    # Ask if the user wants to enter a new key
                    if Confirm.ask("Would you like to enter a new Eleven Labs API key?", default=True):
                        attempts += 1
                        new_key = console.input("[bold green]Enter new Eleven Labs API key: [/bold green]")
                        
                        # Validate the new key
                        console.print("[blue]Validating new API key...[/blue]")
                        key_valid, message, voices = validate_eleven_labs_key(new_key)
                        
                        if key_valid:
                            # Update the key if valid
                            self.elevenlabs_api_key = new_key
                            console.print("[bold green]New API key validated successfully![/bold green]")
                            
                            # Save the key to config file
                            saved_keys["elevenlabs"] = new_key
                            api_keys_file.write_text(json.dumps(saved_keys, indent=2))
                            console.print("[green]Saved new API key to configuration[/green]")
                            
                            # Update environment variable
                            os.environ["ELEVEN_API_KEY"] = new_key
                            break
                        else:
                            console.print(f"[bold red]New API key validation failed: {message}[/bold red]")
                            console.print(f"[yellow]Attempts remaining: {max_attempts - attempts}[/yellow]")
                    else:
                        # User opted not to enter a new key
                        console.print("[yellow]Continuing without a valid Eleven Labs API key[/yellow]")
                        break
                
                # Set flags based on validation result
                self.elevenlabs_key_valid = key_valid
                if key_valid:
                    console.print("[green]Successfully validated Eleven Labs API key![/green]")
                    self.elevenlabs_voices = voices
                    if voices:
                        console.print(f"[green]Found {len(voices)} voices in your Eleven Labs account[/green]")
                    else:
                        console.print("[yellow]No voices found in your Eleven Labs account[/yellow]")
                else:
                    self.elevenlabs_voices = []
                
                # Set flags to indicate we have a configured client
                self.elevenlabs_configured = True
                
                # Enable manual voice ID entry that will always work
                self.elevenlabs_manual_voice_id = True
                console.print("[green]Configured for manual voice ID entry with Eleven Labs[/green]")
                console.print("[cyan]You'll be able to enter your voice ID when selecting a voice[/cyan]")
                
            except Exception as e:
                console.print(f"[bold red]Error initializing Eleven Labs: {e}[/bold red]")
                self.elevenlabs_available = False

        # Cache directory for storing processed results
        self.cache_dir = Path.home() / ".text2speech_cache"
        self.cache_dir.mkdir(exist_ok=True)

        # Voice options
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
        
        # Add Eleven Labs voices if available
        self.elevenlabs_voices = []
        if self.elevenlabs_available:
            console.print("[bold blue]Checking for Eleven Labs voices...[/bold blue]")
            logger.info(f"Elevenlabs API key is set: {bool(self.elevenlabs_api_key)}")
            try:
                # Skip trying to get voices automatically - it's more reliable to let the user enter their voice ID
                console.print("[yellow]Setting up Eleven Labs for manual voice ID entry...[/yellow]")
                
                # Create a placeholder for Eleven Labs voices
                self.elevenlabs_voices = []
                
                # Create an empty voices list that will prompt the user to enter their voice ID
                self.voice_options["Eleven Labs voices"] = []
                
                console.print("[green]Eleven Labs voice option enabled. You'll be prompted for your voice ID when selected.[/green]")
                
                # Print the found voices for debugging
                for voice in self.elevenlabs_voices:
                    console.print(f"[dim]Found Eleven Labs voice: {voice['name']} (ID: {voice['id']})[/dim]")
                
                if self.elevenlabs_voices:
                    # Add the voices to the options
                    self.voice_options["Eleven Labs voices"] = [
                        f"elevenlabs_{voice['id']}" for voice in self.elevenlabs_voices
                    ]
                    
                    logger.info(f"Found {len(self.elevenlabs_voices)} Eleven Labs voices")
                    console.print(f"[bold green]Successfully added {len(self.elevenlabs_voices)} Eleven Labs voices to options[/bold green]")
                else:
                    logger.warning("No Eleven Labs voices found.")
                    console.print("[yellow]No Eleven Labs voices found in your account. Please add a voice in your Eleven Labs dashboard.[/yellow]")
            except Exception as e:
                logger.error(f"Error fetching Eleven Labs voices: {e}")
                console.print(f"[bold red]Error connecting to Eleven Labs: {e}[/bold red]")
                console.print("[yellow]Please check your Eleven Labs API key and internet connection.[/yellow]")

        # Voice descriptions to help users choose - more detailed with gender and style information
        self.voice_descriptions = {
            # OpenAI Voices
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
        
        # Add Eleven Labs voice descriptions
        if self.elevenlabs_available and self.elevenlabs_voices:
            for voice in self.elevenlabs_voices:
                voice_id = f"elevenlabs_{voice['id']}"
                self.voice_descriptions[voice_id] = f"Eleven Labs: {voice['name']} (custom voice)"

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
        """Load Anthropic API key from environment, config file, or user input."""
        # Setup central config directory
        config_dir = Path.home() / ".speech_config"
        config_dir.mkdir(exist_ok=True)
        api_keys_file = config_dir / "api_keys.json"
        
        # Load existing API keys if file exists
        saved_keys = {}
        if api_keys_file.exists():
            try:
                saved_keys = json.loads(api_keys_file.read_text())
                # If we have a saved key, use it
                if "anthropic" in saved_keys and saved_keys["anthropic"]:
                    api_key = saved_keys["anthropic"]
                    logger.info("Loaded Anthropic API key from config file")
                    console.print("[green]Found API key in config file[/green]")
                    
                    # Validate the key if possible (for now just check format)
                    if re.match(r"^sk-ant-api\w+$", api_key):
                        return api_key
                    else:
                        console.print("[yellow]Saved Anthropic API key has invalid format, will try other sources[/yellow]")
            except json.JSONDecodeError:
                console.print("[yellow]Error reading API keys file. Starting with empty configuration.[/yellow]")
        
        # First try environment variable
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if api_key:
            logger.info("Loaded Anthropic API key from environment variable")
            
            # Save to config for future use
            saved_keys["anthropic"] = api_key
            api_keys_file.write_text(json.dumps(saved_keys, indent=2))
            console.print("[green]Saved environment API key to configuration[/green]")
            
            return api_key

        # List of potential paths to check for Anthropic API key
        potential_paths = [
            Path.home() / ".anthropic",
            Path.home() / ".apikey",
            Path.home() / ".apikeys",
            Path.home() / ".api-keys",
            Path.home() / ".keys" / "anthropic",
            Path.home() / ".keys",
            Path.home() / ".config" / "anthropic",
            Path.home() / ".config" / "keys" / "anthropic",
            Path(".anthropic")  # Also check current directory
        ]
        
        # Try all potential paths
        for anthropic_file in potential_paths:
            if anthropic_file.exists():
                try:
                    console.print(f"[dim]Checking for Anthropic API key in {anthropic_file}...[/dim]")
                    content = anthropic_file.read_text().strip()
                    
                    # Look for API key pattern directly
                    if re.match(r"^sk-ant-api\w+$", content):
                        logger.info(f"Loaded Anthropic API key from {anthropic_file}")
                        console.print(f"[green]Found API key in {anthropic_file}[/green]")
                        
                        # Save to config for future use
                        saved_keys["anthropic"] = content
                        api_keys_file.write_text(json.dumps(saved_keys, indent=2))
                        console.print("[green]Saved API key to configuration[/green]")
                        
                        return content

                    # Try to find key in a key=value format or ANTHROPIC_API_KEY=sk-...
                    match = re.search(
                        r"(?:ANTHROPIC_API_KEY|api_key)[=:]\s*([^\s]+)", content
                    )
                    if match:
                        api_key = match.group(1)
                        logger.info(f"Loaded Anthropic API key from {anthropic_file}")
                        console.print(f"[green]Found API key in {anthropic_file}[/green]")
                        
                        # Save to config for future use
                        saved_keys["anthropic"] = api_key
                        api_keys_file.write_text(json.dumps(saved_keys, indent=2))
                        console.print("[green]Saved API key to configuration[/green]")
                        
                        return api_key
                    
                    # Check for any line that looks like an Anthropic API key
                    for line in content.splitlines():
                        line = line.strip()
                        if line.startswith("sk-ant-api"):
                            logger.info(f"Loaded Anthropic API key from {anthropic_file}")
                            console.print(f"[green]Found API key in {anthropic_file}[/green]")
                            
                            # Save to config for future use
                            saved_keys["anthropic"] = line
                            api_keys_file.write_text(json.dumps(saved_keys, indent=2))
                            console.print("[green]Saved API key to configuration[/green]")
                            
                            return line

                    # If it's a JSON file, try to parse it
                    try:
                        data = json.loads(content)
                        if isinstance(data, dict):
                            # Try various key names
                            for key_name in ["api_key", "key", "anthropic_api_key", "anthropic_key", "ANTHROPIC_API_KEY"]:
                                if key_name in data:
                                    api_key = data[key_name]
                                    logger.info(f"Loaded Anthropic API key from {anthropic_file} JSON file")
                                    console.print(f"[green]Found API key in {anthropic_file} JSON[/green]")
                                    
                                    # Save to config for future use
                                    saved_keys["anthropic"] = api_key
                                    api_keys_file.write_text(json.dumps(saved_keys, indent=2))
                                    console.print("[green]Saved API key to configuration[/green]")
                                    
                                    return api_key
                    except json.JSONDecodeError:
                        pass
                except Exception as e:
                    logger.warning(f"Error reading {anthropic_file}: {e}")

        # If we get here, prompt the user for a key
        console.print(
            "[bold red]Error:[/bold red] Anthropic API key not found in environment or any of the expected files"
        )
        console.print(
            "Please provide your Anthropic API key"
        )
        api_key = console.input("[bold green]Enter Anthropic API key: [/bold green]")
        
        # Only proceed if we got a key
        if api_key:
            # Save to central config
            saved_keys["anthropic"] = api_key
            api_keys_file.write_text(json.dumps(saved_keys, indent=2))
            console.print(f"[green]Saved API key to {api_keys_file}[/green]")
            
            # Also save to traditional location for compatibility
            try:
                anthropic_file = Path.home() / ".anthropic"
                anthropic_file.write_text(api_key)
                console.print(f"[green]Also saved API key to {anthropic_file} for compatibility[/green]")
            except Exception as e:
                logger.warning(f"Error saving Anthropic API key to compatibility location: {e}")
        
        return api_key

    def _load_openai_api_key(self) -> str:
        """Load OpenAI API key from config file, environment, or user input."""
        # Setup central config directory
        config_dir = Path.home() / ".speech_config"
        config_dir.mkdir(exist_ok=True)
        api_keys_file = config_dir / "api_keys.json"
        
        # Load existing API keys if file exists
        saved_keys = {}
        if api_keys_file.exists():
            try:
                saved_keys = json.loads(api_keys_file.read_text())
                # If we have a saved key, use it
                if "openai" in saved_keys and saved_keys["openai"]:
                    api_key = saved_keys["openai"]
                    logger.info("Loaded OpenAI API key from config file")
                    console.print("[green]Found OpenAI API key in config file[/green]")
                    
                    # Simple validation - just check format
                    if re.match(r"^sk-\w+$", api_key):
                        return api_key
                    else:
                        console.print("[yellow]Saved OpenAI API key has invalid format, will try other sources[/yellow]")
            except json.JSONDecodeError:
                console.print("[yellow]Error reading API keys file. Starting with empty configuration.[/yellow]")
        
        # First try environment variable
        api_key = os.environ.get("OPENAI_API_KEY")
        if api_key:
            logger.info("Loaded OpenAI API key from environment variable")
            
            # Save to config for future use
            saved_keys["openai"] = api_key
            api_keys_file.write_text(json.dumps(saved_keys, indent=2))
            console.print("[green]Saved OpenAI environment API key to configuration[/green]")
            
            return api_key

        # Try ~/.apikey file
        apikey_file = Path.home() / ".apikey"
        if apikey_file.exists():
            try:
                content = apikey_file.read_text().strip()

                # Try to find OpenAI key in content
                if re.match(r"^sk-\w+$", content):
                    logger.info("Loaded OpenAI API key from ~/.apikey file")
                    
                    # Save to config for future use
                    saved_keys["openai"] = content
                    api_keys_file.write_text(json.dumps(saved_keys, indent=2))
                    console.print("[green]Saved OpenAI API key to configuration[/green]")
                    
                    return content

                # Try to find key in a key=value format
                match = re.search(r"(?:OPENAI_API_KEY|api_key)[=:]\s*([^\s]+)", content)
                if match:
                    api_key = match.group(1)
                    logger.info("Loaded OpenAI API key from ~/.apikey file")
                    
                    # Save to config for future use
                    saved_keys["openai"] = api_key
                    api_keys_file.write_text(json.dumps(saved_keys, indent=2))
                    console.print("[green]Saved OpenAI API key to configuration[/green]")
                    
                    return api_key

                # If it's a JSON file, try to parse it
                try:
                    data = json.loads(content)
                    if isinstance(data, dict) and "api_key" in data:
                        api_key = data["api_key"]
                        logger.info("Loaded OpenAI API key from ~/.apikey JSON file")
                        
                        # Save to config for future use
                        saved_keys["openai"] = api_key
                        api_keys_file.write_text(json.dumps(saved_keys, indent=2))
                        console.print("[green]Saved OpenAI API key to configuration[/green]")
                        
                        return api_key
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
                    
                    # Save to config for future use
                    saved_keys["openai"] = content
                    api_keys_file.write_text(json.dumps(saved_keys, indent=2))
                    console.print("[green]Saved OpenAI API key to configuration[/green]")
                    
                    return content

                # Try to find key in a key=value format
                match = re.search(r"(?:OPENAI_API_KEY|api_key)[=:]\s*([^\s]+)", content)
                if match:
                    api_key = match.group(1)
                    logger.info("Loaded OpenAI API key from ~/.openai file")
                    
                    # Save to config for future use
                    saved_keys["openai"] = api_key
                    api_keys_file.write_text(json.dumps(saved_keys, indent=2))
                    console.print("[green]Saved OpenAI API key to configuration[/green]")
                    
                    return api_key

                # If it's a JSON file, try to parse it
                try:
                    data = json.loads(content)
                    if isinstance(data, dict) and "api_key" in data:
                        api_key = data["api_key"]
                        logger.info("Loaded OpenAI API key from ~/.openai JSON file")
                        
                        # Save to config for future use
                        saved_keys["openai"] = api_key
                        api_keys_file.write_text(json.dumps(saved_keys, indent=2))
                        console.print("[green]Saved OpenAI API key to configuration[/green]")
                        
                        return api_key
                except json.JSONDecodeError:
                    pass
            except Exception as e:
                logger.warning(f"Error reading ~/.openai file: {e}")

        # If we get here, prompt the user for a key
        console.print(
            "[bold red]Error:[/bold red] OpenAI API key not found in saved config, environment, or expected files"
        )
        console.print(
            "Please provide your OpenAI API key"
        )
        api_key = console.input("[bold green]Enter OpenAI API key: [/bold green]")
        
        # Save the new key if provided
        if api_key:
            # Save to central config
            saved_keys["openai"] = api_key
            api_keys_file.write_text(json.dumps(saved_keys, indent=2))
            console.print(f"[green]Saved OpenAI API key to {api_keys_file}[/green]")
        
        return api_key
        
    def _load_elevenlabs_api_key(self) -> Optional[str]:
        """Load Eleven Labs API key from central config, environment, or user input."""
        console.print("[bold blue]Looking for Eleven Labs API key...[/bold blue]")
        
        # Setup central config directory
        config_dir = Path.home() / ".speech_config"
        config_dir.mkdir(exist_ok=True)
        api_keys_file = config_dir / "api_keys.json"
        
        # Load existing API keys if file exists
        saved_keys = {}
        if api_keys_file.exists():
            try:
                saved_keys = json.loads(api_keys_file.read_text())
                # If we have a saved key, use it
                if "elevenlabs" in saved_keys and saved_keys["elevenlabs"]:
                    api_key = saved_keys["elevenlabs"]
                    logger.info("Loaded Eleven Labs API key from config file")
                    
                    # Mask API key for security in logs
                    if len(api_key) > 8:
                        masked_key = "*" * (len(api_key) - 8) + api_key[-8:]
                    else:
                        masked_key = "*" * len(api_key)
                    console.print(f"[green]Found Eleven Labs API key in config: {masked_key}[/green]")
                    
                    return api_key
            except json.JSONDecodeError:
                console.print("[yellow]Error reading API keys file. Starting with empty configuration.[/yellow]")
        
        # First try environment variable
        api_key = os.environ.get("ELEVENLABS_API_KEY")
        if api_key:
            logger.info("Loaded Eleven Labs API key from environment variable")
            console.print("[green]Found Eleven Labs API key in environment variable[/green]")
            
            # Save to config for future use
            saved_keys["elevenlabs"] = api_key
            api_keys_file.write_text(json.dumps(saved_keys, indent=2))
            console.print("[green]Saved Eleven Labs API key from environment to configuration[/green]")
            
            return api_key

        # List of potential paths to check for Eleven Labs API key
        potential_paths = [
            Path.home() / ".elevenlabs",
            Path.home() / ".eleven-labs",
            Path.home() / ".apikey",
            Path.home() / ".elevenlabs.txt",
            Path.home() / ".keys" / "elevenlabs",
            Path.home() / ".config" / "elevenlabs",
            Path(".elevenlabs")  # Also check current directory
        ]
        
        # Try all potential paths
        for elevenlabs_file in potential_paths:
            if elevenlabs_file.exists():
                try:
                    console.print(f"[dim]Checking for Eleven Labs API key in {elevenlabs_file}...[/dim]")
                    content = elevenlabs_file.read_text().strip()
                    
                    # Print out the content for debugging (with masking)
                    if len(content) > 8:
                        masked_content = "*" * (len(content) - 8) + content[-8:]
                    else:
                        masked_content = "*" * len(content)
                    console.print(f"[dim]Content in {elevenlabs_file} (masked): {masked_content}[/dim]")
                    
                    # Accept any non-empty content as potentially a valid key
                    if content.strip():
                        logger.info(f"Loaded Eleven Labs API key from {elevenlabs_file}")
                        console.print(f"[green]Found Eleven Labs API key in {elevenlabs_file}[/green]")
                        # Try to clean up the content by removing any non-alphanumeric chars
                        clean_key = re.sub(r'[^a-zA-Z0-9]', '', content)
                        
                        # Save to config for future use
                        if clean_key:
                            saved_keys["elevenlabs"] = clean_key
                            api_key = clean_key
                        else:
                            saved_keys["elevenlabs"] = content
                            api_key = content
                            
                        api_keys_file.write_text(json.dumps(saved_keys, indent=2))
                        console.print("[green]Saved Eleven Labs API key to configuration[/green]")
                        
                        return api_key
                    
                    # Try to find key in a key=value format
                    match = re.search(r"(?:ELEVENLABS_API_KEY|api_key|key)[=:]\s*([^\s]+)", content)
                    if match:
                        api_key = match.group(1)
                        logger.info(f"Loaded Eleven Labs API key from {elevenlabs_file}")
                        console.print(f"[green]Found Eleven Labs API key in {elevenlabs_file}[/green]")
                        
                        # Save to config for future use
                        saved_keys["elevenlabs"] = api_key
                        api_keys_file.write_text(json.dumps(saved_keys, indent=2))
                        console.print("[green]Saved Eleven Labs API key to configuration[/green]")
                        
                        return api_key
                    
                    # If it's a JSON file, try to parse it
                    try:
                        data = json.loads(content)
                        if isinstance(data, dict):
                            # Try various key names
                            for key_name in ["api_key", "key", "elevenlabs_api_key", "elevenlabs_key", "ELEVENLABS_API_KEY"]:
                                if key_name in data:
                                    api_key = data[key_name]
                                    logger.info(f"Loaded Eleven Labs API key from {elevenlabs_file} JSON file")
                                    console.print(f"[green]Found Eleven Labs API key in {elevenlabs_file} JSON[/green]")
                                    
                                    # Save to config for future use
                                    saved_keys["elevenlabs"] = api_key
                                    api_keys_file.write_text(json.dumps(saved_keys, indent=2))
                                    console.print("[green]Saved Eleven Labs API key to configuration[/green]")
                                    
                                    return api_key
                    except json.JSONDecodeError:
                        pass
                    
                    # Check if the content itself might be a key (any line that looks like a typical API key)
                    for line in content.splitlines():
                        line = line.strip()
                        if re.match(r"^[a-zA-Z0-9]{32,}$", line):
                            logger.info(f"Loaded Eleven Labs API key from {elevenlabs_file}")
                            console.print(f"[green]Found Eleven Labs API key in {elevenlabs_file}[/green]")
                            
                            # Save to config for future use
                            saved_keys["elevenlabs"] = line
                            api_keys_file.write_text(json.dumps(saved_keys, indent=2))
                            console.print("[green]Saved Eleven Labs API key to configuration[/green]")
                            
                            return line
                except Exception as e:
                    logger.warning(f"Error reading {elevenlabs_file}: {e}")
        
        # If Eleven Labs is available and we couldn't find the API key, prompt the user
        if ELEVENLABS_AVAILABLE:
            console.print(
                "[bold yellow]Note:[/bold yellow] Eleven Labs support is available but no API key was found."
            )
            console.print(
                "To use your custom Eleven Labs voice, please provide your API key."
            )
            console.print(
                "You can find your API key at https://elevenlabs.io/app/account/api-key"
            )
            
            if Confirm.ask("Would you like to add an Eleven Labs API key?"):
                api_key = console.input("[bold green]Enter Eleven Labs API key: [/bold green]")
                
                # Save the key for future use
                if api_key:
                    try:
                        # Save to central config
                        saved_keys["elevenlabs"] = api_key
                        api_keys_file.write_text(json.dumps(saved_keys, indent=2))
                        console.print(f"[green]Saved Eleven Labs API key to {api_keys_file}[/green]")
                        
                        # Also save to traditional location for compatibility
                        elevenlabs_file = Path.home() / ".elevenlabs"
                        elevenlabs_file.write_text(api_key)
                        console.print("[green]Also saved API key to ~/.elevenlabs file for compatibility[/green]")
                        
                        # We'll validate this key during initialization
                    except Exception as e:
                        logger.warning(f"Error saving Eleven Labs API key: {e}")
                        console.print(f"[yellow]Could not save API key: {e}[/yellow]")
                    
                    return api_key
                else:
                    console.print("[yellow]No API key entered. Eleven Labs support will be disabled.[/yellow]")
                    return None
            else:
                console.print("[yellow]Eleven Labs support will be disabled.[/yellow]")
                return None
        
        console.print("[yellow]No Eleven Labs API key found. Custom voices will not be available.[/yellow]")
        return None

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

    def display_voice_options(self, non_interactive=False) -> Tuple[str, List[str], List[float]]:
        """Display voice options in a table and let user select or use defaults."""
        if non_interactive:
            # Use highest quality options for non-interactive mode
            logger.info(
                "Running in non-interactive mode, using default voice: onyx-hd, wav format, 1.0x speed"
            )
            return "onyx-hd", ["wav"], [1.0]

        # Create a visually appealing main selection panel with more details
        console.print(
            Panel(
                "[bold white]Voice and Audio Settings[/bold white]\n"
                "[dim]Configure how your text will sound when converted to speech.[/dim]\n\n"
                "[cyan]In the following steps, you'll configure:[/cyan]\n"
                "1️⃣ [bold yellow]Voice Provider[/bold yellow] - Choose between OpenAI or Eleven Labs\n"
                "2️⃣ [bold yellow]Voice Selection[/bold yellow] - Pick from available premium voices\n"
                "3️⃣ [bold yellow]Speech Rate[/bold yellow] - Set the speaking speed and rhythm\n"
                "4️⃣ [bold yellow]Output Format[/bold yellow] - Select audio file format and quality",
                title="🔊 Audio Configuration",
                subtitle="Make your content sound natural",
                border_style="cyan",
                padding=(1, 2),
                width=100,
            )
        )

        # First select voice quality
        # Use a panel for the section header with consistent width
        console.print(
            Panel(
                "[bold cyan]Step 1: Voice Provider and Quality Options[/bold cyan]\n"
                "[dim]Choose the voice technology that will be used for your audio[/dim]",
                border_style="cyan",
                padding=(1, 1),
                width=100,
            )
        )

        # Use a consistent width for the table
        quality_table = Table(
            show_header=True, 
            header_style="bold magenta", 
            box=box.ROUNDED, 
            title="Voice Provider Options",
            width=100,
            padding=(0, 1)
        )
        quality_table.add_column("Option", style="dim", justify="center", width=8)
        quality_table.add_column("Provider & Quality", style="cyan", width=25)
        quality_table.add_column("Description", style="green", width=35)
        quality_table.add_column("Best For", style="yellow", width=30)

        # Count available choices
        choice_count = 2  # Standard and HD are always available
        elevenlabs_option = None
        
        # Add standard options with more detail
        quality_table.add_row(
            "1", 
            "[cyan]OpenAI Standard[/cyan]", 
            "Regular quality, efficient processing", 
            "Quick conversions, everyday use"
        )
        quality_table.add_row(
            "2", 
            "[bold cyan]OpenAI HD[/bold cyan]", 
            "Premium quality with natural intonation",
            "Professional recordings, presentations"
        )
        
        # Always add Eleven Labs option if the module is available
        if self.elevenlabs_available:
            # Make sure we have this category
            if "Eleven Labs voices" not in self.voice_options:
                self.voice_options["Eleven Labs voices"] = []
                
            choice_count += 1
            elevenlabs_option = str(choice_count)
            quality_table.add_row(
                elevenlabs_option, 
                "[bold magenta]Eleven Labs Custom[/bold magenta]", 
                "Ultra-realistic custom voices and clones",
                "Personalized narration, character voices"
            )

        console.print(quality_table)

        # If Eleven Labs is available, show a special note in a stylized box
        if self.elevenlabs_available and "Eleven Labs voices" in self.voice_options:
            console.print(
                Panel(
                    "[bold cyan]Eleven Labs integration is active.[/bold cyan]\n"
                    "You can use your custom cloned voices or premium Eleven Labs voices.",
                    title="🎤 Custom Voice Support",
                    border_style="magenta",
                    padding=(1, 1),
                    width=100,
                )
            )

        # Generate valid choices with improved prompt
        valid_choices = [str(i) for i in range(1, choice_count + 1)]
        quality_choice = ""
        while quality_choice not in valid_choices:
            quality_choice = console.input(
                f"\n[bold yellow]Select voice provider/quality ({'/'.join(valid_choices)}): [/bold yellow]"
            )

        # Determine voice category based on selection
        if quality_choice == "1":
            voice_category = "Standard voices"
        elif quality_choice == "2":
            voice_category = "HD voices"
        elif elevenlabs_option and quality_choice == elevenlabs_option:
            voice_category = "Eleven Labs voices"

        # Then select specific voice with improved visuals and consistent width
        console.print(
            Panel(
                f"[bold cyan]Step 2: Select Voice Type[/bold cyan]\n"
                f"[dim]Choose from the available {voice_category.lower()}[/dim]",
                border_style="cyan",
                padding=(1, 1),
                width=100,
            )
        )

        voice_table = Table(
            show_header=True, 
            header_style="bold magenta", 
            box=box.ROUNDED,
            width=100,
            padding=(0, 1)
        )
        voice_table.add_column("Option", style="dim", justify="center", width=8)
        voice_table.add_column("Voice", style="cyan", width=15)
        voice_table.add_column("Description", style="green", width=45)
        voice_table.add_column("Best For", style="yellow", width=30)

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
        
        # Handle special case for Eleven Labs with no voices
        if voice_category == "Eleven Labs voices":
            # First check if we have saved voices from a previous run
            saved_voices_file = Path.home() / ".speech_saved_elevenlabs_voices.json"
            saved_voices = []
            
            if saved_voices_file.exists():
                try:
                    saved_data = json.loads(saved_voices_file.read_text())
                    saved_voices = saved_data.get("voices", [])
                    
                    if saved_voices:
                        # Create a more attractive display table for saved voices
                        console.print(Panel(
                            "[bold green]Saved Eleven Labs Voices Found[/bold green]\n"
                            "The following custom voices were found in your profile:",
                            title="🎤 Custom Voices",
                            border_style="cyan",
                            padding=(1, 2),
                            width=100
                        ))
                        
                        voice_table = Table(
                            show_header=True, 
                            header_style="bold magenta", 
                            box=box.ROUNDED,
                            width=100,
                            padding=(0, 1)
                        )
                        voice_table.add_column("Option", style="dim", justify="center", width=8)
                        voice_table.add_column("Voice Name", style="cyan", width=35)
                        voice_table.add_column("Voice ID", style="green", width=55)
                        
                        for i, voice_data in enumerate(saved_voices, 1):
                            voice_table.add_row(
                                str(i),
                                voice_data['name'],
                                voice_data['id']
                            )
                        
                        console.print(voice_table)
                        
                        # Use a custom prompt instead of Confirm.ask to handle numeric input
                        prompt_text = "Would you like to use a saved voice? [y/n/1-" + str(len(saved_voices)) + "] (y): "
                        user_response = console.input(f"\n[bold yellow]{prompt_text}[/bold yellow]") or "y"
                        
                        # Check if the user entered a number (trying to select voice directly)
                        try:
                            voice_idx = int(user_response)
                            if 1 <= voice_idx <= len(saved_voices):
                                # User entered a valid voice number as confirmation
                                use_saved = True
                                choice = str(voice_idx)  # Use this number as direct selection
                                valid_choice = True  # Mark as valid choice already
                            else:
                                console.print(f"[yellow]Invalid voice number. Please enter 1-{len(saved_voices)} or y/n.[/yellow]")
                                use_saved = Confirm.ask("Would you like to use a saved voice?", default=True)
                                valid_choice = False  # Will need to select a voice
                        except ValueError:
                            # User entered y/n or invalid text
                            if user_response.lower() in ('y', 'yes'):
                                use_saved = True
                                valid_choice = False  # Will need to select a voice
                            elif user_response.lower() in ('n', 'no'):
                                use_saved = False
                                valid_choice = True  # No need to select a voice
                            else:
                                console.print("[yellow]Invalid input. Please enter y/n or a voice number.[/yellow]")
                                use_saved = Confirm.ask("Would you like to use a saved voice?", default=True)
                                valid_choice = False  # Will need to select a voice
                        
                        if use_saved and not valid_choice:
                            choice = ""
                            
                            while not valid_choice:
                                choice = console.input("\n[bold yellow]Enter voice number (or 'n' for new): [/bold yellow]")
                                
                                if choice.lower() == 'n':
                                    valid_choice = True
                                    break
                                
                                try:
                                    idx = int(choice) - 1
                                    if 0 <= idx < len(saved_voices):
                                        valid_choice = True
                                    else:
                                        console.print(f"[bold red]Invalid selection. Please enter a number between 1 and {len(saved_voices)} or 'n' for new.[/bold red]")
                                except ValueError:
                                    console.print("[bold red]Please enter a valid number or 'n' for new.[/bold red]")
                            
                            if choice.lower() != 'n':
                                try:
                                    idx = int(choice) - 1
                                    if 0 <= idx < len(saved_voices):
                                        selected_voice_data = saved_voices[idx]
                                        voice_id_clean = selected_voice_data["id"]
                                        elevenlabs_voice_name = selected_voice_data["name"]
                                        
                                        # Add to voices and descriptions
                                        voice_entry = f"elevenlabs_{voice_id_clean}"
                                        if voice_entry not in voices:
                                            voices.append(voice_entry)
                                            self.voice_descriptions[voice_entry] = f"Eleven Labs: {elevenlabs_voice_name}"
                                        
                                        console.print(f"[green]Selected voice: {elevenlabs_voice_name} with ID: {voice_id_clean}[/green]")
                                        
                                        # Set as selected voice directly
                                        selected_voice = voice_entry
                                        # Since we're returning early, we need to prompt for speech rates and formats
                                        # Prompt for speech rates
                                        console.print("\n[bold cyan]Step 3: Speech Rate[/bold cyan]")
                                        console.print("[cyan]How fast should the voice speak?[/cyan]")
                                        
                                        # Display the rate table with options
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
                                        
                                        console.print("\n[cyan]You can select multiple speech rates (comma-separated) to generate multiple versions.[/cyan]")
                                        
                                        rate_choices = []
                                        valid_input = False
                                        
                                        while not valid_input:
                                            try:
                                                rate_input = console.input(
                                                    f"\n[bold yellow]Select speech rate(s) (1-{len(self.speech_rate_options)}, comma-separated): [/bold yellow]"
                                                )
                                                
                                                # Parse the input - support for comma separated and range notation
                                                parts = rate_input.split(',')
                                                rate_choices = []
                                                
                                                for part in parts:
                                                    part = part.strip()
                                                    
                                                    # Check if it's a range (e.g., "1-3")
                                                    if '-' in part:
                                                        start, end = map(int, part.split('-'))
                                                        rate_choices.extend(range(start, end + 1))
                                                    else:
                                                        # Single number
                                                        rate_choices.append(int(part))
                                                
                                                # Validate all entries
                                                all_valid = True
                                                for idx in rate_choices:
                                                    if idx < 1 or idx > len(self.speech_rate_options):
                                                        console.print(f"[bold red]Invalid option: {idx} - must be between 1 and {len(self.speech_rate_options)}[/bold red]")
                                                        all_valid = False
                                                        break
                                                
                                                if all_valid:
                                                    valid_input = True
                                                    
                                            except ValueError:
                                                console.print("[bold red]Please enter valid numbers (e.g., 1,2,3 or 1-3)[/bold red]")
                                        
                                        # Convert to list of rate values
                                        selected_rates = [self.speech_rate_options[idx - 1]["value"] for idx in rate_choices]
                                        
                                        # Prompt for formats
                                        console.print("\n[bold cyan]Step 4: Output Format[/bold cyan]")
                                        console.print("[cyan]Choose the audio file format:[/cyan]")
                                        
                                        # Add note about HD quality and formats
                                        if "-hd" in selected_voice:
                                            console.print(
                                                "[bold yellow]Note:[/bold yellow] For HD voices, output format significantly affects quality. "
                                                "WAV and FLAC provide the highest fidelity but larger file sizes."
                                            )
                                        
                                        # Display format table
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
                                            "Compressed audio format, balanced quality/size",
                                            "Sharing, mobile devices, standard use",
                                        )
                                        format_table.add_row(
                                            "2",
                                            "WAV",
                                            "Uncompressed audio format, highest quality",
                                            "Professional use, editing, highest quality needs",
                                        )
                                        format_table.add_row(
                                            "3",
                                            "FLAC",
                                            "Lossless compressed format, very high quality",
                                            "Audiophile quality with smaller file size than WAV",
                                        )
                                        format_table.add_row(
                                            "4",
                                            "AAC",
                                            "Advanced compressed format, better than MP3",
                                            "Higher quality than MP3 with similar file size",
                                        )
                                        
                                        console.print(format_table)
                                        
                                        console.print("\n[cyan]You can select multiple output formats (comma-separated) to generate multiple versions.[/cyan]")
                                        
                                        format_choices = []
                                        valid_input = False
                                        
                                        while not valid_input:
                                            try:
                                                format_input = console.input(
                                                    "\n[bold yellow]Select output format(s) (1-4, comma-separated): [/bold yellow]"
                                                )
                                                
                                                # Parse the input - support for comma separated and range notation
                                                parts = format_input.split(',')
                                                format_choices = []
                                                
                                                for part in parts:
                                                    part = part.strip()
                                                    
                                                    # Check if it's a range (e.g., "1-3")
                                                    if '-' in part:
                                                        start, end = map(int, part.split('-'))
                                                        format_choices.extend(range(start, end + 1))
                                                    else:
                                                        # Single number
                                                        format_choices.append(int(part))
                                                
                                                # Validate all entries
                                                all_valid = True
                                                for idx in format_choices:
                                                    if idx < 1 or idx > 4:
                                                        console.print(f"[bold red]Invalid option: {idx} - must be between 1 and 4[/bold red]")
                                                        all_valid = False
                                                        break
                                                
                                                if all_valid:
                                                    valid_input = True
                                                    
                                            except ValueError:
                                                console.print("[bold red]Please enter valid numbers (e.g., 1,2,3 or 1-3)[/bold red]")
                                        
                                        format_options = {
                                            1: "mp3",
                                            2: "wav",
                                            3: "flac",
                                            4: "aac"
                                        }
                                        
                                        # Convert to list of format values
                                        selected_formats = [format_options[idx] for idx in format_choices]
                                        
                                        return selected_voice, selected_formats, selected_rates  # Skip the rest of the selection process
                                except (ValueError, IndexError):
                                    console.print("[yellow]Invalid selection, entering a new voice instead[/yellow]")
                except Exception as e:
                    console.print(f"[yellow]Error loading saved voices: {e}[/yellow]")
            
            # If we get here, we need to enter a new voice
            console.print("[yellow]Please provide an Eleven Labs voice ID manually.[/yellow]")
            console.print("[cyan]You can find your voice IDs at https://elevenlabs.io/app/voice-library[/cyan]")
            
            # Prompt for voice ID
            voice_id = console.input("\n[bold yellow]Enter your Eleven Labs voice ID: [/bold yellow]")
            
            # Create a placeholder voice entry
            voice_id_clean = voice_id.strip()
            elevenlabs_voice_name = console.input("\n[bold yellow]Enter a name for this voice (or leave blank for 'Custom Voice'): [/bold yellow]") or "Custom Voice"
            
            # Add to voices and descriptions
            voice_entry = f"elevenlabs_{voice_id_clean}"
            if voice_entry not in voices:
                voices.append(voice_entry)
                self.voice_descriptions[voice_entry] = f"Eleven Labs: {elevenlabs_voice_name}"
            
            # Save the voice for future use
            save_voice = Confirm.ask("Would you like to save this voice for future use?", default=True)
            if save_voice:
                try:
                    # Add to saved voices
                    voice_data = {"id": voice_id_clean, "name": elevenlabs_voice_name}
                    
                    # Load existing saved voices if any
                    if saved_voices_file.exists():
                        try:
                            saved_data = json.loads(saved_voices_file.read_text())
                            saved_voices = saved_data.get("voices", [])
                        except:
                            saved_voices = []
                    
                    # Check if voice already exists
                    voice_exists = False
                    for v in saved_voices:
                        if v["id"] == voice_id_clean:
                            voice_exists = True
                            v["name"] = elevenlabs_voice_name  # Update name if ID exists
                            break
                    
                    # Add if not exists
                    if not voice_exists:
                        saved_voices.append(voice_data)
                    
                    # Save back
                    saved_voices_file.write_text(json.dumps({"voices": saved_voices}, indent=2))
                    console.print(f"[green]Voice saved for future use: {elevenlabs_voice_name}[/green]")
                except Exception as e:
                    console.print(f"[yellow]Error saving voice: {e}[/yellow]")
            
            console.print(f"[green]Added voice: {elevenlabs_voice_name} with ID: {voice_id_clean}[/green]")
            
            # Set as selected voice directly
            selected_voice = voice_entry
            
        else:
            # Normal voice selection
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

        # Let user select speech rate with consistent styling
        console.print(
            Panel(
                "[bold cyan]Step 3: Speech Rate[/bold cyan]\n"
                "[dim]How fast should the voice speak?[/dim]",
                border_style="cyan",
                padding=(1, 1),
                width=100,
            )
        )

        rate_table = Table(
            show_header=True, 
            header_style="bold magenta", 
            box=box.ROUNDED,
            width=100,
            padding=(0, 1)
        )
        rate_table.add_column("Option", style="dim", justify="center", width=8)
        rate_table.add_column("Speed", style="cyan", width=25)
        rate_table.add_column("Best For", style="green", width=65)

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

        console.print("\n[cyan]You can select multiple speech rates (comma-separated) to generate multiple versions.[/cyan]")
        
        rate_choices = []
        valid_input = False
        
        while not valid_input:
            try:
                rate_input = console.input(
                    f"\n[bold yellow]Select speech rate(s) (1-{len(self.speech_rate_options)}, comma-separated): [/bold yellow]"
                )
                
                # Parse the input - support for comma separated and range notation
                parts = rate_input.split(',')
                rate_choices = []
                
                for part in parts:
                    part = part.strip()
                    
                    # Check if it's a range (e.g., "1-3")
                    if '-' in part:
                        start, end = map(int, part.split('-'))
                        rate_choices.extend(range(start, end + 1))
                    else:
                        # Single number
                        rate_choices.append(int(part))
                
                # Validate all entries
                all_valid = True
                for idx in rate_choices:
                    if idx < 1 or idx > len(self.speech_rate_options):
                        console.print(f"[bold red]Invalid option: {idx} - must be between 1 and {len(self.speech_rate_options)}[/bold red]")
                        all_valid = False
                        break
                
                if all_valid:
                    valid_input = True
                    
            except ValueError:
                console.print("[bold red]Please enter valid numbers (e.g., 1,2,3 or 1-3)[/bold red]")
        
        # Convert to list of rate values
        selected_rates = [self.speech_rate_options[idx - 1]["value"] for idx in rate_choices]
        
        # Display the selected rates for confirmation
        rate_labels = [self.speech_rate_options[idx - 1]["label"] for idx in rate_choices]
        console.print(f"[green]Selected rates: {', '.join(rate_labels)}[/green]")

        # Finally select format with enhanced visuals and consistent width
        console.print(
            Panel(
                "[bold cyan]Step 4: Output Format[/bold cyan]\n"
                "[dim]Choose the format(s) for your audio output.[/dim]\n\n"
                "[italic]WAV and FLAC are lossless formats that preserve all audio quality. \n"
                "MP3 and AAC are compressed formats with smaller file sizes but some quality loss.[/italic]",
                title="🔊 Audio Format Selection",
                border_style="cyan",
                padding=(1, 2),
                width=100,
            )
        )
        
        # Add color-coded format recommendation based on voice selection with consistent width
        if "-hd" in selected_voice or "elevenlabs_" in selected_voice:
            console.print(
                Panel(
                    "[bold yellow]Quality Recommendation[/bold yellow]\n"
                    f"You selected {'an HD' if '-hd' in selected_voice else 'a custom'} voice which can produce premium audio quality.\n\n"
                    "[bold green]Recommended:[/bold green] WAV or FLAC for maximum quality\n"
                    "[dim]These formats will preserve the high-fidelity audio characteristics of your selected voice.[/dim]",
                    border_style="yellow",
                    padding=(1, 2),
                    width=100,
                )
            )

        format_table = Table(
            show_header=True, 
            header_style="bold magenta", 
            box=box.ROUNDED,
            width=100,
            padding=(0, 1)
        )
        format_table.add_column("Option", style="dim", justify="center", width=8)
        format_table.add_column("Format", style="cyan", width=12)
        format_table.add_column("Description", style="green", width=40)
        format_table.add_column("Best For", style="yellow", width=38)

        format_table.add_row(
            "1",
            "MP3",
            "Compressed audio format, balanced quality/size",
            "Sharing, mobile devices, standard use",
        )
        format_table.add_row(
            "2",
            "WAV",
            "Uncompressed audio format, highest quality",
            "Professional use, editing, highest quality needs",
        )
        format_table.add_row(
            "3",
            "FLAC",
            "Lossless compressed format, very high quality",
            "Audiophile quality with smaller file size than WAV",
        )
        format_table.add_row(
            "4",
            "AAC",
            "Advanced compressed format, better than MP3",
            "Higher quality than MP3 with similar file size",
        )

        console.print(format_table)

        console.print("\n[cyan]You can select multiple output formats (comma-separated) to generate multiple versions.[/cyan]")
        
        format_choices = []
        valid_input = False
        
        while not valid_input:
            try:
                format_input = console.input(
                    "\n[bold yellow]Select output format(s) (1-4, comma-separated): [/bold yellow]"
                )
                
                # Parse the input - support for comma separated and range notation
                parts = format_input.split(',')
                format_choices = []
                
                for part in parts:
                    part = part.strip()
                    
                    # Check if it's a range (e.g., "1-3")
                    if '-' in part:
                        start, end = map(int, part.split('-'))
                        format_choices.extend(range(start, end + 1))
                    else:
                        # Single number
                        format_choices.append(int(part))
                
                # Validate all entries
                all_valid = True
                for idx in format_choices:
                    if idx < 1 or idx > 4:
                        console.print(f"[bold red]Invalid option: {idx} - must be between 1 and 4[/bold red]")
                        all_valid = False
                        break
                
                if all_valid:
                    valid_input = True
                    
            except ValueError:
                console.print("[bold red]Please enter valid numbers (e.g., 1,2,3 or 1-3)[/bold red]")
        
        format_options = {
            1: "mp3",
            2: "wav",
            3: "flac",
            4: "aac"
        }
        
        # Convert to list of format values
        selected_formats = [format_options[idx] for idx in format_choices]
        
        # Display the selected formats for confirmation with nicer formatting
        format_names = [format_options[idx].upper() for idx in format_choices]
        
        # Create a nice confirmation box with more details
        format_details = {
            "mp3": {"size": "Small", "quality": "Good", "compatibility": "Excellent (all devices)"},
            "wav": {"size": "Very large", "quality": "Lossless", "compatibility": "Good (most devices)"},
            "flac": {"size": "Large", "quality": "Lossless", "compatibility": "Limited (some devices)"},
            "aac": {"size": "Small", "quality": "Very good", "compatibility": "Good (most devices)"}
        }
        
        # Create a mini-table for selected formats with consistent width
        format_confirm_table = Table(
            show_header=True, 
            header_style="bold bright_cyan",
            box=box.ROUNDED,
            title="Selected Audio Formats",
            title_style="bold bright_green",
            border_style="green",
            width=100
        )
        
        format_confirm_table.add_column("Format", style="bright_cyan")
        format_confirm_table.add_column("File Size", style="bright_yellow")
        format_confirm_table.add_column("Quality", style="bright_green")
        format_confirm_table.add_column("Device Compatibility", style="bright_magenta")
        
        for fmt in selected_formats:
            details = format_details.get(fmt, {"size": "Unknown", "quality": "Unknown", "compatibility": "Unknown"})
            format_confirm_table.add_row(
                fmt.upper(),
                details["size"],
                details["quality"],
                details["compatibility"]
            )
            
        console.print(format_confirm_table)

        # Enhanced summary of selections with more visual appeal
        # Calculate total files that will be generated
        total_files = len(selected_rates) * len(selected_formats)
        
        # Process time estimate based on character count and format
        avg_processing_time_per_char = 0.005  # approx 5ms per character
        has_lossless = any(fmt in ['wav', 'flac'] for fmt in selected_formats)
        time_modifier = 1.5 if has_lossless else 1.0  # Lossless takes longer
        
        # Character count from earlier in the process, default to 3000 if not available
        char_count = 3000  # default
        if hasattr(self, 'char_count'):
            char_count = self.char_count
        
        est_processing_time = char_count * avg_processing_time_per_char * time_modifier * total_files
        
        # Format time estimate
        time_str = ""
        if est_processing_time < 60:
            time_str = f"about {int(est_processing_time)} seconds"
        else:
            minutes = int(est_processing_time / 60)
            seconds = int(est_processing_time % 60)
            time_str = f"about {minutes} minute{'s' if minutes > 1 else ''}"
            if seconds > 0:
                time_str += f", {seconds} second{'s' if seconds > 1 else ''}"

        voice_desc = self.voice_descriptions.get(selected_voice, "Custom voice")
        if "elevenlabs_" in selected_voice:
            voice_name = "Eleven Labs: " + selected_voice.replace("elevenlabs_", "")
        else:
            voice_name = selected_voice
        
        # Create a configuration summary table
        config_table = Table(show_header=False, box=box.SIMPLE_HEAD)
        config_table.add_column("Setting", style="bright_cyan", justify="right", width=20)
        config_table.add_column("Value", style="bright_yellow")
        
        config_table.add_row("Voice", f"{voice_name}")
        config_table.add_row("Description", f"[italic]{voice_desc}[/italic]")
        config_table.add_row(f"Speech Rate{'s' if len(selected_rates) > 1 else ''}", f"{', '.join([f'{rate}x' for rate in selected_rates])}")
        config_table.add_row(f"Format{'s' if len(selected_formats) > 1 else ''}", f"{', '.join([fmt.upper() for fmt in selected_formats])}")
        config_table.add_row("Files to Generate", f"{total_files}")
        config_table.add_row("Estimated Processing Time", f"{time_str}")
            
        console.print(
            Panel(
                config_table,
                title="✅ Your Configuration",
                title_align="center",
                title_style="bold bright_green",
                border_style="green",
                box=box.DOUBLE,
                padding=(1, 2),
                width=100
            )
        )

        return selected_voice, selected_formats, selected_rates

    def prompt_for_dummy_mode(self, non_interactive=False, cli_dummy=False) -> List[str]:
        """Prompt the user for whether to enable dummy mode or use CLI flag."""
        if non_interactive:
            # In non-interactive mode, use the CLI flag
            mode = "dummy" if cli_dummy else "standard"
            logger.info(f"Running in non-interactive mode, processing mode: {mode}")
            return [mode]

        # Create a visually appealing selection panel for dummy mode with consistent width
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
                width=100,
            )
        )

        # Show a table explaining processing modes with consistent width
        mode_table = Table(
            show_header=True, 
            header_style="bold magenta", 
            box=box.ROUNDED,
            width=100,
            padding=(0, 1)
        )
        mode_table.add_column("Option", style="dim", justify="center", width=8)
        mode_table.add_column("Mode", style="cyan", width=15)
        mode_table.add_column("Description", style="green", width=52)
        mode_table.add_column("Reading Level", style="yellow", justify="center", width=18)

        mode_table.add_row(
            "1",
            "Standard Mode",
            "Maintains original language while optimizing for speech",
            "College/Professional",
        )
        mode_table.add_row(
            "2",
            "Dummy Mode",
            "Rewrites everything in super-simple language a 13-year-old could understand",
            "Middle School",
        )
        mode_table.add_row(
            "3",
            "Literal Mode",
            "No preprocessing - keeps text exactly as written except for formatting incompatible with speech",
            "Unchanged",
        )

        console.print(mode_table)
        
        console.print("\n[cyan]You can select multiple processing modes (comma-separated) to generate multiple versions.[/cyan]")

        # If the CLI flag was set, use it as the default selection
        default_choice = "2" if cli_dummy else "1"

        mode_choices = []
        valid_input = False
        
        while not valid_input:
            try:
                mode_input = console.input(
                    f"\n[bold yellow]Select processing mode(s) (1-3, comma-separated) [{default_choice}]: [/bold yellow]"
                ) or default_choice
                
                # Parse the input - support for comma separated and range notation
                parts = mode_input.split(',')
                mode_choices = []
                
                for part in parts:
                    part = part.strip()
                    
                    # Check if it's a range (e.g., "1-3")
                    if '-' in part:
                        start, end = map(int, part.split('-'))
                        mode_choices.extend(range(start, end + 1))
                    else:
                        # Single number
                        mode_choices.append(int(part))
                
                # Validate all entries
                all_valid = True
                for idx in mode_choices:
                    if idx < 1 or idx > 3:
                        console.print(f"[bold red]Invalid option: {idx} - must be between 1 and 3[/bold red]")
                        all_valid = False
                        break
                
                if all_valid:
                    valid_input = True
                    
            except ValueError:
                console.print("[bold red]Please enter valid numbers (e.g., 1,2,3 or 1-3)[/bold red]")
        
        # Create mapping of selected modes
        selected_modes = {
            1: "standard", 
            2: "dummy",
            3: "literal"
        }
        
        # Extract selected modes
        processing_modes = [selected_modes[idx] for idx in mode_choices]
        
        # Display the selected modes for confirmation
        mode_names = ["Standard" if m == "standard" else "Dummy" if m == "dummy" else "Literal" for m in processing_modes]
        console.print(f"[green]Selected modes: {', '.join(mode_names)}[/green]")

        return processing_modes

    def preprocess_text(self, input_text: str, processing_mode="standard") -> Tuple[str, Dict]:
        """Preprocess text using Claude to optimize it for speech.
        
        Args:
            input_text: The text to preprocess
            processing_mode: Mode to use - 'standard', 'dummy', or 'literal'
        
        Returns:
            Tuple of (processed_text, usage_metrics)
        """
        # Calculate input text metrics
        char_count = len(input_text)
        word_count = len(input_text.split())
        paragraph_count = len(input_text.split("\n\n"))
        estimated_input_tokens = max(1, char_count // CHARS_PER_TOKEN_APPROX)
        
        # Handle literal mode specially - minimal processing
        if processing_mode == "literal":
            console.print(
                Panel(
                    f"[cyan]Using Literal Mode - text will be kept as-is with minimal formatting changes.[/cyan]",
                    title="Step 2: Text Preprocessing (Literal Mode)",
                    border_style="green",
                    padding=(1, 2),
                    width=100
                )
            )
            
            # Only process formatting that is incompatible with speech
            processed_text = input_text
            
            # Replace URLs with brief descriptions
            processed_text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', 
                                    "[URL link]", processed_text)
            
            # Create empty metrics with zero cost since we're not using Claude
            usage_metrics = {
                "input_tokens": 0,
                "output_tokens": 0,
                "input_cost": 0.0,
                "output_cost": 0.0,
                "total_cost": 0.0,
            }
            
            # Calculate and display change metrics
            processed_char_count = len(processed_text)
            processed_word_count = len(processed_text.split())
            processed_paragraph_count = len(processed_text.split("\n\n"))
            
            # Display stats table
            stats_table = Table(
                title="Text Preprocessing Results (Literal Mode)",
                show_header=True,
                box=box.ROUNDED,
            )
            stats_table.add_column("Metric", style="cyan")
            stats_table.add_column("Original", style="yellow", justify="right")
            stats_table.add_column("Processed", style="green", justify="right")
            stats_table.add_column("Change", style="magenta", justify="right")

            char_diff = processed_char_count - char_count
            char_pct_change = (char_diff / char_count * 100) if char_count > 0 else 0
            word_diff = processed_word_count - word_count
            word_pct_change = (word_diff / word_count * 100) if word_count > 0 else 0
            
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
            console.print("[green]Literal mode processing complete - minimal changes made to preserve original text[/green]")
            
            return processed_text, usage_metrics
            
        # For standard and dummy modes, use Claude
        # Set the processing mode name
        mode_name = "Dummy Mode" if processing_mode == "dummy" else "Standard Mode"

        console.print(
            Panel(
                f"[cyan]Claude 3.7 will now preprocess the text in {mode_name} to optimize it for speech while preserving all important content.[/cyan]",
                title=f"Step 2: Text Preprocessing ({mode_name})",
                border_style="green",
                padding=(1, 2),
                width=100
            )
        )

        # Create a more attractive progress display with pulsing effects
        with Progress(
            SpinnerColumn(spinner_name="dots12"),
            TextColumn("[bold bright_blue]{task.description}[/bold bright_blue]"),
            BarColumn(bar_width=40, complete_style="cyan", finished_style="green"),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            # Initialize progress variables
            start_time = time.time()
            received_chars = 0
            total_task = progress.add_task(
                f"✨ Claude 3.7 Processing: 0/{char_count:,} characters",
                total=char_count,
            )

            logger.info(
                f"Preprocessing text with Claude 3.7 in {mode_name}: {char_count} characters, {word_count} words"
            )

            # Create a very specific prompt for Claude based on the mode
            if processing_mode == "dummy":
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
            elif processing_mode == "standard":
                system_prompt = """
                You are a text preprocessing expert tasked with preparing text for text-to-speech conversion.

                CRITICAL INSTRUCTIONS:
                1. DO NOT remove any substantial content - keep all meaningful information, arguments, data points, and insights.
                2. DO remove formatting that doesn't make sense when read aloud:
                   - URLs: Replace with brief descriptions like "[Link to research paper]" only when the URL appears as a raw link
                   - Markdown formatting: Remove *, #, -, _, ~, >, etc. symbols but preserve the underlying text and structure
                   - Code blocks: Convert to natural language descriptions if brief, or indicate "[Code section omitted]" for lengthy blocks
                   - Tables: Convert simple tables to natural language, using phrases like "The data shows the following values..."
                3. Format numbers and abbreviations for speech:
                   - ALWAYS expand abbreviations: "200k" → "200,000 dollars" or "200,000"
                   - Add "dollars" or "percent" when referring to currency or percentages
                   - Write numbers in full: "5M" → "5 million" 
                   - Replace special characters with their spoken equivalents (e.g., % → "percent")
                4. Improve text flow and readability:
                   - Add natural transitions between sections
                   - Convert headers to spoken-word phrases ("Chapter 1: Introduction" or "Moving on to the next section about...")
                   - Spell out abbreviations on first use when helpful for clarity
                   - Use "and" instead of "&" and other speech-friendly replacements
                5. Make no other substantive changes - do not summarize, rewrite, or alter the substance of the text
                6. Your output should be a clean, flowing script ready to be converted to audio that reads like a professional audiobook

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
                            # Calculate percentage for a more meaningful display
                            pct = min(100, int((received_chars / max(1, char_count)) * 100))
                            
                            # Use emojis to indicate progress stages
                            emoji = "🤔" if pct < 25 else "✍️" if pct < 50 else "🔍" if pct < 75 else "✨" 
                            
                            progress.update(
                                total_task,
                                completed=min(received_chars, char_count),
                                description=f"{emoji} Claude 3.7 Processing: {received_chars:,}/{char_count:,} characters",
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
                if processing_mode == "dummy" and char_percent_change < -15:
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
                    Panel(
                        f"[dim]• Processing Speed: {received_chars / max(1, elapsed):.1f} chars/sec\n"
                        f"• Time: {elapsed:.1f}s total[/dim]",
                        border_style="dim",
                        padding=(1, 2),
                        width=100,
                    )
                )

                mode_name = "Dummy Mode" if processing_mode == "dummy" else "Standard Mode"
                logger.info(
                    f"Text preprocessing complete in {mode_name}. Original: {char_count} chars, Processed: {processed_char_count} chars."
                )

                # Compare length to ensure we didn't lose too much content
                retention_rate = (
                    processed_word_count / word_count * 100 if word_count > 0 else 100
                )

                if retention_rate < 85:
                    console.print(
                        Panel(
                            f"[bold yellow]Warning:[/bold yellow] Processed text is {retention_rate:.1f}% of the original length. Some content may have been lost.",
                            border_style="yellow",
                            box=box.DOUBLE,
                            padding=(1, 2),
                            width=100,
                        )
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
            # Determine if this is an Eleven Labs voice
            is_elevenlabs = voice.startswith("elevenlabs_")
            
            # Use output format based on the file extension
            output_format = "mp3" if temp_file.endswith(".mp3") else "wav"
            
            # Check if Eleven Labs is available and API key is valid
            use_elevenlabs = is_elevenlabs and self.elevenlabs_available and hasattr(self, 'elevenlabs_key_valid') and self.elevenlabs_key_valid
            
            # If Eleven Labs is requested but known to be invalid, prompt for confirmation
            if is_elevenlabs and not use_elevenlabs and not hasattr(self, 'use_fallback_confirmed'):
                # Only ask once per session
                self.use_fallback_confirmed = Confirm.ask(
                    "[bold yellow]WARNING:[/bold yellow] Your Eleven Labs API key is invalid or has expired. "
                    "Would you like to continue using OpenAI's onyx voice as a fallback?",
                    default=True
                )
                
                # If user doesn't want to use fallback, raise exception to stop processing
                if not self.use_fallback_confirmed:
                    console.print("\n[bold cyan]How to fix your Eleven Labs API key:[/bold cyan]")
                    console.print("1. Visit https://elevenlabs.io/app/account/api-key to get your current API key")
                    console.print("2. Run the script again and when prompted, enter your new API key")
                    console.print("   or update your API key in ~/.speech_config/api_keys.json")
                    
                    raise Exception(
                        "Text-to-speech processing cancelled. Please update your Eleven Labs API key and try again."
                    )
                else:
                    console.print("[green]Continuing with OpenAI's onyx voice as fallback[/green]")
                    console.print("\n[yellow]Note:[/yellow] Your saved Eleven Labs API key doesn't work. "
                                 "Next time you run the script, consider updating your API key.")
            
            if use_elevenlabs:
                # Extract the voice ID from the voice string (format: elevenlabs_VOICE_ID)
                elevenlabs_voice_id = voice.replace("elevenlabs_", "")
                
                # Generate audio with Eleven Labs
                # Going directly to API call since elevenlabs package structure varies widely between versions
                try:
                    # Direct API call with requests
                    logger.info("Using direct API call for Eleven Labs")
                    import requests
                    
                    url = f"https://api.elevenlabs.io/v1/text-to-speech/{elevenlabs_voice_id}"
                    headers = {
                        "Accept": "audio/mpeg",
                        "Content-Type": "application/json",
                        "xi-api-key": self.elevenlabs_api_key
                    }
                    data = {
                        "text": chunk,
                        "model_id": "eleven_multilingual_v2",
                        "voice_settings": {
                            "stability": 0.5,
                            "similarity_boost": 0.5
                        }
                    }
                    
                    # Log request details for debugging (without API key)
                    masked_headers = headers.copy()
                    if "xi-api-key" in masked_headers:
                        masked_headers["xi-api-key"] = "*****"
                    logger.info(f"Eleven Labs API request to: {url}")
                    logger.info(f"Headers: {masked_headers}")
                    
                    # Make the request
                    response = requests.post(url, json=data, headers=headers)
                    
                    # Check for errors
                    if response.status_code != 200:
                        logger.error(f"Eleven Labs API error: {response.status_code} - {response.text}")
                        raise Exception(f"Eleven Labs API error: {response.status_code}")
                    
                    # Save the audio data
                    with open(temp_file, "wb") as f:
                        f.write(response.content)
                    
                    logger.info(f"Successfully generated audio with Eleven Labs API: {len(response.content)} bytes written to {temp_file}")
                    return temp_file
                        
                except Exception as e:
                    logger.error(f"Eleven Labs API request failed: {e}")
                    logger.warning("Falling back to OpenAI TTS with onyx-hd voice")
                    # Fall back to OpenAI - don't raise an exception
            
            # Default or fallback to OpenAI TTS
            # If this was an Eleven Labs voice that failed, use onyx-hd as fallback model but "onyx" as voice parameter
            # OpenAI doesn't accept "-hd" in the voice parameter
            fallback_voice = "onyx"  # Base voice name only - OpenAI doesn't accept "voice-hd" format
            fallback_model = "tts-1-hd" if output_format == "wav" or output_format == "flac" else "tts-1"
            
            # For OpenAI voices, we need to remove the "-hd" suffix when sending to the API
            # The HD vs. standard is determined by the model parameter, not the voice parameter
            api_voice_clean = api_voice.replace("-hd", "") if not is_elevenlabs else api_voice
            
            # If we reached here, either:
            # 1. OpenAI was explicitly selected from the start
            # 2. Eleven Labs failed and we're falling back to OpenAI
            try:
                voice_to_use = fallback_voice if is_elevenlabs else api_voice_clean
                model_to_use = fallback_model if is_elevenlabs else model
                
                logger.info(f"Using OpenAI TTS service with {voice_to_use} voice and {model_to_use} model")
                response = self.openai_client.audio.speech.create(
                    model=model_to_use,
                    voice=voice_to_use,
                    input=chunk,
                    response_format=output_format,
                    speed=1.0  # Normal speech rate applied at base level - we adjust speed separately later
                )
                response.stream_to_file(temp_file)
                
                if is_elevenlabs:
                    logger.info(f"Successfully generated fallback audio with OpenAI {fallback_voice}")
            except Exception as e:
                logger.error(f"OpenAI TTS failed: {e}")
                raise Exception(f"Failed to generate audio: {e}")
                
            return temp_file
        except Exception as e:
            logger.error(f"Error processing chunk: {e}")
            raise

    def text_to_speech(
        self, text: str, voice: str, output_file: str, speech_rate: float = 1.0
    ) -> Tuple[str, dict]:
        """
        Convert text to speech using the selected voice API with intelligent chunking and parallel processing.

        Args:
            text: The processed text to convert to speech
            voice: The selected voice
            output_file: Path to the output audio file
            speech_rate: Speed multiplier for speech (0.8-1.5x)

        Returns:
            Tuple containing the output file path and audio metadata
        """
        # Determine which TTS provider to use
        is_elevenlabs = voice.startswith("elevenlabs_")
        
        if is_elevenlabs:
            # Using Eleven Labs
            model = "eleven_multilingual_v2"
            api_voice = voice  # Keep the full voice ID for Eleven Labs
            is_hd = True  # Eleven Labs voices are considered high quality
            # Price is based on the model used
            price_per_million = ELEVENLABS_STANDARD_PRICE_PER_MILLION
            
            # Check if Eleven Labs integration is working
            elevenlabs_working = hasattr(self, 'elevenlabs_key_valid') and self.elevenlabs_key_valid
            if not elevenlabs_working:
                # If Eleven Labs isn't working, warn the user and confirm if they want to continue with fallback
                logger.warning("Eleven Labs API key invalid, may need to fallback to OpenAI")
                self.use_fallback_confirmed = False  # Will be set to True if user confirms
                
                # We'll prompt for confirmation at the point of actual conversion
        else:
            # Using OpenAI TTS
            is_hd = "-hd" in voice
            model = "tts-1-hd" if is_hd else "tts-1"
            # For HD voices, remove the -hd suffix for the actual API call
            # OpenAI doesn't accept "voice-hd" format in the voice parameter
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
                f"[cyan]Converting text to speech using {'Eleven Labs' if is_elevenlabs else 'OpenAI'}'s TTS API with {voice.split('_')[-1] if is_elevenlabs else voice} voice at {speech_rate}x speed...[/cyan]",
                title="Step 5: Text-to-Speech Conversion",
                border_style="green",
                padding=(1, 2),
                width=100,
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
            SpinnerColumn(spinner_name="dots12"),
            TextColumn("[bold bright_blue]{task.description}[/bold bright_blue]"),
            BarColumn(bar_width=40, complete_style="bright_cyan", finished_style="bright_green"),
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
                # Limit to 5 concurrent chunks to avoid API rate limits but improve throughput
                max_workers = min(5, total_chunks) if total_chunks > 0 else 1
                
                # Add more detailed progress information
                progress.update(
                    process_task,
                    description=f"Processing {total_chunks} audio chunks with {max_workers} parallel workers",
                )

                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=max_workers
                ) as executor:
                    # Map of future to chunk index to track order
                    future_to_index = {}
                    active_futures = set()  # Track currently processing tasks
                    chunk_status = ["Pending" for _ in range(total_chunks)]  # Track status of each chunk
                    
                    # Start with the first batch of chunks
                    next_chunk = 0
                    
                    # Process chunks with improved error handling
                    while next_chunk < total_chunks or active_futures:
                        # Submit new tasks up to max_workers
                        while next_chunk < total_chunks and len(active_futures) < max_workers:
                            chunk_idx = next_chunk
                            chunk = chunks[chunk_idx]
                            temp_file = f"{output_file}.part{chunk_idx}"
                            
                            future = executor.submit(
                                self.process_chunk,
                                chunk,
                                voice,
                                model,
                                api_voice,
                                temp_file,
                            )
                            future_to_index[future] = chunk_idx
                            active_futures.add(future)
                            chunk_status[chunk_idx] = "Processing"
                            next_chunk += 1
                            
                            # Update progress with detailed status
                            progress_desc = f"Processing audio chunks: {processed_chunks}/{total_chunks} complete"
                            progress.update(process_task, description=progress_desc)
                        
                        # Wait for at least one task to complete
                        if active_futures:
                            # Get results as they complete
                            done, active_futures = concurrent.futures.wait(
                                active_futures, 
                                return_when=concurrent.futures.FIRST_COMPLETED
                            )
                            
                            # Process completed tasks
                            for future in done:
                                try:
                                    chunk_index = future_to_index[future]
                                    temp_file = future.result()
                                    temp_files[chunk_index] = temp_file
                                    processed_chunks += 1
                                    chunk_status[chunk_index] = "Complete"
                                    
                                    # Update progress
                                    progress_desc = f"Processing audio chunks: {processed_chunks}/{total_chunks} complete"
                                    if processed_chunks < total_chunks:
                                        next_status = " | ".join([f"Chunk {i}: {status}" for i, status in enumerate(chunk_status) if status != "Complete"][:3])
                                        if len([s for s in chunk_status if s != "Complete"]) > 3:
                                            next_status += " | ..."
                                        progress_desc += f" | {next_status}"
                                    
                                    progress.update(
                                        process_task,
                                        advance=1,
                                        description=progress_desc,
                                    )
                                    
                                except Exception as e:
                                    # Handle errors for this chunk
                                    chunk_index = future_to_index[future]
                                    logger.error(f"Error processing chunk {chunk_index}: {e}")
                                    chunk_status[chunk_index] = f"Failed: {e}"
                                    processed_chunks += 1
                                    
                                    # Create an empty file to maintain order
                                    empty_file = f"{output_file}.error_part{chunk_index}"
                                    Path(empty_file).touch()
                                    temp_files[chunk_index] = empty_file
                                    
                                    progress.update(
                                        process_task,
                                        advance=1,
                                        description=f"ERROR on chunk {chunk_index}: {str(e)[:30]}... | {processed_chunks}/{total_chunks}",
                                    )

                # Add a merging task
                merge_task = progress.add_task(
                    "Combining audio segments", total=len(temp_files)
                )

                # Combine all chunks using pydub with enhanced error handling
                combine_start = time.time()
                combined = AudioSegment.empty()
                segments_merged = 0
                empty_segments = 0

                for i, temp_file in enumerate(temp_files):
                    try:
                        # Check if this is an error placeholder
                        if "error_part" in temp_file:
                            # Create a small silence segment instead
                            logger.warning(f"Using 1 second silence for failed segment {i}")
                            segment = AudioSegment.silent(duration=1000)  # 1 second of silence
                            empty_segments += 1
                        else:
                            # Load the audio segment
                            segment = AudioSegment.from_file(temp_file)
                            segments_merged += 1
                        
                        # Add to the combined segment
                        combined += segment
                        
                        # Update progress
                        progress.update(
                            merge_task,
                            advance=1,
                            description=f"Merging segment {i+1}/{len(temp_files)} | {segments_merged} successful, {empty_segments} silent",
                        )
                    except Exception as e:
                        logger.error(f"Error processing audio segment {i}: {e}")
                        console.print(
                            f"[bold bright_red]Warning:[/bold bright_red] Error processing segment {i}, inserting silence instead."
                        )
                        # Insert silence for the failed segment
                        segment = AudioSegment.silent(duration=1000)  # 1 second of silence
                        combined += segment
                        empty_segments += 1
                        progress.update(
                            merge_task,
                            advance=1,
                            description=f"Merged silent segment for error {i+1}/{len(temp_files)}",
                        )
                
                # Show final merging summary if there were errors
                if empty_segments > 0:
                    console.print(
                        Panel(
                            f"[bold yellow]Warning: {empty_segments} segments failed and were replaced with silence.[/bold yellow]",
                            border_style="yellow",
                            box=box.DOUBLE,
                            padding=(1, 2),
                            width=100,
                        )
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

                # Add necessary import for subprocess at the top if not already present
                import subprocess
                
                # Determine output format and apply appropriate high-quality bitrate settings
                output_format = output_file.split(".")[-1]
                
                # First export to a temp file
                temp_export_file = f"{output_file}.temp_export"
                combined.export(temp_export_file, format=output_format)
                
                # Use FFmpeg to apply high bitrate settings if available
                try:
                    # Check if FFmpeg is installed
                    result = subprocess.run(["which", "ffmpeg"], capture_output=True, text=True)
                    ffmpeg_available = len(result.stdout.strip()) > 0
                    
                    if ffmpeg_available:
                        # Update task description
                        progress.update(export_task, description="Enhancing audio quality with FFmpeg")
                        
                        ffmpeg_cmd = ["ffmpeg", "-y", "-i", temp_export_file]
                        
                        # Apply format-specific high-quality bitrate settings
                        if output_format == "mp3":
                            # Use high-quality VBR encoding for MP3
                            ffmpeg_cmd.extend(["-codec:a", "libmp3lame", "-q:a", "0", output_file])
                        elif output_format == "aac":
                            # Use high-quality AAC encoding
                            ffmpeg_cmd.extend(["-codec:a", "aac", "-b:a", "256k", output_file])
                        elif output_format == "flac":
                            # For FLAC, set highest compression level
                            ffmpeg_cmd.extend(["-codec:a", "flac", "-compression_level", "12", output_file])
                        elif output_format == "wav":
                            # For WAV, ensure 24-bit PCM for HD audio
                            ffmpeg_cmd.extend(["-codec:a", "pcm_s24le", output_file])
                        else:
                            # For other formats, just copy
                            ffmpeg_cmd.extend(["-codec:a", "copy", output_file])
                        
                        # Run FFmpeg for high-quality output
                        subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                        logger.info(f"Enhanced audio quality using FFmpeg with high-quality settings for {output_format}")
                        
                        # Update file size after FFmpeg processing
                        file_size_bytes = os.path.getsize(output_file)
                        file_size_mb = file_size_bytes / (1024 * 1024)
                        
                        # Remove the temp file
                        os.remove(temp_export_file)
                    else:
                        # If FFmpeg not available, just use the original export
                        logger.info("FFmpeg not found; using standard pydub export without bitrate optimization")
                        os.rename(temp_export_file, output_file)
                except Exception as e:
                    logger.warning(f"Error applying high-quality bitrate settings with FFmpeg: {e}")
                    logger.info("Falling back to standard pydub export")
                    # If FFmpeg fails, use the original exported file
                    if os.path.exists(temp_export_file):
                        os.rename(temp_export_file, output_file)
                
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

                # Determine if we had to use a fallback for Eleven Labs
                used_fallback = is_elevenlabs and (not hasattr(self, 'elevenlabs_key_valid') or not self.elevenlabs_key_valid)
                actual_model = model
                actual_voice = api_voice
                
                # Define fallback values
                fallback_model = "tts-1-hd" if output_format == "wav" or output_format == "flac" else "tts-1"
                fallback_voice = "onyx" # Use base voice name without HD suffix for OpenAI
                
                # If we used a fallback, update the model and voice info to reflect what was actually used
                if used_fallback:
                    actual_model = fallback_model
                    actual_voice = fallback_voice
                
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
                    "model": actual_model,
                    "is_hd": "-hd" in actual_model if isinstance(actual_model, str) else is_hd,
                    "price_per_million": price_per_million,
                    "tts_cost": float(tts_cost_display),
                    "cost_per_minute": float(cost_per_minute_display),
                    "used_fallback": used_fallback,
                    "intended_voice": voice,
                    "actual_voice": actual_voice 
                }

            except Exception as e:
                logger.error(f"Error during text-to-speech conversion: {e}")
                raise

        # Display a compact summary
        # Add notes about quality and fallbacks
        notes = []
        
        # Add HD quality note if applicable
        if is_hd:
            # Different message based on format
            if output_file.endswith(".wav") or output_file.endswith(".flac"):
                notes.append(f"[bold cyan]Note:[/bold cyan] HD quality audio with {output_file.split('.')[-1].upper()} format will provide the highest fidelity.")
            else:
                notes.append(f"[bold yellow]Note:[/bold yellow] For maximum HD quality, consider using WAV or FLAC format next time.")
        
        # Add fallback note if Eleven Labs failed
        if metadata.get("used_fallback", False):
            notes.append(f"[bold red]Warning:[/bold red] Eleven Labs API key was invalid. Used OpenAI {metadata.get('actual_voice', 'onyx-hd')} voice as fallback.")
        
        # Combine all notes
        hd_quality_note = ""
        if notes:
            hd_quality_note = "\n" + "\n".join(notes)
        
        console.print(
            Panel(
                f"[bold bright_green]Audio Generation Complete![/bold bright_green]\n\n"
                f"[cyan]• Output:[/cyan] {output_file}\n"
                f"[cyan]• Audio Length:[/cyan] {metadata['audio_duration_formatted']} ({metadata['audio_duration']:.1f}s)\n"
                f"[cyan]• File Size:[/cyan] {metadata['file_size_mb']:.2f} MB\n"
                f"[cyan]• Speech Rate:[/cyan] {speech_rate}x\n"
                f"[cyan]• TTS Model:[/cyan] {metadata.get('model', model)}\n"
                f"[cyan]• Estimated Cost:[/cyan] ${metadata['tts_cost']:.4f}\n"
                f"[italic]• Processing Time: {total_time:.1f}s ({metadata['chars_per_second']:.1f} chars/sec)[/italic]"
                f"{hd_quality_note}",
                title="Audio Generated Successfully",
                border_style="green",
                box=box.DOUBLE,
                padding=(1, 2),
                width=100,
            )
        )

        # Display pricing information
        pricing_table = Table(
            title="OpenAI TTS API Usage Costs", 
            show_header=True, 
            box=box.ROUNDED,
            title_style="bold bright_blue",
            border_style="blue",
            header_style="bright_cyan"
        )
        pricing_table.add_column("Cost Item", style="cyan")
        pricing_table.add_column("Rate", style="yellow", justify="right")
        pricing_table.add_column("Usage", style="bright_blue", justify="right")
        pricing_table.add_column("Estimated Cost", style="bright_green", justify="right")

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
        """Process a text file through the entire pipeline with user interaction.
        
        Args:
            input_file: Path to the text file to process
            non_interactive: Whether to run in non-interactive mode with defaults
            dummy_mode: Legacy parameter - whether to use dummy mode (simplified language)
        """
        # Show an elegant welcome banner with gradient-like styling
        welcome_title = "🎙️ [bold]TTS Processor[/bold] 🎙️"
        
        # Create a rainbow gradient style for the welcome message
        console.print()
        
        # Create a gradient banner with a more vibrant style
        banner = Panel(
            Group(
                Text("Text-to-Speech Processor", style="bold bright_blue") + Text(" v1.2", style="bright_black"),
                Text("✨ Convert text files to professional-sounding speech with AI optimization ✨", style="bold bright_cyan"),
                Text(""),  # Empty line for spacing
                Text("• ", style="bright_cyan") + Text("Smart AI Preprocessing", style="bold bright_cyan") + Text(": Claude 3.7 AI optimizes text for natural speech", style="bright_cyan"),
                Text("• ", style="bright_green") + Text("Premium Voice Options", style="bold bright_green") + Text(": Use OpenAI or your custom Eleven Labs voices", style="bright_green"),
                Text("• ", style="bright_yellow") + Text("Multiple Output Formats", style="bold bright_yellow") + Text(": MP3, WAV, FLAC, or AAC with customizable settings", style="bright_yellow"),
                Text("• ", style="bright_magenta") + Text("Intelligent Caching", style="bold bright_magenta") + Text(": Reuse processed text to save time and API costs", style="bright_magenta"),
            ),
            title=welcome_title,
            subtitle="by Kris Kibak",
            title_align="center",
            border_style="bright_blue",
            box=box.DOUBLE,
            padding=(1, 2),
            width=100,
        )
        
        console.print(banner)

        # Start tracking all process metrics for final report
        process_metrics = {
            "start_time": time.time(),
            "input_file": input_file,
        }

        # Read input file with nicer UI
        input_path = Path(input_file)
        
        # Create stylized input file display
        file_size_bytes = os.path.getsize(input_file)
        file_size_kb = file_size_bytes / 1024
        file_size_display = f"{file_size_kb:.1f} KB" if file_size_kb < 1024 else f"{file_size_kb/1024:.2f} MB"
        file_modified = time.strftime("%Y-%m-%d %H:%M", time.localtime(os.path.getmtime(input_file)))
        
        # Read a short preview of the file content
        file_preview = ""
        try:
            with open(input_file, "r", encoding="utf-8") as f:
                preview_text = f.read(400)  # Read first 400 chars for a reasonable preview
                # Clean up the preview for display
                preview_text = re.sub(r'[\r\n]+', ' ', preview_text)  # Replace newlines with spaces
                if len(preview_text) >= 400:
                    preview_text = preview_text[:397] + "..."
                file_preview = preview_text
        except Exception as e:
            file_preview = f"[italic](Preview unavailable: {str(e)})[/italic]"

        # Get the absolute path
        absolute_path = str(input_path.absolute())
        
        # Create a file info table
        file_info_table = Table(show_header=False, box=box.SIMPLE_HEAD)
        file_info_table.add_column("Property", style="bright_cyan", justify="right", width=12)
        file_info_table.add_column("Value", style="bright_yellow")
        
        file_info_table.add_row("File", f"{input_path.name}")
        file_info_table.add_row("Path", f"[italic]{absolute_path}[/italic]")
        file_info_table.add_row("Size", f"{file_size_display}")
        file_info_table.add_row("Modified", f"{file_modified}")
                
        console.print(
            Panel(
                Group(
                    file_info_table,
                    Panel(
                        f"\"{file_preview}\"",
                        title="Preview",
                        title_align="left",
                        border_style="bright_blue",
                        padding=(0, 1)
                    )
                ),
                title="📄 Input File",
                title_align="center",
                border_style="bright_blue",
                box=box.ROUNDED,
                padding=(1, 2),
                width=100
            )
        )

        try:
            with open(input_file, "r", encoding="utf-8") as f:
                input_text = f.read()

            # Show text stats
            char_count = len(input_text)
            word_count = len(input_text.split())
            paragraph_count = len(input_text.split("\n\n"))
            
            # Add reading time estimate (average 200 words per minute)
            reading_time_minutes = word_count / 200
            reading_time_str = ""
            
            if reading_time_minutes < 1:
                reading_time_str = f"Less than 1 minute"
            elif reading_time_minutes < 60:
                reading_time_str = f"About {int(reading_time_minutes)} minute{'s' if int(reading_time_minutes) != 1 else ''}"
            else:
                hours = int(reading_time_minutes / 60)
                minutes = int(reading_time_minutes % 60)
                reading_time_str = f"About {hours} hour{'s' if hours != 1 else ''}"
                if minutes > 0:
                    reading_time_str += f", {minutes} minute{'s' if minutes != 1 else ''}"

            # Save input text metrics
            process_metrics["input_text"] = {
                "char_count": char_count,
                "word_count": word_count,
                "paragraph_count": paragraph_count,
                "file_size_bytes": file_size_bytes,
                "file_size_mb": file_size_bytes / (1024 * 1024),
                "reading_time_minutes": reading_time_minutes,
            }
            
            # Store character count for use in time estimation
            self.char_count = char_count

            # Create a table for text stats
            stats_table = Table(show_header=False, box=box.SIMPLE_HEAD)
            stats_table.add_column("Metric", style="bright_cyan", justify="right", width=15)
            stats_table.add_column("Value", style="bright_yellow")
            
            stats_table.add_row("Characters", f"{char_count:,}")
            stats_table.add_row("Words", f"{word_count:,}")
            stats_table.add_row("Paragraphs", f"{paragraph_count:,}")
            stats_table.add_row("Reading Time", reading_time_str)
            
            console.print(
                Panel(
                    stats_table,
                    title="📊 Text Statistics",
                    title_align="center",
                    border_style="cyan",
                    box=box.ROUNDED,
                    width=100,
                    padding=(1, 2)
                )
            )
        except Exception as e:
            console.print(f"[bold red]Error reading input file:[/bold red] {e}")
            return

        # Show info about dummy mode if the legacy parameter was passed
        if dummy_mode and "dummy" not in processing_modes:  # Only show this if we haven't already converted to the new format
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
                    width=100
                )
            )

        voice, selected_formats, selected_rates = self.display_voice_options(non_interactive)

        # Save voice selection metrics
        process_metrics["voice_settings"] = {
            "voice": voice,
            "output_formats": selected_formats,
            "speech_rates": selected_rates,
        }

        # Prompt for processing modes
        processing_modes = self.prompt_for_dummy_mode(non_interactive, dummy_mode)
        
        # If legacy dummy_mode parameter was used, convert to the new format
        if isinstance(processing_modes, bool):
            processing_modes = ["dummy" if processing_modes else "standard"]
            
        # Save processing modes to metrics
        process_metrics["processing_modes"] = processing_modes
        
        # Define our audio output combinations
        audio_configurations = []
        
        # Iterate through all combinations of modes, rates, and formats to generate all selected versions
        for mode in processing_modes:
            for rate in selected_rates:
                for format in selected_formats:
                    audio_configurations.append({
                        "processing_mode": mode,
                        "speech_rate": rate,
                        "output_format": format
                    })
        
        # Display the number of audio files that will be generated with more visual appeal
        modes_str = ", ".join([f"{m.capitalize()}" for m in processing_modes])
        rates_str = ", ".join([f"{r}x" for r in selected_rates])
        formats_str = ", ".join([f.upper() for f in selected_formats])
        
        console.print(
            Panel(
                f"[bold green]Audio Generation Plan[/bold green]\n\n"
                f"[cyan]Total Output Files:[/cyan] [bold yellow]{len(audio_configurations)}[/bold yellow]\n\n"
                f"[cyan]Processing Mode{'s' if len(processing_modes) > 1 else ''}:[/cyan] {modes_str}\n"
                f"[cyan]Speech Rate{'s' if len(selected_rates) > 1 else ''}:[/cyan] {rates_str}\n"
                f"[cyan]Output Format{'s' if len(selected_formats) > 1 else ''}:[/cyan] {formats_str}\n\n"
                f"[italic]Each unique combination of the above settings will produce one audio file.[/italic]",
                title="🔊 Output Files",
                border_style="cyan",
                padding=(1, 2),
                width=100,
            )
        )
        
        # Generate a hash of just the input text for checking preprocessed text cache
        text_hash = hashlib.md5(input_text.encode()).hexdigest()
        text_cache_dir = self.cache_dir / f"processed_{text_hash}"
        
        # Check for cached preprocessed text files for this input text
        cached_processed_text = {}
        cached_text_exists = False
        
        if text_cache_dir.exists():
            cached_text_exists = True
            # Look for cached processed text files for each mode
            for mode in processing_modes:
                cached_file = text_cache_dir / f"processed_{mode}.txt"
                if cached_file.exists():
                    try:
                        cached_processed_text[mode] = cached_file.read_text(encoding="utf-8")
                    except Exception as e:
                        logger.warning(f"Error reading cached processed text file: {e}")
        
        # For checking audio cache, use the first configuration
        first_config = audio_configurations[0]
        model = "tts-1-hd" if "-hd" in voice else "tts-1"
        content_hash = self._get_content_hash(input_text, voice, model, first_config["output_format"])
        cache_data = self._check_cache(content_hash)

        if not non_interactive:
            # Determine what's cached and offer smart reuse options
            if cache_data and cached_text_exists:
                # Both text preprocessing and audio conversion are cached
                console.print(
                    Panel(
                        "[bold yellow]Cache Found![/bold yellow]\n\n"
                        f"[cyan]1. Preprocessed text is available[/cyan] - Text has already been processed by Claude\n"
                        f"[cyan]2. Audio files are available[/cyan] - Previous directory: [bold]{cache_data['output_dir']}[/bold]\n\n"
                        "You can choose to:\n"
                        "• Skip both (use existing files)\n"
                        "• Reuse processed text but create new audio with different voices/formats\n"
                        "• Process everything from scratch",
                        title="🔍 Smart Cache Detection",
                        border_style="yellow",
                        padding=(1, 2),
                    )
                )
                
                # Create options
                options = [
                    "Use existing audio files (skip processing)",
                    "Reuse processed text but create new audio files",
                    "Process everything from scratch"
                ]
                
                # Present as a table
                options_table = Table(show_header=True, box=box.ROUNDED)
                options_table.add_column("Option", style="dim", justify="center", width=8)
                options_table.add_column("Action", style="cyan")
                options_table.add_column("Best For", style="green")
                
                options_table.add_row("1", options[0], "When you're happy with existing files")
                options_table.add_row("2", options[1], "When you want to try different voices or formats")
                options_table.add_row("3", options[2], "When you need completely fresh processing")
                
                console.print(options_table)
                
                # Get user choice
                choice = "0"
                while choice not in ["1", "2", "3"]:
                    choice = console.input("[bold yellow]Select an option (1-3): [/bold yellow]")
                
                if choice == "1":
                    console.print(f"[bold green]Using existing audio files from {cache_data['output_dir']}[/bold green]")
                    return
                elif choice == "2":
                    console.print("[bold green]Reusing preprocessed text but creating new audio files...[/bold green]")
                    # Use cached processed text
                    processed_texts = cached_processed_text
                    # Skip text preprocessing steps later
                    skip_text_preprocessing = True
                elif choice == "3":
                    console.print("[bold green]Processing everything from scratch...[/bold green]")
                    # Clear cache variables to force full processing
                    cached_text_exists = False
                    cached_processed_text = {}
            
            elif cached_text_exists:
                # Only text preprocessing is cached
                console.print(
                    Panel(
                        "[bold yellow]Preprocessed Text Cache Found![/bold yellow]\n\n"
                        f"[cyan]This text has already been processed by Claude.[/cyan]\n\n"
                        "You can choose to:\n"
                        "• Reuse processed text (saves time and Claude API costs)\n"
                        "• Process text from scratch",
                        title="🔍 Text Cache Detection",
                        border_style="yellow",
                        padding=(1, 2),
                        width=100,
                    )
                )
                
                reuse_text = Confirm.ask("Would you like to reuse the preprocessed text?", default=True)
                
                if reuse_text:
                    console.print("[bold green]Reusing preprocessed text...[/bold green]")
                    # Use cached processed text
                    processed_texts = cached_processed_text
                    # Skip text preprocessing steps later
                    skip_text_preprocessing = True
                else:
                    console.print("[bold green]Processing text from scratch...[/bold green]")
                    # Clear cache variables to force text processing
                    cached_text_exists = False
                    cached_processed_text = {}
            
            elif cache_data:
                # Only audio conversion is cached (unusual case)
                console.print(
                    Panel(
                        f"[bold yellow]Audio Cache Found![/bold yellow]\n\n"
                        f"Previous output directory: [cyan]{cache_data['output_dir']}[/cyan]\n\n"
                        f"Processing again will create a new set of output files.",
                        title="Cache Found",
                        border_style="yellow",
                        padding=(1, 2),
                        width=100,
                    )
                )

                reprocess = Confirm.ask("Do you want to process it again?", default=False)

                if not reprocess:
                    console.print("[bold green]Using cached results. Exiting.[/bold green]")
                    return
            
            # If we get here, either no cache or user chose to reprocess
            skip_text_preprocessing = cached_text_exists and len(cached_processed_text) > 0
            
        elif cache_data and non_interactive:
            # In non-interactive mode, always reprocess
            console.print(
                "[yellow]Cache found, but reprocessing in non-interactive mode.[/yellow]"
            )
            skip_text_preprocessing = False

        # Process text for each mode
        processed_texts = {}
        all_claude_metrics = {}
        
        # Ensure skip_text_preprocessing is defined
        skip_text_preprocessing = locals().get('skip_text_preprocessing', False)
            
        # Check if we can skip text preprocessing
        if skip_text_preprocessing and cached_processed_text:
            console.print(
                Panel(
                    "[bold green]Using Cached Text Processing[/bold green]\n"
                    "Skipping Claude preprocessing and using previously processed text.\n"
                    "[dim]This saves time and avoids redundant API charges.[/dim]",
                    title="⏩ Skipping Text Preprocessing",
                    border_style="green",
                    padding=(1, 2),
                )
            )
            
            # Verify all required modes are available in cache
            missing_modes = [mode for mode in processing_modes if mode not in cached_processed_text]
            if missing_modes:
                console.print(f"[yellow]Warning: Cached text missing for mode(s): {', '.join(missing_modes)}[/yellow]")
                console.print("[yellow]Will process these modes from scratch.[/yellow]")
            
            # Use cached text where available
            processed_texts = cached_processed_text.copy()
            
            # Create dummy metrics for cached modes to avoid errors
            for mode in cached_processed_text:
                if mode in processing_modes:
                    char_count = len(cached_processed_text[mode])
                    # Create comprehensive dummy metrics with all possible fields
                    all_claude_metrics[mode] = {
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "input_cost": 0.0,
                        "output_cost": 0.0,
                        "total_cost": 0.0,
                        "cached": True
                    }
        
        # Process each required mode once (to avoid duplicate processing)
        # This will either process all modes or just the ones missing from the cache
        modes_to_process = [mode for mode in processing_modes if mode not in processed_texts]
        for mode in modes_to_process:
            mode_name = "Dummy Mode" if mode == "dummy" else "Standard Mode" if mode == "standard" else "Literal Mode"
            
            if mode in ["standard", "dummy"]:
                console.print(
                    Panel(
                        f"[cyan]Claude 3.7 will now preprocess the text in {mode_name} to optimize it for speech while preserving all important content.[/cyan]",
                        title=f"Step 2: Text Preprocessing ({mode_name})",
                        border_style="green",
                        padding=(1, 2),
                        width=100,
                    )
                )
            
            # Process the text with the specific mode
            processed_text, claude_usage_metrics = self.preprocess_text(
                input_text, mode
            )
            
            # Save for each mode
            processed_texts[mode] = processed_text
            all_claude_metrics[mode] = claude_usage_metrics
            
            # Save to the text cache directory for future use
            try:
                # Create cache directory if it doesn't exist
                text_cache_dir.mkdir(exist_ok=True, parents=True)
                
                # Save processed text to cache
                cached_file = text_cache_dir / f"processed_{mode}.txt"
                cached_file.write_text(processed_text, encoding="utf-8")
                logger.info(f"Saved processed {mode} text to cache: {cached_file}")
            except Exception as e:
                logger.warning(f"Failed to save processed text to cache: {e}")
            
            # Save processed text metrics
            process_metrics[f"processed_text_{mode}"] = {
                "char_count": len(processed_text),
                "word_count": len(processed_text.split()),
                "paragraph_count": len(processed_text.split("\n\n")),
                "char_diff": len(processed_text) - char_count,
                "char_diff_pct": (len(processed_text) - char_count)
                / max(1, char_count)
                * 100,
            }
        
        # Add Claude usage metrics - combine all modes, respecting cached modes that had no API cost
        # Check if any metrics have the cached flag
        cached_modes = [mode for mode, metrics in all_claude_metrics.items() if metrics.get("cached", False)]
        fresh_modes = [mode for mode in all_claude_metrics if mode not in cached_modes]
        
        # Sum up the costs for fresh modes (not cached)
        fresh_cost = sum(metrics.get("total_cost", 0) for mode, metrics in all_claude_metrics.items() if mode in fresh_modes)
        
        # Calculate tokens only for non-cached modes
        input_tokens = sum(metrics.get("input_tokens", 0) for mode, metrics in all_claude_metrics.items() if mode in fresh_modes)
        output_tokens = sum(metrics.get("output_tokens", 0) for mode, metrics in all_claude_metrics.items() if mode in fresh_modes)
        
        # Calculate input and output costs only for non-cached modes
        input_cost = sum(metrics.get("input_cost", 0) for mode, metrics in all_claude_metrics.items() if mode in fresh_modes)
        output_cost = sum(metrics.get("output_cost", 0) for mode, metrics in all_claude_metrics.items() if mode in fresh_modes)
        
        # Set up the metrics with cached info and ensure all expected keys exist
        process_metrics["claude_usage"] = {
            "total_cost": fresh_cost,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "input_cost": input_cost,
            "output_cost": output_cost,
            "cached_modes": cached_modes,
            "fresh_modes": fresh_modes,
            "by_mode": all_claude_metrics,
            "savings_from_cache": len(cached_modes) > 0
        }
        
        # Add processed text metrics for any cached texts that might not have been processed above
        for mode, text in processed_texts.items():
            if f"processed_text_{mode}" not in process_metrics:
                process_metrics[f"processed_text_{mode}"] = {
                    "char_count": len(text),
                    "word_count": len(text.split()),
                    "paragraph_count": len(text.split("\n\n")),
                    "char_diff": len(text) - char_count if hasattr(self, 'char_count') else 0,
                    "char_diff_pct": ((len(text) - char_count) / max(1, char_count) * 100) if hasattr(self, 'char_count') else 0,
                    "from_cache": True
                }

        # Generate a descriptive filename/directory name
        console.print(
            Panel(
                "[cyan]Generating a descriptive name for output files based on content...[/cyan]",
                title="Step 3: Filename Generation",
                border_style="green",
                padding=(1, 2),
                width=100
            )
        )

        # Choose the first available processed text for filename generation
        # When using cached text, processed_text might not be defined directly
        if processed_texts and len(processed_texts) > 0:
            # Get text from the first available processing mode
            first_mode = next(iter(processed_texts))
            text_for_filename = processed_texts[first_mode]
            
            base_name = self.generate_filename(text_for_filename)
            console.print(
                Panel(
                    f"[bold green]Generated base name:[/bold green] [cyan]{base_name}[/cyan]",
                    border_style="green",
                    padding=(1, 2),
                    width=100,
                )
            )
        else:
            # Fallback if no processed text is available (unlikely)
            console.print(
                Panel(
                    "[yellow]Warning: No processed text available for filename generation[/yellow]",
                    border_style="yellow",
                    padding=(1, 2),
                    width=100,
                )
            )
            base_name = f"tts_output_{int(time.time())}"
            console.print(
                Panel(
                    f"[bold yellow]Using fallback base name:[/bold yellow] [cyan]{base_name}[/cyan]",
                    border_style="yellow",
                    padding=(1, 2),
                    width=100,
                )
            )

        # Create output directory
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        language_code = (
            "en"  # Default to English, could be expanded to detect or select language
        )
        output_dir = Path(f"{base_name}_{timestamp}")
        output_dir.mkdir(exist_ok=True)
        console.print(
            Panel(
                f"[bold green]Created output directory:[/bold green] [cyan]{output_dir}[/cyan]",
                border_style="green",
                padding=(1, 2),
                width=100,
            )
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
                width=100
            )
        )

        original_text_path = output_dir / f"{base_name}_original.txt"
        
        # Write the original text
        with open(original_text_path, "w", encoding="utf-8") as f:
            f.write(input_text)
            
        # Write each processed version
        processed_text_paths = {}
        for mode, text in processed_texts.items():
            processed_text_path = output_dir / f"{base_name}_processed_{mode}.txt"
            processed_text_paths[mode] = processed_text_path
            
            # Track whether this came from cache
            cache_note = " (from cache)" if skip_text_preprocessing and mode in cached_processed_text else ""
            
            with open(processed_text_path, "w", encoding="utf-8") as f:
                f.write(text)
                
            console.print(
                Panel(
                    f"[green]Saved {mode} processed text to {processed_text_path.name}{cache_note}[/green]",
                    border_style="green",
                    padding=(1, 2),
                    width=100,
                )
            )
            
        # For backward compatibility with the code below, use the first processed text
        if processed_text_paths:
            processed_text_path = next(iter(processed_text_paths.values()))
        else:
            # Fallback if no processed texts (should not happen)
            processed_text_path = original_text_path
            console.print(
                Panel(
                    "[yellow]Warning: No processed text paths available, using original text path as fallback[/yellow]",
                    border_style="yellow",
                    padding=(1, 2),
                    width=100,
                )
            )

        # Generate audio for each configuration
        all_audio_metadata = []
        
        # Check if this is an Eleven Labs voice
        is_elevenlabs = voice.startswith("elevenlabs_")
        
        console.print(
            Panel(
                f"[cyan]Converting text to speech using {'Eleven Labs' if is_elevenlabs else 'OpenAI'}'s TTS API with {voice.split('_')[-1] if is_elevenlabs else voice} voice[/cyan]",
                title="Step 5: Text-to-Speech Conversion",
                border_style="green",
                padding=(1, 2),
                width=100
            )
        )
        
        # Process each configuration
        audio_paths = []
        for i, config in enumerate(audio_configurations, 1):
            processing_mode = config["processing_mode"]
            speech_rate = config["speech_rate"]
            output_format = config["output_format"]
            
            mode_name = "Dummy" if processing_mode == "dummy" else "Standard" if processing_mode == "standard" else "Literal"
            
            console.print(f"[bold cyan]Processing configuration {i}/{len(audio_configurations)}:[/bold cyan]")
            console.print(f"[cyan]• Mode: {mode_name}[/cyan]")
            console.print(f"[cyan]• Speed: {speech_rate}x[/cyan]")
            console.print(f"[cyan]• Format: {output_format.upper()}[/cyan]")
            
            # Get the correct processed text for this mode
            processed_text = processed_texts[processing_mode]
            
            # Construct output filename
            mode_indicator = f"_{processing_mode}"
            audio_output_path = (
                output_dir
                / f"{base_name}_{language_code}_{voice}_{speech_rate}x{mode_indicator}.{output_format}"
            )
            
            # Generate the audio
            audio_path, audio_metadata = self.text_to_speech(
                processed_text, voice, str(audio_output_path), speech_rate
            )
            
            # Add to our collection
            audio_paths.append(audio_path)
            audio_metadata["config"] = config.copy()
            all_audio_metadata.append(audio_metadata)
            
            console.print(
                Panel(
                    f"[green]Completed audio file: {audio_output_path.name}[/green]",
                    border_style="green",
                    padding=(1, 2),
                    width=100,
                )
            )

        # Save all audio metadata to process metrics
        process_metrics["audio_metadata"] = all_audio_metadata
        
        # Set the first audio as the "primary" one for reporting
        process_metrics["primary_audio"] = all_audio_metadata[0] if all_audio_metadata else None

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
            "dummy_mode": "dummy" in processing_modes,
            "metrics": process_metrics,
        }
        self._save_to_cache(content_hash, metadata)
        
        # Show a visually appealing completion message with audio player hint
        console.print("\n")
        
        # Format time more nicely
        total_time = process_metrics['total_time']
        if total_time < 60:
            time_str = f"{total_time:.1f} seconds"
        else:
            minutes = int(total_time // 60)
            seconds = int(total_time % 60)
            time_str = f"{minutes}m {seconds}s"
        
        # Get first audio file path for playback example
        example_audio = audio_paths[0] if audio_paths else ""
        example_cmd = f"open {example_audio}" if os.name != 'nt' else f"start {example_audio}"
        
        # Create a nice completion panel with playback instructions
        console.print(
            Panel(
                "[bold green]✅ Text-to-Speech Processing Complete![/bold green]\n\n"
                f"[cyan]• Generated:[/cyan] [bold yellow]{len(audio_paths)}[/bold yellow] [cyan]audio file{'s' if len(audio_paths) != 1 else ''}[/cyan]\n"
                f"[cyan]• Total Processing Time:[/cyan] [bold yellow]{time_str}[/bold yellow]\n"
                f"[cyan]• Output Directory:[/cyan] [bold yellow]{output_dir}[/bold yellow]\n\n"
                f"[bold white on bright_green] How to Play Your Audio Files [/bold white on bright_green]\n\n"
                f"[italic]To play your audio directly from the terminal:[/italic]\n"
                f"[bright_green]$ {example_cmd}[/bright_green]\n\n"
                f"[italic]Or open the output folder to browse all files:[/italic]\n"
                f"[bright_green]$ open {output_dir}[/bright_green]",
                title="🎉 Processing Complete",
                subtitle="Generating detailed report...",
                border_style="green",
                box=box.DOUBLE,
                padding=(1, 2),
                width=100,
            )
        )

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
                box=box.DOUBLE,
                padding=(1, 2),
                width=100,
            )
        )

        # Calculate total cost
        claude_cost = metrics["claude_usage"]["total_cost"]
        
        # Handle audio_metadata as either a list (for multiple outputs) or dict (single output)
        if isinstance(metrics["audio_metadata"], list):
            # If list, use the first item as primary for the report
            if metrics["audio_metadata"]:
                primary_audio = metrics["audio_metadata"][0]
                tts_cost = primary_audio.get("tts_cost", 0.0)
            else:
                # Fallback if no audio was generated
                primary_audio = {"tts_cost": 0.0}
                tts_cost = 0.0
            # Store as primary for the rest of the report
            metrics["primary_audio"] = primary_audio
        else:
            # Legacy mode - single output
            tts_cost = metrics["audio_metadata"].get("tts_cost", 0.0)
            metrics["primary_audio"] = metrics["audio_metadata"]
            
        total_cost = claude_cost + tts_cost

        # Create a layout for the report with fixed width
        layout = Layout(size=100)  # Constrain the entire layout to width 100
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
                width=100,
                padding=(1, 2)
            )
        )

        # Stats section with multiple tables
        stats_layout = Layout()
        stats_layout.split_row(Layout(name="text_stats"), Layout(name="audio_stats"))

        # Text statistics table with consistent width
        text_stats = Table(
            title="📝 Text Processing Statistics", 
            show_header=True, 
            box=box.ROUNDED,
            title_style="bold bright_blue",
            border_style="blue",
            header_style="bright_cyan",
            width=48  # Half of the total layout width (100/2 - small margin)
        )
        text_stats.add_column("Metric", style="bright_cyan", width=15)
        text_stats.add_column("Original", style="bright_yellow", justify="right", width=10)
        text_stats.add_column("Processed", style="bright_green", justify="right", width=10)
        text_stats.add_column("Change", style="bright_magenta", justify="right", width=13)

        # Add text metrics
        input_text = metrics["input_text"]
        
        # Handle multiple processing modes
        processed_text_key = None
        for mode in ["processed_text_standard", "processed_text_dummy", "processed_text_literal"]:
            if mode in metrics:
                processed_text_key = mode
                break
                
        # If we can't find a processed text entry, use a dummy one
        if processed_text_key is None:
            # Create a safe fallback
            if "primary_audio" in metrics and "char_count" in metrics["primary_audio"]:
                processed_char_count = metrics["primary_audio"]["char_count"]
            else:
                processed_char_count = input_text['char_count']
                
            # Create a dummy processed_text entry
            metrics["processed_text_standard"] = {
                "char_count": processed_char_count,
                "word_count": input_text['word_count'],
                "paragraph_count": input_text['paragraph_count'],
                "char_diff": 0,
                "char_diff_pct": 0.0
            }
            processed_text_key = "processed_text_standard"
                
        processed_text = metrics[processed_text_key]
        
        # Ensure we have char_diff and char_diff_pct
        if "char_diff" not in processed_text:
            processed_text["char_diff"] = processed_text["char_count"] - input_text["char_count"]
            
        if "char_diff_pct" not in processed_text:
            processed_text["char_diff_pct"] = (processed_text["char_diff"] / input_text["char_count"] * 100) if input_text["char_count"] > 0 else 0.0
            
        # Now we can safely add the row
        text_stats.add_row(
            "Characters",
            f"{input_text['char_count']:,}",
            f"{processed_text['char_count']:,}",
            f"{processed_text['char_diff']:+,} ({processed_text['char_diff_pct']:.1f}%)",
        )

        # Add word count stats with safety checks
        if "word_count" not in processed_text:
            processed_text["word_count"] = input_text["word_count"]
            
        text_stats.add_row(
            "Words",
            f"{input_text['word_count']:,}",
            f"{processed_text['word_count']:,}",
            f"{processed_text['word_count'] - input_text['word_count']:+,}",
        )

        # Add paragraph count stats with safety checks
        if "paragraph_count" not in processed_text:
            processed_text["paragraph_count"] = input_text["paragraph_count"]
            
        text_stats.add_row(
            "Paragraphs",
            f"{input_text['paragraph_count']:,}",
            f"{processed_text['paragraph_count']:,}",
            f"{processed_text['paragraph_count'] - input_text['paragraph_count']:+,}",
        )

        # Audio statistics table with consistent width
        audio_stats = Table(
            title="🔊 Audio Output Statistics", 
            show_header=True, 
            box=box.ROUNDED,
            title_style="bold bright_blue",
            border_style="blue",
            header_style="bright_cyan",
            width=48  # Half of the total layout width (100/2 - small margin)
        )
        audio_stats.add_column("Metric", style="bright_cyan", width=18)
        audio_stats.add_column("Value", style="bright_green", width=30)

        # Add audio metrics - use primary_audio that we set up earlier
        audio_metadata = metrics["primary_audio"]
        voice_settings = metrics["voice_settings"]
        dummy_mode = "dummy" in metrics.get("processing_modes", [])
        
        # Determine if this is an Eleven Labs voice
        voice = voice_settings['voice']
        is_elevenlabs = voice.startswith("elevenlabs_")
        
        # Provider and model info
        provider = "Eleven Labs" if is_elevenlabs else "OpenAI"
        model_info = "Multilingual v2" if is_elevenlabs else audio_metadata.get('model', '')

        audio_stats.add_row(
            "Duration",
            f"{audio_metadata['audio_duration_formatted']} ({audio_metadata['audio_duration']:.1f}s)",
        )
        audio_stats.add_row("File Size", f"{audio_metadata['file_size_mb']:.2f} MB")
        
        # Display provider and voice differently based on the service
        audio_stats.add_row("Provider", f"{provider}")
        
        if is_elevenlabs:
            # For Eleven Labs, show a cleaner voice name
            voice_id = voice.replace("elevenlabs_", "")
            # Find the actual name if possible
            voice_name = voice_id
            for elabs_voice in self.elevenlabs_voices:
                if elabs_voice["id"] == voice_id:
                    voice_name = elabs_voice["name"]
                    break
            audio_stats.add_row("Voice", f"{voice_name} (Custom)")
        else:
            audio_stats.add_row("Voice", f"{voice}")
            
        audio_stats.add_row("Model", f"{model_info}")
        
        # Safely access speech rate - in some cases with multiple configurations this might not be set properly
        speech_rate = None
        # Try different ways to get the speech rate
        if "speech_rates" in voice_settings and voice_settings["speech_rates"]:
            speech_rate = voice_settings["speech_rates"][0]  # Use the first rate if multiple
        elif "speech_rate" in voice_settings:
            speech_rate = voice_settings["speech_rate"]
        elif "speech_rate" in audio_metadata:
            speech_rate = audio_metadata["speech_rate"]
        else:
            # Default to 1.0 if we can't find it
            speech_rate = 1.0
            
        audio_stats.add_row("Speech Rate", f"{speech_rate}x")
        
        # Safely access output format
        output_format = None
        if "output_formats" in voice_settings and voice_settings["output_formats"]:
            output_format = voice_settings["output_formats"][0].upper()  # Use the first format if multiple
        elif "output_format" in voice_settings:
            output_format = voice_settings["output_format"].upper()
        else:
            # Derive from filename if possible
            ext = os.path.splitext(str(audio_output_path))[1].replace(".", "").upper()
            if ext:
                output_format = ext
            else:
                output_format = "MP3"  # Default
                
        audio_stats.add_row("Format", f"{output_format}")
        
        # Add bitrate information if available
        if "bitrate_info" in audio_metadata:
            audio_stats.add_row("Audio Quality", f"{audio_metadata['bitrate_info']}")
            
        audio_stats.add_row("Dummy Mode", f"{'Enabled' if dummy_mode else 'Disabled'}")
        audio_stats.add_row("Chars/Second", f"{audio_metadata['chars_per_second']:.1f}")
        # Calculate words per minute safely
        if "audio_duration" in audio_metadata and audio_metadata["audio_duration"] > 0:
            words_per_minute = (processed_text['word_count'] / audio_metadata['audio_duration']) * 60
        else:
            words_per_minute = 0.0
            
        audio_stats.add_row(
            "Words/Minute",
            f"{words_per_minute:.1f}",
        )

        # Update stats layout
        stats_layout["text_stats"].update(text_stats)
        stats_layout["audio_stats"].update(audio_stats)
        layout["stats"].update(stats_layout)

        # Files table with consistent width
        files_table = Table(
            title="📁 Generated Output Files", 
            show_header=True, 
            box=box.ROUNDED,
            title_style="bold bright_blue",
            border_style="blue",
            header_style="bright_cyan",
            width=100,
            padding=(0, 1)
        )
        files_table.add_column("File Type", style="bright_cyan", width=15)
        files_table.add_column("Description", style="bright_yellow", width=35)
        files_table.add_column("Path", style="bright_green", width=35)
        files_table.add_column("Size", style="bright_magenta", justify="right", width=10)

        # Add file information
        original_size = os.path.getsize(original_text_path)
        processed_size = os.path.getsize(processed_text_path)
        audio_size = audio_metadata["file_size_bytes"]
        
        # Add bitrate information if available
        output_format = os.path.splitext(audio_output_path)[1][1:].lower()
        if output_format == "mp3":
            audio_metadata["bitrate_info"] = "VBR 320kbps equivalent (using -q:a 0)"
        elif output_format == "aac":
            audio_metadata["bitrate_info"] = "256kbps AAC"
        elif output_format == "flac":
            audio_metadata["bitrate_info"] = "Lossless (level 12 compression)"
        elif output_format == "wav":
            audio_metadata["bitrate_info"] = "24-bit PCM (lossless)"
        else:
            audio_metadata["bitrate_info"] = "Standard quality"

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
        # Safely get voice and speech rate values
        voice_name = voice_settings.get('voice', voice)
        
        # Get speech rate using the previously defined variable or try to get it again
        if not speech_rate:
            if "speech_rates" in voice_settings and voice_settings["speech_rates"]:
                speech_rate = voice_settings["speech_rates"][0]
            elif "speech_rate" in voice_settings:
                speech_rate = voice_settings["speech_rate"]
            else:
                speech_rate = 1.0
        
        audio_description = f"{voice_name} voice at {speech_rate}x speed"
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
                width=100,
                padding=(1, 2)
            )
        )

        # Print the full report
        console.print(layout)

        # Create pricing summary table with consistent width
        pricing_summary = Table(
            title="💰 Cost Summary", 
            show_header=True, 
            box=box.ROUNDED,
            title_style="bold bright_blue",
            border_style="blue",
            header_style="bright_cyan",
            width=100,
            padding=(0, 1)
        )
        pricing_summary.add_column("Service", style="bright_cyan", width=30)
        pricing_summary.add_column("Usage", style="bright_yellow", width=45)
        pricing_summary.add_column("Cost", style="bright_green", justify="right", width=20)

        # Claude costs - account for cached processing
        claude_usage = metrics["claude_usage"]
        
        # Check if we used any cached text processing
        if claude_usage.get("savings_from_cache", False):
            # Get more details about cached content
            cached_modes = claude_usage.get("cached_modes", [])
            cached_modes_str = ", ".join([m.capitalize() for m in cached_modes])
            
            # Estimate savings based on typical costs
            # Assume ~5000 input tokens and ~5000 output tokens per mode as a rough estimate
            estimated_input_savings = len(cached_modes) * 5000
            estimated_output_savings = len(cached_modes) * 5000
            
            # Calculate estimated cost savings
            input_savings = (estimated_input_savings / 1_000_000) * CLAUDE_INPUT_PRICE_PER_MILLION
            output_savings = (estimated_output_savings / 1_000_000) * CLAUDE_OUTPUT_PRICE_PER_MILLION
            total_savings = input_savings + output_savings
            
            # Add a row showing cache savings
            pricing_summary.add_row(
                "💾 Text Processing Cache",
                f"Reused: {cached_modes_str}",
                f"Est. Savings: ${total_savings:.4f}",
            )
            
            # Add a note for each reused mode
            for mode in cached_modes:
                pricing_summary.add_row(
                    f"   ↳ {mode.capitalize()} Mode",
                    "Reused from cache",
                    "No API cost 👍",
                )
            
        # Only show these rows if we used Claude API (not cached)
        if claude_usage.get("input_tokens", 0) > 0:
            pricing_summary.add_row(
                "Claude 3.7 Sonnet (Input)",
                f"{claude_usage.get('input_tokens', 0):,} tokens",
                f"${claude_usage.get('input_cost', 0.0):.4f}",
            )
            pricing_summary.add_row(
                "Claude 3.7 Sonnet (Output)",
                f"{claude_usage.get('output_tokens', 0):,} tokens",
                f"${claude_usage.get('output_cost', 0.0):.4f}",
            )
        else:
            # Add a placeholder row to show that Claude was used minimally or not at all
            pricing_summary.add_row(
                "Claude 3.7 API Usage",
                "Minimal or cached",
                "$0.0000",
            )

        # TTS costs - use primary_audio
        audio_metadata = metrics["primary_audio"]
        # Get the actual model used (whether original or fallback)
        if audio_metadata.get("used_fallback", False):
            tts_model = audio_metadata.get("model", "tts-1-hd")
        else:
            tts_model = audio_metadata.get("model", "unknown")
        char_count = metrics["processed_text_standard"]["char_count"] if "processed_text_standard" in metrics else 0
        pricing_summary.add_row(
            f"{'Eleven Labs' if is_elevenlabs else 'OpenAI'} {tts_model}",
            f"{char_count:,} characters",
            f"${audio_metadata.get('tts_cost', 0.0):.4f}",
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

        # Add an enhanced playback tips panel with command examples
        # Determine OS-specific commands
        file_explorer_cmd = "open" if os.name != 'nt' else "explorer"
        play_cmd = "open" if os.name != 'nt' else "start"
        text_editor_cmd = "open -a TextEdit" if os.name != 'nt' else "notepad"
        
        # Save a report file in markdown format (moving this up to define report_path before using it)
        report_path = output_dir / f"{metrics['output_info']['base_name']}_report.md"
        report_path_str = str(report_path)
        
        console.print(
            Panel(
                "[bold white on bright_blue] 🎧 Audio Playback [/bold white on bright_blue]\n"
                f"[bright_yellow]$ {play_cmd} {audio_output_path}[/bright_yellow]  [italic]# Play the generated audio file[/italic]\n\n"
                
                "[bold white on bright_blue] 📂 File Management [/bold white on bright_blue]\n"
                f"[bright_yellow]$ {file_explorer_cmd} {output_dir}[/bright_yellow]  [italic]# Open the output folder[/italic]\n\n"
                
                "[bold white on bright_blue] 📝 Compare Text Versions [/bold white on bright_blue]\n"
                f"[bright_yellow]$ {text_editor_cmd} {original_text_path}[/bright_yellow]  [italic]# View original text[/italic]\n"
                f"[bright_yellow]$ {text_editor_cmd} {processed_text_path}[/bright_yellow]  [italic]# View AI-optimized text[/italic]\n\n"
                
                "[bold white on bright_blue] 📊 Review Report [/bold white on bright_blue]\n"
                f"[bright_yellow]$ {text_editor_cmd} {report_path_str}[/bright_yellow]  [italic]# View detailed report[/italic]\n\n"
                
                "[italic]Tip: The processed text files can be reused in future runs to save API costs.[/italic]",
                title="🚀 Next Steps",
                border_style="bright_blue",
                box=box.DOUBLE,
                padding=(1, 2),
                width=100,
            )
        )

        # Report file has already been defined above
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
            # Calculate words per minute safely
            if "audio_duration" in audio_metadata and audio_metadata["audio_duration"] > 0:
                words_per_minute = (processed_text['word_count'] / audio_metadata['audio_duration']) * 60
            else:
                words_per_minute = 0.0
                
            f.write(
                f"| Words/Minute | {words_per_minute:.1f} |\n\n"
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
            # Update the TTS provider name if fallback was used
            provider = "OpenAI" 
            if is_elevenlabs:
                if audio_metadata.get("used_fallback", False):
                    provider = "OpenAI (Fallback from Eleven Labs)"
                else:
                    provider = "Eleven Labs"
                    
            # Get a safe character count value
            char_count = audio_metadata.get('char_count', 0)
            if not char_count and "processed_text_standard" in metrics:
                char_count = metrics["processed_text_standard"].get("char_count", 0)
                
            f.write(
                f"| {provider} {tts_model} | {char_count:,} characters | ${audio_metadata.get('tts_cost', 0.0):.4f} |\n"
            )
            f.write(f"| **Total Cost** | | **${total_cost:.4f}** |\n")
            f.write(
                f"| Cost per Minute of Audio | {audio_minutes:.2f} minutes | ${cost_per_minute:.4f}/min |\n"
            )

        # Mention the report file
        console.print(
            f"[italic]A detailed report has been saved to: [cyan]{report_path}[/cyan][/italic]"
        )


def show_help_banner():
    """Show a stylized help banner with example commands."""
    console = Console()
    console.print(
        Panel(
            "[bold blue]Text-to-Speech Processor[/bold blue] [bright_black]v1.2[/bright_black]\n"
            "[cyan]A powerful CLI tool to convert text files to professional audio with AI preprocessing[/cyan]\n\n"
            "[bold green]Basic Usage:[/bold green]\n"
            "  [yellow]python tts_processor.py [filename.txt][/yellow]\n\n"
            "[bold green]Common Options:[/bold green]\n"
            "  [yellow]--voice [voice_name][/yellow]     Select voice (e.g., onyx-hd, alloy)\n"
            "  [yellow]--rate [speed][/yellow]           Set speech rate (0.8-1.5x)\n"
            "  [yellow]--format [format][/yellow]        Choose audio format (mp3, wav, flac, aac)\n"
            "  [yellow]--dummy[/yellow]                  Use simplified language processing\n"
            "  [yellow]--non-interactive[/yellow]        Run with default settings\n\n"
            "[bold green]Examples:[/bold green]\n"
            "  [italic]# Basic processing:[/italic]\n"
            "  [yellow]python tts_processor.py myfile.txt[/yellow]\n\n"
            "  [italic]# HD voice with custom settings:[/italic]\n"
            "  [yellow]python tts_processor.py --voice onyx-hd --rate 1.2 --format mp3 myfile.txt[/yellow]\n\n"
            "  [italic]# Simplified language for beginners:[/italic]\n"
            "  [yellow]python tts_processor.py --dummy myfile.txt[/yellow]",
            title="🎙️ TTS Processor Help",
            border_style="bright_blue",
            box=box.DOUBLE,
            padding=(1, 2),
            width=100,
        )
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
  python tts_processor.py myfile.txt                                      # Basic processing
  python tts_processor.py --voice onyx-hd --rate 1.2 --format mp3 myfile.txt  # HD voice with custom settings
  python tts_processor.py --non-interactive document.txt                  # Non-interactive mode
  python tts_processor.py --dummy myfile.txt                              # Simplified language
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
        "--format", choices=["mp3", "wav", "flac", "aac"], help="Audio output format"
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
        # If argument parsing fails, display the fancy help banner
        show_help_banner()
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
        # Create an elegant exit message
        console.print("\n")
        console.print(
            Panel(
                "[bold yellow]Processing cancelled by user at [cyan]" + 
                time.strftime("%H:%M:%S") + "[/cyan][/bold yellow]\n\n" +
                "[bright_green]Progress has been saved to the cache.[/bright_green]\n" +
                "[bright_green]You can resume later by running the script again.[/bright_green]",
                title="🛑 Process Interrupted",
                border_style="red",
                box=box.DOUBLE,
                padding=(1, 2),
                width=100
            )
        )
    except EOFError:
        console.print(
            "\n[bold bright_red]Error:[/bold bright_red] Cannot get input in non-interactive mode. Please set API keys in environment."
        )
        return 1
    except Exception as e:
        # More detailed error message with suggestions
        console.print("\n")
        console.print(
            Panel(
                f"[bold bright_red]Error:[/bold bright_red] {e}\n\n" +
                "[bold yellow]Possible solutions:[/bold yellow]\n" +
                "• Check your API keys are valid\n" +
                "• Ensure the input file exists and is readable\n" +
                "• Check network connectivity for API calls\n" +
                "• Try running with the --non-interactive flag\n\n" +
                f"[italic]Error details have been logged to the console output[/italic]",
                title="⚠️ Error Encountered",
                border_style="red",
                box=box.DOUBLE,
                padding=(1, 2),
                width=100
            )
        )
        logger.exception("Unexpected error")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
