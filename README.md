# TTS Processor

A powerful CLI tool to convert text files to professional-sounding speech with AI preprocessing.

![TTS Processor Banner](https://github.com/walksalot/tts-processor/raw/main/.github/images/tts-processor-banner.png)

## Features

- **Smart AI Preprocessing**: Claude 3.7 AI optimizes text for natural speech
- **Premium Voice Options**: Use OpenAI or your custom Eleven Labs voices
- **Multiple Output Formats**: MP3, WAV, FLAC, or AAC with customizable settings
- **Intelligent Caching**: Reuse processed text to save time and API costs
- **Dummy Mode**: Simplify complex text to make it accessible to beginners
- **High-Quality Audio**: HD voice models with customizable speech rates

## Requirements

- Python 3.10+
- OpenAI API key
- Anthropic API key (Claude)
- Eleven Labs API key (optional, for custom voices)

## Installation

```bash
# Clone the repository
git clone https://github.com/walksalot/tts-processor.git
cd tts-processor

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
# Basic usage
python tts_processor.py input_file.txt

# Using HD voice with custom settings
python tts_processor.py --voice onyx-hd --rate 1.2 --format mp3 input_file.txt

# Using simplified language (Dummy Mode)
python tts_processor.py --dummy input_file.txt

# Non-interactive mode
python tts_processor.py --non-interactive input_file.txt
```

## Configuration

The script automatically looks for API keys in these locations:
- Environment variables: `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `ELEVENLABS_API_KEY`
- Configuration files: `~/.speech_config/api_keys.json`
- Traditional locations: `~/.anthropic`, `~/.openai`, `~/.elevenlabs`

## Output

For each processed file, the script creates:
- A timestamped directory named after the content
- Original and processed text files
- Audio file(s) in your selected format(s)
- A detailed report with processing statistics

## License

MIT