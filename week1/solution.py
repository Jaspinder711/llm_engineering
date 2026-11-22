"""
Day 2 challenge: the Day 1 website summarizer, using a local open-source
model served by Ollama instead of a paid frontier API.

Run with:
    uv run solution.py https://edwarddonner.com
    uv run solution.py                      # prompts for a URL
    uv run solution.py URL --model llama3.2:1b

Requires Ollama running locally (`ollama serve`) and the model pulled
(`ollama pull llama3.2`).
"""

import argparse
import sys

import requests
from openai import OpenAI

from scraper import fetch_website_contents

OLLAMA_URL = "http://localhost:11434"
OLLAMA_BASE_URL = f"{OLLAMA_URL}/v1"  # Ollama's OpenAI-compatible endpoint
DEFAULT_MODEL = "llama3.2"  # use "llama3.2:1b" on smaller machines

system_prompt = """
You are a snarky assistant that analyzes the contents of a website,
and provides a short, snarky, humorous summary, ignoring text that might be navigation related.
Respond in markdown. Do not wrap the markdown in a code block - respond just with the markdown.
"""

user_prompt_prefix = """
Here are the contents of a website.
Provide a short summary of this website.
If it includes news or announcements, then summarize these too.

"""

# The OpenAI client is only a thin wrapper over the HTTP endpoint; no OpenAI
# model is involved. The api_key is required by the client but ignored by Ollama.
ollama = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")


def messages_for(website):
    """Build the chat messages: a system prompt plus the website as the user prompt."""
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt_prefix + website},
    ]


def summarize(url, model=DEFAULT_MODEL):
    """Fetch the page at `url` and return a summary from the local Ollama model."""
    website = fetch_website_contents(url)
    response = ollama.chat.completions.create(
        model=model,
        messages=messages_for(website),
    )
    return response.choices[0].message.content


def ollama_is_running():
    try:
        return requests.get(OLLAMA_URL, timeout=3).ok
    except requests.RequestException:
        return False


def main():
    parser = argparse.ArgumentParser(description="Summarize a web page with a local Ollama model.")
    parser.add_argument("url", nargs="?", help="URL to summarize (prompted for if omitted)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Ollama model (default: {DEFAULT_MODEL})")
    args = parser.parse_args()

    url = args.url or input("Enter a URL to summarize: ").strip()
    if not url:
        sys.exit("No URL given.")
    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    if not ollama_is_running():
        sys.exit(f"Ollama isn't reachable at {OLLAMA_URL}. Start it with `ollama serve`.")

    try:
        print(summarize(url, args.model))
    except requests.RequestException as e:
        sys.exit(f"Could not fetch {url}: {e}")
    except Exception as e:  # e.g. model not pulled
        sys.exit(f"Summarization failed: {e}\nIs the model pulled? Try `ollama pull {args.model}`.")


if __name__ == "__main__":
    main()
