#!/usr/bin/env python3
"""
Test script for OpenAI-compatible /v1/audio/transcriptions endpoint.

Usage:
    python test_openai_api.py [audio_file]
    
Example:
    python test_openai_api.py test_audio.wav
"""

import requests
import sys
import json
from pathlib import Path

# Configuration
BASE_URL = "http://localhost:8000"
ENDPOINT = f"{BASE_URL}/v1/audio/transcriptions"

def test_basic_transcription(audio_file: str):
    """Test basic JSON transcription."""
    print("=" * 80)
    print("TEST 1: Basic JSON Transcription")
    print("=" * 80)
    
    with open(audio_file, "rb") as f:
        response = requests.post(
            ENDPOINT,
            files={"file": f},
            data={
                "response_format": "json"
            }
        )
    
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()
    return response.status_code == 200


def test_text_format(audio_file: str):
    """Test plain text format."""
    print("=" * 80)
    print("TEST 2: Plain Text Format")
    print("=" * 80)
    
    with open(audio_file, "rb") as f:
        response = requests.post(
            ENDPOINT,
            files={"file": f},
            data={
                "response_format": "text"
            }
        )
    
    print(f"Status: {response.status_code}")
    print("Raw Response:")
    print(response.text)
    print()
    return response.status_code == 200


def test_diarized_json(audio_file: str):
    """Test diarized JSON format."""
    print("=" * 80)
    print("TEST 3: Diarized JSON Format")
    print("=" * 80)
    
    with open(audio_file, "rb") as f:
        response = requests.post(
            ENDPOINT,
            files={"file": f},
            data={
                "response_format": "diarized_json",
                "diarize": "true"
            }
        )
    
    print(f"Status: {response.status_code}")
    result = response.json()
    print("Raw Response:")
    print(json.dumps(result, indent=2))
    print()
    return response.status_code == 200


def test_verbose_json_with_words(audio_file: str):
    """Test verbose JSON with word-level timestamps."""
    print("=" * 80)
    print("TEST 4: Verbose JSON with Word Timestamps")
    print("=" * 80)
    
    with open(audio_file, "rb") as f:
        response = requests.post(
            ENDPOINT,
            files={"file": f},
            data={
                "response_format": "verbose_json",
                "timestamp_granularities": "word"
            }
        )
    
    print(f"Status: {response.status_code}")
    result = response.json()
    print("Raw Response:")
    print(json.dumps(result, indent=2))
    print()
    return response.status_code == 200


def test_with_language(audio_file: str):
    """Test with explicit language parameter."""
    print("=" * 80)
    print("TEST 5: Explicit Language Parameter")
    print("=" * 80)
    
    with open(audio_file, "rb") as f:
        response = requests.post(
            ENDPOINT,
            files={"file": f},
            data={
                "response_format": "json",
                "language": "en"
            }
        )
    
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()
    return response.status_code == 200


def test_no_diarization(audio_file: str):
    """Test with diarization disabled."""
    print("=" * 80)
    print("TEST 6: No Diarization (Merged Speakers)")
    print("=" * 80)
    
    with open(audio_file, "rb") as f:
        response = requests.post(
            ENDPOINT,
            files={"file": f},
            data={
                "response_format": "json",
                "diarize": "false"
            }
        )
    
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()
    return response.status_code == 200


def test_invalid_format(audio_file: str):
    """Test with invalid response format."""
    print("=" * 80)
    print("TEST 7: Invalid Response Format (Should Fail)")
    print("=" * 80)
    
    with open(audio_file, "rb") as f:
        response = requests.post(
            ENDPOINT,
            files={"file": f},
            data={
                "response_format": "invalid_format"
            }
        )
    
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()
    return response.status_code == 400


def test_no_file():
    """Test with no file uploaded."""
    print("=" * 80)
    print("TEST 8: No File Uploaded (Should Fail)")
    print("=" * 80)
    
    response = requests.post(
        ENDPOINT,
        data={
            "response_format": "json"
        }
    )
    
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()
    return response.status_code == 400


def main():
    if len(sys.argv) < 2:
        print("Usage: python test_openai_api.py <audio_file>")
        print("Example: python test_openai_api.py test_audio.wav")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    
    if not Path(audio_file).exists():
        print(f"Error: File '{audio_file}' not found")
        sys.exit(1)
    
    print(f"\nTesting OpenAI-compatible API with file: {audio_file}\n")
    
    tests = [
        ("Basic JSON", lambda: test_basic_transcription(audio_file)),
        ("Text Format", lambda: test_text_format(audio_file)),
        ("Diarized JSON", lambda: test_diarized_json(audio_file)),
        ("Verbose JSON + Words", lambda: test_verbose_json_with_words(audio_file)),
        ("Language Parameter", lambda: test_with_language(audio_file)),
        ("No Diarization", lambda: test_no_diarization(audio_file)),
        ("Invalid Format", lambda: test_invalid_format(audio_file)),
        ("No File", lambda: test_no_file()),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            passed = test_func()
            results.append((test_name, passed))
        except Exception as e:
            print(f"❌ ERROR: {e}\n")
            results.append((test_name, False))
    
    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)
    
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed_count}/{total_count} tests passed")
    
    sys.exit(0 if passed_count == total_count else 1)


if __name__ == "__main__":
    main()
