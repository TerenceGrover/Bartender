from openai import OpenAI
import sounddevice as sd
from dotenv import load_dotenv
import numpy as np
import wave
import asyncio
import os

# Load environment variables from a .env file
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.environ.get('OPENAI'))

def record_audio(duration, filename):
    print("Recording audio for 5 seconds...")
    # Sampling rate and number of channels
    fs = 44100
    channels = 1

    # Record audio
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=channels, dtype='int16')
    sd.wait()  # Wait until recording is finished

    # Save as WAV file
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(np.dtype('int16').itemsize)
        wf.setframerate(fs)
        wf.writeframes(recording.tobytes())

    print(f"Audio recording saved to {filename}")

async def transcribe_with_whisper(audio_file):
    print("Transcribing audio with Whisper...")
    with open(audio_file, "rb") as audio:
        transcription = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio
        )
    print("Transcription complete.")
    print(f"Transcription: {transcription.text}")
    return transcription.text

async def check_cocktail_request(transcription):
    print("Checking if the transcription is a valid cocktail request...")
    prompt = f"""
    Check if the input includes a cocktail request. If not, return "error". If it is, check if it can be made with the ingredients: vodka, gin, tonic, triple sec, orange juice. If possible, return a JSON with the ingredient percentages. If not, return "impossible". You may adapt the recipes slightly to make more cocktails possible.

Input examples:
1. "Can you make a screwdriver?"
   Output: {{"vodka": "33%", "orange juice": "67%"}}
2. "Can you make a margarita?"
   Output: "impossible"
3. "I want a sandwich."
   Output: "error"

Transcription: "{transcription}"
    """

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )

    response_content = response.choices[0].message.content

    print("Cocktail check complete.")
    print(f"GPT-4 Response: {response_content}")
    return response_content

async def main():
    output_file = "output.wav"

    # Record audio for 5 seconds
    record_audio(5, output_file)

    # Transcribe the audio using Whisper
    transcription = await transcribe_with_whisper(output_file)

    if transcription:
        # Check if the transcription is a valid cocktail request
        result = await check_cocktail_request(transcription)
        print("Final result:")
        print(result)

# Run the main function
asyncio.run(main())
