import os
import random
import time
import threading
import math
import numpy as np
import wave
import sounddevice as sd
import asyncio
import RPi.GPIO as GPIO
from openai import OpenAI
from dotenv import load_dotenv
import board
import adafruit_ws2801

# Load environment variables from a .env file
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))

class AudioManager:
    def __init__(self):
        self.fs = 44100  # Sampling rate
        self.channels = 1  # Number of channels

    def check_audio_devices(self):
        """Check if audio input and output devices are available."""
        input_devices = sd.query_devices(kind='input')
        output_devices = sd.query_devices(kind='output')
        if not input_devices:
            raise RuntimeError("No audio input device found.")
        if not output_devices:
            raise RuntimeError("No audio output device found.")
        print("Audio input and output devices are available.")

    def play_random_audio(self, file_pattern):
        files = [f for f in os.listdir('lines') if f.startswith(file_pattern)]
        if files:
            selected_file = random.choice(files)
            audio_path = os.path.join('lines', selected_file)
            print(f"Playing audio: {audio_path}")
            os.system(f"aplay {audio_path}")

    def record_audio(self, duration, filename):
        print(f"Recording audio for {duration} seconds...")
        recording = sd.rec(int(duration * self.fs), samplerate=self.fs, channels=self.channels, dtype='int16')
        sd.wait()

        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(np.dtype('int16').itemsize)
            wf.setframerate(self.fs)
            wf.writeframes(recording.tobytes())

        print(f"Audio recording saved to {filename}")

class MotorController:
    def __init__(self, pul_pin=14, dir_pin=15, step_count=200):
        self.pul_pin = pul_pin
        self.dir_pin = dir_pin
        self.step_count = step_count

    def setup_gpio(self):
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        GPIO.setup(self.pul_pin, GPIO.OUT)
        GPIO.setup(self.dir_pin, GPIO.OUT)

    def rotate_motor(self, degrees=180):
        self.setup_gpio()
        steps = int(self.step_count * degrees / 360)
        print(f"Rotating motor {degrees} degrees...")

        GPIO.output(self.dir_pin, GPIO.HIGH)
        for _ in range(steps):
            GPIO.output(self.pul_pin, GPIO.HIGH)
            time.sleep(0.01)
            GPIO.output(self.pul_pin, GPIO.LOW)
            time.sleep(0.01)

        GPIO.cleanup()

class WeightSensor:
    def __init__(self, data_pin, clock_pin, threshold=50):
        self.data_pin = data_pin
        self.clock_pin = clock_pin
        self.threshold = threshold
        self.previous_weight = 0

    def setup_sensor(self):
        # Setup GPIO pins for weight sensor
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.data_pin, GPIO.IN)
        GPIO.setup(self.clock_pin, GPIO.OUT)

    def read_weight(self):
        # Implement reading from the weight sensor (this depends on your sensor)
        # Placeholder value:
        weight = random.randint(0, 1000)
        return weight

    def detect_significant_change(self):
        self.setup_sensor()
        while True:
            current_weight = self.read_weight()
            if abs(current_weight - self.previous_weight) > self.threshold:
                print("Significant weight change detected.")
                return True
            self.previous_weight = current_weight
            time.sleep(0.1)

class LEDController:
    def __init__(self, num_leds, data_pin, clock_pin):
        self.num_leds = num_leds
        self.pixels = adafruit_ws2801.WS2801(board.SCK, board.MOSI, num_leds, brightness=1.0, auto_write=False)

    def set_color(self, color):
        for i in range(self.num_leds):
            self.pixels[i] = color
        self.pixels.show()

    def ease_in(self, target_color, steps=50, duration=1.0):
        """
        Gradually changes the LED color to the target color with an ease-in effect.
        :param target_color: Tuple of (R, G, B) values for the target color.
        :param steps: Number of steps in the transition.
        :param duration: Total duration of the ease-in effect in seconds.
        """
        start_color = (0, 0, 0)  # Starting with LEDs off
        for step in range(steps + 1):
            factor = math.pow(step / steps, 2)  # Quadratic ease-in
            intermediate_color = tuple(
                int(start_color[i] + factor * (target_color[i] - start_color[i]))
                for i in range(3)
            )
            self.set_color(intermediate_color)
            time.sleep(duration / steps)

    def light_up(self, color):
        self.ease_in(color)

    def change_color_step(self, color):
        self.ease_in(color)
        time.sleep(0.5)

class BartenderBot:
    def __init__(self, num_leds, led_data_pin, led_clock_pin, weight_data_pin, weight_clock_pin):
        self.audio_manager = AudioManager()
        self.motor_controller = MotorController()
        self.led_controller = LEDController(num_leds, led_data_pin, led_clock_pin)
        self.weight_sensor = WeightSensor(weight_data_pin, weight_clock_pin)

    async def transcribe_with_whisper(self, audio_file):
        print("Transcribing audio with Whisper...")
        with open(audio_file, "rb") as audio:
            transcription = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio
            )
        print("Transcription complete.")
        print(f"Transcription: {transcription.text}")
        return transcription.text

    async def check_cocktail_request(self, transcription):
        print("Checking if the transcription is a valid cocktail request...")
        prompt = f"""
        Check if the input includes a cocktail request. If not, return "error". If it is, check if it can be made with the ingredients: vodka, gin, tonic, triple sec, orange juice. If possible, return a JSON with the ingredient percentages. If not, return "impossible".

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

    async def process_order(self):
        self.audio_manager.check_audio_devices()

        # Wait until significant weight change is detected
        print("Waiting for glass to be placed...")
        self.weight_sensor.detect_significant_change()

        # Light up LEDs when glass is detected
        self.led_controller.light_up((0, 255, 0))  # Green color for detection

        # Play an initial audio file
        self.audio_manager.play_random_audio('in')

        output_file = "output.wav"

        # Record audio for 5 seconds
        self.audio_manager.record_audio(5, output_file)

        # Start the mid-processing audio in a separate thread
        mid_audio_thread = threading.Thread(target=self.audio_manager.play_random_audio, args=('mid',))
        mid_audio_thread.start()

        # Transcribe the audio using Whisper
        transcription = await self.transcribe_with_whisper(output_file)

        if transcription:
            # Check if the transcription is a valid cocktail request
            result = await self.check_cocktail_request(transcription)
            print("Final result:")
            print(result)

            # Change LED color for processing
            self.led_controller.change_color_step((255, 255, 0))  # Yellow for processing

            # Rotate motor after processing
            self.motor_controller.rotate_motor()

        # Ensure the mid-audio thread has finished
        mid_audio_thread.join()

        # Play a final audio file
        self.audio_manager.play_random_audio('out')

        # Change LED color to indicate completion
        self.led_controller.change_color_step((0, 0, 255))  # Blue for completion

        print("Process complete.")

async def main():
    bot = BartenderBot(num_leds=30, led_data_pin=18, led_clock_pin=23, weight_data_pin=5, weight_clock_pin=6)
    await bot.process_order()

# Run the main function
asyncio.run(main())