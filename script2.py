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
from rpi_ws281x import PixelStrip, Color
import requests
import json

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
    def __init__(self, pul_pin=14, dir_pin=15, step_count=800, limit_switch_pin=17):
        self.pul_pin = pul_pin
        self.dir_pin = dir_pin
        self.step_count = step_count
        self.esp32_ip = '192.168.1.10'  # Replace with your ESP32's IP address
        self.limit_switch_pin = limit_switch_pin
        self.positions = {}  # To store alcohol, angle, and relay assignments
        self.current_position = 0  # Track current position in degrees

        self.setup_gpio()
        self.home_motor()  # Perform homing sequence on boot

    def setup_gpio(self):
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        GPIO.setup(self.pul_pin, GPIO.OUT)
        GPIO.setup(self.dir_pin, GPIO.OUT)
        GPIO.setup(self.limit_switch_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)  # Limit switch input with pull-up resistor

    def add_position(self, alcohol, angle, relay_number):
        """Add a new position for an alcohol, associated with an angle and relay number."""
        self.positions[alcohol] = {'angle': angle, 'relay': relay_number}

    def home_motor(self):
        """Homing sequence to find the limit switch and set the current position to 0."""
        print("Starting homing sequence...")
        GPIO.output(self.dir_pin, GPIO.HIGH)  # Set motor direction to home
        while GPIO.input(self.limit_switch_pin) == GPIO.HIGH:  # Wait until limit switch is pressed
            GPIO.output(self.pul_pin, GPIO.HIGH)
            time.sleep(0.003)
            GPIO.output(self.pul_pin, GPIO.LOW)
            time.sleep(0.003)
        print("Home position found.")
        self.current_position = 0  # Reset current position to 0

    def rotate_motor(self, alcohol):
        """Rotate the motor to the position associated with the specified alcohol."""
        if alcohol not in self.positions:
            print(f"Alcohol '{alcohol}' not found in positions.")
            return

        target_angle = self.positions[alcohol]['angle']
        relay_number = self.positions[alcohol]['relay']
        steps = self.calculate_steps_to_target(target_angle)
        
        GPIO.output(self.dir_pin, GPIO.HIGH)

        # Rotate motor to target position
        for _ in range(steps):
            GPIO.output(self.pul_pin, GPIO.HIGH)
            time.sleep(0.003)
            GPIO.output(self.pul_pin, GPIO.LOW)
            time.sleep(0.003)

        # Update current position
        self.current_position = target_angle

        # Trigger the relay for this alcohol
        self.trigger_relay(relay_number, "on")
        time.sleep(5)  # Simulate valve open for 3 seconds (adjust for actual pour time)
        self.trigger_relay(relay_number, "off")

        time.sleep(1)

        # After pouring, return to home position
        self.home_motor()

    def calculate_steps_to_target(self, target_angle):
        """Calculate the number of steps needed to reach the target angle."""
        angle_difference = abs(target_angle - self.current_position)
        steps = int(self.step_count * angle_difference / 360)
        return steps

    def trigger_relay(self, relay_number, state):
        """Send an HTTP request to the ESP32 to control the relay."""

        url = f"http://{self.esp32_ip}/relay?number={relay_number}&state={state}"
        try:
            response = requests.get(url)
            if response.status_code == 200:
                print(f"Relay {relay_number} turned {state}")
            else:
                print(f"Failed to control relay {relay_number}: {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"Error controlling relay {relay_number}: {e}")


class WeightSensor:
    def __init__(self, data_pin, clock_pin, threshold=10000, stability_checks=1):
        self.data_pin = data_pin
        self.clock_pin = clock_pin
        self.threshold = threshold
        self.previous_weight = 0
        self.stability_checks = stability_checks

    def setup_sensor(self):
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.data_pin, GPIO.IN)
        GPIO.setup(self.clock_pin, GPIO.OUT)

    def read_weight(self):
        # Simulating actual reading logic with a placeholder for HX711 sensor data.
        count = 0

        # Wait for data ready (data pin goes low)
        while GPIO.input(self.data_pin):
            pass

        # Read 24-bit data from the HX711
        for _ in range(24):
            GPIO.output(self.clock_pin, GPIO.HIGH)
            count = count << 1
            GPIO.output(self.clock_pin, GPIO.LOW)
            if GPIO.input(self.data_pin):
                count += 1

        # Set the gain to 128 and pulse once more to complete the reading
        GPIO.output(self.clock_pin, GPIO.HIGH)
        GPIO.output(self.clock_pin, GPIO.LOW)

        # Convert the two's complement 24-bit result to a signed integer
        if count & 0x800000:
            count -= 0x1000000

        return count


    def detect_stability(self):
        self.setup_sensor()
        self.previous_weight = self.read_weight()
        stable_readings = 0

        while True:
            current_weight = self.read_weight()
            weight_difference = abs(current_weight - self.previous_weight)

            # Detect if the change exceeds the threshold
            print(weight_difference)
            print(self.threshold)
            if weight_difference <= self.threshold:
                stable_readings += 1
                print(stable_readings, 'stable readings')
                if stable_readings >= self.stability_checks:
                    print(f"Weight stable for {self.stability_checks} consecutive checks.")
                    return False  # Weight has stabilized
            else:
                print(f"Significant weight change detected! Difference: {weight_difference}")
                stable_readings = 0  # Reset stability counter if a significant change occurs

            self.previous_weight = current_weight
            time.sleep(0.5)

    def detect_significant_change(self):
        self.setup_sensor()
        self.previous_weight = self.read_weight()
        while True:
            current_weight = self.read_weight()
            if abs(current_weight - self.previous_weight) > self.threshold:
                print("Significant weight change detected.")
                return True
            self.previous_weight = current_weight
            time.sleep(0.5)
        

class LEDHandler:
    def __init__(self, num_leds, led_pin, led_freq_hz=800000, led_dma=5, led_brightness=55, led_channel=1):
        self.strip = PixelStrip(num_leds, led_pin, led_freq_hz, led_dma, False, led_brightness, led_channel)
        self.strip.begin()

    def set_color(self, color):
        for i in range(self.strip.numPixels()):
            self.strip.setPixelColor(i, color)
        self.strip.show()

    def oscillate(self, color1):
        self.set_color(color1)
        time.sleep(0.5)
        self.set_color(Color(0, 0, 0))
        time.sleep(0.5)

    def ease_in(self, target_color, steps=50, duration=1.0):
        start_color = Color(0, 0, 0)

        start_r = (start_color >> 16) & 0xFF
        start_g = (start_color >> 8) & 0xFF
        start_b = start_color & 0xFF

        target_r = (target_color >> 16) & 0xFF
        target_g = (target_color >> 8) & 0xFF
        target_b = target_color & 0xFF

        for step in range(steps + 1):
            factor = math.pow(step / steps, 2)
            intermediate_color = Color(
                int(start_r + factor * (target_r - start_r)),
                int(start_g + factor * (target_g - start_g)),
                int(start_b + factor * (target_b - start_b))
            )
            self.set_color(intermediate_color)
            time.sleep(duration / steps)

    def light_up(self, color):
        self.ease_in(color)

    def light_down(self):
        self.ease_in(Color(0, 0, 0))

    def change_color_step(self, color):
        self.ease_in(color)
        time.sleep(0.5)

class BartenderBot:
    def __init__(self, num_leds, led_pin, weight_data_pin, weight_clock_pin):
        self.audio_manager = AudioManager()
        self.motor_controller = MotorController()
        self.led_controller = LEDHandler(num_leds, led_pin)
        self.weight_sensor = WeightSensor(weight_data_pin, weight_clock_pin)
        self.setup_positions()  # Configure positions for the alcohols

    def setup_positions(self):
        """Setup positions for different alcohols, assigning an angle and relay number."""
        self.motor_controller.add_position('vodka', 30, 1)
        self.motor_controller.add_position('gin', 180, 0)
        self.motor_controller.add_position('tonic', 300, 4)
        self.motor_controller.add_position('juice', 100, 3)
        # self.motor_controller.add_position('orange juice', 240, 4)

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
        Check if the input includes a cocktail request. If not, return "error". If it is, check if it can be made with the ingredients: vodka, gin, tonic, juice. If possible, return a JSON with the ingredient percentages, you can be lenient on the feasibility and arrange a bit to make it possible. If not, return "impossible".

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

    def process_cocktail(self, recipe):
        """Process the cocktail recipe by rotating to each ingredient and pouring."""
        for ingredient, percentage in recipe.items():
            print(f"Processing {ingredient} ({percentage})")

            # Wait until no more significant weight changes (i.e., weight has stabilized)
            print("Waiting for weight sensor stability...")
            while self.weight_sensor.detect_stability():
                print("Waiting for stability...")
                time.sleep(0.5)

            # Once stable, rotate to position and pour
            self.motor_controller.rotate_motor(ingredient)
            time.sleep(0.5)  # Delay between pours (adjust as needed)

    async def process_order(self):
        self.audio_manager.check_audio_devices()

        # Wait until significant weight change is detected
        print("Waiting for significant weight change...")
        self.weight_sensor.detect_significant_change()
        print("Significant change detected, proceeding...")

        # Light up LEDs when glass is detected
        self.led_controller.light_up(Color(200, 100, 120))  # Green color for detection

        # Play an initial audio file
        self.audio_manager.play_random_audio('in')

        output_file = "output.wav"

        # Record audio for 5 seconds
        self.audio_manager.record_audio(5, output_file)

        # Play the mid-processing audio in a separate thread
        mid_audio_thread = threading.Thread(target=self.audio_manager.play_random_audio, args=('mid',))
        mid_audio_thread.start()

        # Transcribe the audio using Whisper
        transcription = await self.transcribe_with_whisper(output_file)

        # Wait for the mid-audio thread to finish before proceeding
        mid_audio_thread.join()

        if transcription:
            # Check if the transcription is a valid cocktail request
            result = await self.check_cocktail_request(transcription)
            print("Final result:")
            print(result)

            if "error" in result:
                print("Error: Invalid request.")
                self.led_controller.change_color_step(Color(160, 70, 70))
                return
            elif "impossible" in result:
                print("Error: Cocktail request is impossible.")
                self.led_controller.change_color_step(Color(160, 70, 70))
                return
            else:
                # Extract the JSON portion from the GPT-4 response
                start_index = result.find("{")
                end_index = result.rfind("}") + 1
                json_str = result[start_index:end_index]

                try:
                    recipe = json.loads(json_str)  # Parse the JSON safely
                    print("Processing cocktail recipe:", recipe)
                    self.process_cocktail(recipe)
                except json.JSONDecodeError:
                    print("Error: Failed to parse JSON.")
                    return

                # Change LED color to indicate completion
                self.led_controller.change_color_step(Color(70, 160, 70))  # Blue for completion
                # Play a final audio file
                self.audio_manager.play_random_audio('out')

        else:
            print("Error: No transcription received.")
            self.led_controller.change_color_step(Color(160, 70, 70))

        print("Process complete.")
        time.sleep(1)
        self.led_controller.light_down()



async def main():
    bot = BartenderBot(num_leds=20, led_pin=13, weight_data_pin=6, weight_clock_pin=5)
    await bot.process_order()

# Run the main function
asyncio.run(main())
