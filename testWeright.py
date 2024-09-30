import RPi.GPIO as GPIO
import time

class HX711:
    def __init__(self, data_pin, clock_pin, threshold=50):
        self.data_pin = data_pin
        self.clock_pin = clock_pin
        self.threshold = threshold
        self.previous_weight = 0

    def setup_sensor(self):
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.data_pin, GPIO.IN)
        GPIO.setup(self.clock_pin, GPIO.OUT)
        GPIO.output(self.clock_pin, GPIO.LOW)

    def read_weight(self):
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
        if count & 0x800000:  # If the sign bit is set
            count -= 0x1000000  # Apply two's complement

        return count

    def test_sensor(self, duration=10):
        self.setup_sensor()
        start_time = time.time()
        while time.time() - start_time < duration:
            weight = self.read_weight()
            print(f"Current weight (raw value): {-weight}")
            time.sleep(0.5)  # Read every 0.5 seconds
        print("Test complete.")

# Test the HX711 weight sensor for 10 seconds
hx711 = HX711(data_pin=6, clock_pin=5, threshold=50)
hx711.test_sensor(duration=10)

