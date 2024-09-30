import RPi.GPIO as GPIO
import time

# Pin configuration
pul_pin = 14  # Pulse pin for motor
dir_pin = 15  # Direction pin for motor
limit_switch_pin = 17  # Limit switch pin (GPIO 17)

# Setup GPIO
GPIO.setmode(GPIO.BCM)
GPIO.setup(pul_pin, GPIO.OUT)
GPIO.setup(dir_pin, GPIO.OUT)
GPIO.setup(limit_switch_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)  # Setup limit switch with pull-up resistor

# Set motor direction
GPIO.output(dir_pin, GPIO.HIGH)  # Set direction (adjust as needed)
print('Starting motor rotation until limit switch is triggered...')

try:
    while GPIO.input(limit_switch_pin) == GPIO.HIGH:  # Keep rotating until limit switch is triggered (LOW)
        GPIO.output(pul_pin, GPIO.HIGH)
        time.sleep(0.01)  # Adjust delay for motor speed
        GPIO.output(pul_pin, GPIO.LOW)
        time.sleep(0.001)

    print("Limit switch triggered. Home position found.")

finally:
    GPIO.cleanup()  # Clean up GPIO settings after use
