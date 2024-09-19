import RPi.GPIO as GPIO
import time

pul_pin = 14
dir_pin = 15

GPIO.setmode(GPIO.BCM)
GPIO.setup(pul_pin, GPIO.OUT)
GPIO.setup(dir_pin, GPIO.OUT)

GPIO.output(dir_pin, GPIO.HIGH)  # Set direction

for _ in range(200):  # Rotate one full step (or more, depending on microstepping)
    GPIO.output(pul_pin, GPIO.HIGH)
    time.sleep(0.001)  # Adjust the delay for speed
    GPIO.output(pul_pin, GPIO.LOW)
    time.sleep(0.001)

GPIO.cleanup()
