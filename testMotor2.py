import RPi.GPIO as GPIO
import time

class MotorController:
    def __init__(self, pul_pin=14, dir_pin=15, step_count=800, limit_switch_pin=17, angle=180):
        self.pul_pin = pul_pin
        self.dir_pin = dir_pin
        self.angle = angle
        self.step_count = step_count
        self.limit_switch_pin = limit_switch_pin
        self.current_position = 0  # Track current position in degrees

        self.setup_gpio()
        self.home_motor()  # Perform homing sequence on boot

    def setup_gpio(self):
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        GPIO.setup(self.pul_pin, GPIO.OUT)
        GPIO.setup(self.dir_pin, GPIO.OUT)
        GPIO.setup(self.limit_switch_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)  # Limit switch input with pull-up resistor

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

    def rotate_motor(self):
        """Rotate the motor to the test angle and return to home."""
        target_angle = self.angle
        steps = self.calculate_steps_to_target(target_angle)
        
        GPIO.output(self.dir_pin, GPIO.HIGH)  # Set direction for rotation

        # Rotate motor to target position
        for _ in range(steps):
            GPIO.output(self.pul_pin, GPIO.HIGH)
            time.sleep(0.003)
            GPIO.output(self.pul_pin, GPIO.LOW)
            time.sleep(0.003)

        # Update current position
        self.current_position = target_angle

        print(f"Motor rotated to {self.current_position} degrees.")
        time.sleep(3)

        # After rotation, return to home position
        self.home_motor()

    def calculate_steps_to_target(self, target_angle):
        """Calculate the number of steps needed to reach the target angle."""
        angle_difference = abs(target_angle - self.current_position)
        steps = int(self.step_count * angle_difference / 360)
        return steps

# Usage
motor = MotorController(angle=310)  # Testing with 180 degrees
motor.rotate_motor()
