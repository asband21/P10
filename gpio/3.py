import time
import lgpio

PIN = 17  # BCM GPIO number, physical pin 11

h = lgpio.gpiochip_open(0)

try:
    lgpio.gpio_claim_output(h, PIN)

    print("GPIO17 HIGH")
    lgpio.gpio_write(h, PIN, 1)
    time.sleep(2)

    print("GPIO17 LOW")
    lgpio.gpio_write(h, PIN, 0)

finally:
    lgpio.gpiochip_close(h)
