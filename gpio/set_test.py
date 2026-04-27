from gpiozero import LED
from time import sleep

pin = LED(2)   # BCM GPIO number, not physical pin number

pin.on()       # GPIO 2 HIGH, 3.3 V
sleep(1)

pin.off()      # GPIO 2 LOW, 0 V
sleep(1)

pin.close()

