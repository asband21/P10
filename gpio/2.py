from gpiozero import LED
from gpiozero.pins.lgpio import LGPIOFactory
from time import sleep

factory = LGPIOFactory(chip=0)

led = LED(17, pin_factory=factory)

while True:
    led.on()
    sleep(1)
    led.off()
    sleep(1)
