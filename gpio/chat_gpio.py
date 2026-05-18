import lgpio

GPIO_CHIP = 0      # often gpiochip4 on Raspberry Pi 5, but check with gpioinfo
GPIO_LINE = 17

h = lgpio.gpiochip_open(GPIO_CHIP)
lgpio.gpio_claim_output(h, GPIO_LINE, 0)

# set high
lgpio.gpio_write(h, GPIO_LINE, 1)

# set low
lgpio.gpio_write(h, GPIO_LINE, 0)

lgpio.gpiochip_close(h)
