import time
import subprocess

# gpio high
print("set high")
subprocess.run(["sudo", "pinctrl", "set", "17", "op", "dh"], check=True)
time.sleep(1)
# gpio lowe
print("set lowe")
subprocess.run(["sudo", "pinctrl", "set", "17", "op", "dl"], check=True)

