import subprocess

try:
    subprocess.call(["python3", "smt_subprocess.py", "7"], timeout=10)
except subprocess.TimeoutExpired:
    print("TIMEOUT")