import time

def log(text, level="INFO"):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [{level}] {text}", flush=True)
