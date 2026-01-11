from datetime import datetime
import torch

# ANSI color codes
RESET = "\033[0m"
GREEN = "\033[32m"
BLUE = "\033[34m"
YELLOW = "\033[33m"
RED = "\033[31m"
GRAY = "\033[90m"

def _now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def log_pass(message: str):
    print(f"{GREEN}[{_now()}] [PASS] {message}{RESET}")

def log_info(message: str):
    print(f"{BLUE}[{_now()}] [INFO]{RESET} {GRAY}{message}{RESET}")

def log_warning(message: str):
    print(f"{YELLOW}[{_now()}] [WARNING]{RESET} {message}")

def log_error(message: str):
    print(f"{RED}[{_now()}] [ERROR] {message}{RESET}")

def log_gpu():
    log_info(f"CUDA available: {torch.cuda.is_available()}")
    log_info(f"GPU count: {torch.cuda.device_count()}")

    if not torch.cuda.is_available():
        return

    log_info(f"Current device: {torch.cuda.current_device()}")

    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)

        total_mem = props.total_memory / 1024 ** 3
        allocated = torch.cuda.memory_allocated(i) / 1024 ** 3
        reserved = torch.cuda.memory_reserved(i) / 1024 ** 3
        free = total_mem - reserved

        log_info(f"--- GPU {i} ---")
        log_info(f"Name: {props.name}")
        log_info(f"Compute capability: {props.major}.{props.minor}")
        log_info(f"Total memory: {total_mem:.2f} GB")
        log_info(f"Allocated by PyTorch: {allocated:.2f} GB")
        log_info(f"Reserved by PyTorch: {reserved:.2f} GB")
        log_info(f"Free (est.): {free:.2f} GB")