from datetime import datetime

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
