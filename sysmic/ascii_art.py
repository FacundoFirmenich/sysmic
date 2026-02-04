"""
Sysmic ASCII Art & Branding Module.
Provides professional-grade visual headers for the CLI.
"""

import sys
import os
import shutil

# ANSI Colors
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def get_terminal_width():
    return shutil.get_terminal_size().columns

def print_centered(text, width=None):
    if width is None:
        width = get_terminal_width()
    for line in text.split('\n'):
        print(line.center(width))

SYSMIC_BANNER = r"""
   _____                  ________  ____    
  / ___/__  ___________  /_  __/  |/  /    
  \__ \/ / / / ___/ __ \  / / / /|_/ /     
 ___/ / /_/ (__  ) / / / / / / /  / /      
/____/\__, /____/_/ /_/ /_/ /_/  /_/       
     /____/                                
   
      B E T A   V E R S I O N   6 . 0
"""

SYSMIC_LOGO_LARGE = """
\033[96m
.d88888b.                             d8b                   
d88P" "Y88b                            Y8P                   
Y88b.                                                        
 "Y88888b.  888  888 .d8888b  88888b.  888  .d8888b          
    "Y88b. 888  888 88K      888 "88b 888 d88P"            
      "888 888  888 "Y8888b. 888  888 888 888              
Y88b  d88P Y88b 888      X88 888  888 888 Y88b.            
 "Y88888P"  "Y88888  88888P' 888  888 888  "Y8888P           
                888                                          
           Y8b d88P                                          
            "Y88P"                                           
\033[0m
"""

def show_splash():
    clear_screen()
    width = get_terminal_width()
    print("\n" * 2)
    print(Colors.CYAN + SYSMIC_LOGO_LARGE + Colors.ENDC)
    print_centered(Colors.HEADER + "S Y S M I C   G E O P H Y S I C A L   F R A M E W O R K" + Colors.ENDC, width)
    print("\n")
    print_centered(Colors.WARNING + "⚠  BETA VERSION - FOR SCIENTIFIC USE ONLY  ⚠" + Colors.ENDC, width)
    print("\n" * 2)

def print_status(message, status="INFO"):
    if status == "INFO":
        print(f"{Colors.BLUE}[INFO]{Colors.ENDC} {message}")
    elif status == "SUCCESS":
        print(f"{Colors.GREEN}[SUCCESS]{Colors.ENDC} {message}")
    elif status == "WARNING":
        print(f"{Colors.WARNING}[WARNING]{Colors.ENDC} {message}")
    elif status == "ERROR":
        print(f"{Colors.FAIL}[ERROR]{Colors.ENDC} {message}")
    elif status == "SYSTEM":
        print(f"{Colors.HEADER}[SYSMIC]{Colors.ENDC} {message}")

def print_menu_header(title):
    print(f"\n{Colors.BOLD}{Colors.UNDERLINE}{title}{Colors.ENDC}")

def input_clean(prompt):
    return input(f"{Colors.GREEN}➜{Colors.ENDC} {prompt} ").strip()
