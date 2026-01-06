"""
BERT CLI v1.2.0

By Matias Nisperuza 
══════════════════════════════════════════════════════════════════════════════

New in v1.2.0:
- Fixed paste support
- Gemini CLI-style clean input prompt  
- Terminal color themes: -color light / -color dark
- Multiline paste mode: /*paste
"""

import os
import sys
import time
import threading
import re
from pathlib import Path
from datetime import datetime
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'

# Platform detection
IS_WINDOWS = sys.platform == 'win32'

# ═══════════════════════════════════════════════════════════════════════════════
# TEXTUAL IMPORTS (for styled input)
# ═══════════════════════════════════════════════════════════════════════════════

TEXTUAL_AVAILABLE = False
try:
    from textual.app import App, ComposeResult
    from textual.widgets import Static, Input, TextArea
    from textual.containers import Container, Vertical
    from textual.css.query import NoMatches
    from textual import events
    from textual.binding import Binding
    TEXTUAL_AVAILABLE = True
except ImportError:
    pass

# ═══════════════════════════════════════════════════════════════════════════════
# KEYBOARD INPUT HANDLING (Cross-platform fallback)
# ═══════════════════════════════════════════════════════════════════════════════

if IS_WINDOWS:
    try:
        import msvcrt
        HAS_MSVCRT = True
    except ImportError:
        HAS_MSVCRT = False
else:
    HAS_MSVCRT = False
    try:
        import termios
        import tty
        HAS_TERMIOS = True
    except ImportError:
        HAS_TERMIOS = False


def check_for_esc() -> bool:
    """Check if ESC key was pressed (non-blocking)"""
    if IS_WINDOWS and HAS_MSVCRT:
        if msvcrt.kbhit():
            key = msvcrt.getch()
            if key == b'\x1b':  # ESC
                return True
            elif key == b'\x03':  # Ctrl+C
                return True
    return False


def get_input_line(prompt: str = "") -> str:
    """Fallback input method when Textual is not available."""
    if prompt:
        print(prompt, end='', flush=True)
    
    try:
        line = sys.stdin.readline()
        if line:
            return line.rstrip('\n\r')
        return ""
    except (EOFError, KeyboardInterrupt):
        return ""


# ═══════════════════════════════════════════════════════════════════════════════
# ENGINE IMPORT (Robust - multiple fallback strategies)
# ═══════════════════════════════════════════════════════════════════════════════

ENGINE_AVAILABLE = False
_engine_module = None

def _import_engine():
    """Robust engine import with multiple fallback strategies."""
    global ENGINE_AVAILABLE, _engine_module
    
    # Get the directory where THIS script is located
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        script_dir = os.getcwd()
    
    # Add script directory to sys.path if not already there
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    
    # Strategy 1: Direct import
    try:
        import engine as eng
        _engine_module = eng
        ENGINE_AVAILABLE = True
        return True
    except ImportError:
        pass
    
    # Strategy 2: Package import (bert.engine)
    try:
        import bert.engine as eng
        _engine_module = eng
        ENGINE_AVAILABLE = True
        return True
    except ImportError:
        pass
    
    # Strategy 3: importlib direct file load
    try:
        import importlib.util
        engine_path = os.path.join(script_dir, 'engine.py')
        
        if os.path.exists(engine_path):
            spec = importlib.util.spec_from_file_location("engine", engine_path)
            eng = importlib.util.module_from_spec(spec)
            sys.modules['engine'] = eng
            spec.loader.exec_module(eng)
            _engine_module = eng
            ENGINE_AVAILABLE = True
            return True
    except Exception:
        pass
    
    return False


def get_engine():
    """Get engine instance."""
    if _engine_module and hasattr(_engine_module, 'get_engine'):
        return _engine_module.get_engine()
    return None

def get_token_manager():
    """Get token manager instance."""
    if _engine_module and hasattr(_engine_module, 'get_token_manager'):
        return _engine_module.get_token_manager()
    return None

def get_interrupt_handler():
    """Get interrupt handler instance."""
    if _engine_module and hasattr(_engine_module, 'get_interrupt_handler'):
        return _engine_module.get_interrupt_handler()
    return None

# Run import at module load
_import_engine()


# ═══════════════════════════════════════════════════════════════════════════════
# COLORS & TERMINAL THEMES
# ═══════════════════════════════════════════════════════════════════════════════

class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    ITALIC = "\033[3m"
    CLEAR_LINE = "\033[2K\r"
    HIDE_CURSOR = "\033[?25l"
    SHOW_CURSOR = "\033[?25h"
    
    BLACK = "\033[90m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"
    
    # Dark Sage palette
    SAGE = "\033[38;2;95;140;110m"
    SAGE_LIGHT = "\033[38;2;130;175;145m"
    SAGE_DARK = "\033[38;2;60;90;70m"
    SAGE_DIM = "\033[38;2;80;110;90m"
    
    # Background colors for themes
    BG_DARK = "\033[48;2;0;0;0m"           # Pitch black
    BG_LIGHT = "\033[48;2;245;245;240m"    # Bone white
    BG_SAGE_DARK = "\033[48;2;25;35;28m"   # Dark sage background
    
    # Foreground for themes
    FG_DARK = "\033[38;2;220;220;220m"     # Light text for dark bg
    FG_LIGHT = "\033[38;2;30;30;30m"       # Dark text for light bg
    
    @staticmethod
    def rgb(r, g, b):
        return f"\033[38;2;{max(0,min(255,int(r)))};{max(0,min(255,int(g)))};{max(0,min(255,int(b)))}m"
    
    @staticmethod
    def bg_rgb(r, g, b):
        return f"\033[48;2;{max(0,min(255,int(r)))};{max(0,min(255,int(g)))};{max(0,min(255,int(b)))}m"


# ═══════════════════════════════════════════════════════════════════════════════
# TERMINAL THEME MANAGER
# ═══════════════════════════════════════════════════════════════════════════════

class TerminalTheme:
    """Manages terminal color themes (dark/light mode)"""
    
    DARK = "dark"
    LIGHT = "light"
    
    # Theme definitions - FIXED: Light mode has darker text for visibility
    THEMES = {
        "dark": {
            "name": "Dark",
            "background": (0, 0, 0),          # Pitch black
            "foreground": (220, 220, 220),    # Light gray
            "accent": (130, 175, 145),        # Sage green
            "dim": (100, 100, 100),
        },
        "light": {
            "name": "Light", 
            "background": (245, 245, 240),    # Bone white
            "foreground": (25, 25, 25),       # DARKER - near black for visibility
            "accent": (40, 70, 50),           # DARKER sage for contrast on white
            "dim": (90, 90, 90),              # DARKER dim for visibility
        }
    }
    
    def __init__(self):
        self.current_theme = self.DARK
    
    def set_theme(self, theme: str) -> bool:
        """Set terminal theme (dark/light)"""
        theme = theme.lower().strip()
        
        if theme not in self.THEMES:
            return False
        
        self.current_theme = theme
        self._apply_theme()
        return True
    
    def _apply_theme(self):
        """Apply the current theme to the terminal"""
        theme = self.THEMES[self.current_theme]
        bg = theme["background"]
        fg = theme["foreground"]
        
        # Set background color for entire screen
        # OSC 11 sets background, OSC 10 sets foreground (works in most terminals)
        bg_hex = f"#{bg[0]:02x}{bg[1]:02x}{bg[2]:02x}"
        fg_hex = f"#{fg[0]:02x}{fg[1]:02x}{fg[2]:02x}"
        
        # OSC sequences for terminal emulators (works in iTerm2, Windows Terminal, etc.)
        print(f"\033]11;{bg_hex}\033\\", end='', flush=True)  # Background
        print(f"\033]10;{fg_hex}\033\\", end='', flush=True)  # Foreground
        
        # Also use ANSI for broader compatibility
        print(f"\033[48;2;{bg[0]};{bg[1]};{bg[2]}m", end='', flush=True)
        print(f"\033[38;2;{fg[0]};{fg[1]};{fg[2]}m", end='', flush=True)
        
        # Clear screen with new background
        if IS_WINDOWS:
            os.system('cls')
        else:
            # Clear and reset scroll position
            print("\033[2J\033[H", end='', flush=True)
    
    def reset_theme(self):
        """Reset to default terminal colors"""
        print("\033]110\033\\", end='', flush=True)  # Reset foreground
        print("\033]111\033\\", end='', flush=True)  # Reset background
        print(Colors.RESET, end='', flush=True)
    
    def get_fg_color(self) -> str:
        """Get current theme's foreground color code"""
        fg = self.THEMES[self.current_theme]["foreground"]
        return f"\033[38;2;{fg[0]};{fg[1]};{fg[2]}m"
    
    def get_accent_color(self) -> str:
        """Get current theme's accent (sage) color code"""
        accent = self.THEMES[self.current_theme]["accent"]
        return f"\033[38;2;{accent[0]};{accent[1]};{accent[2]}m"
    
    def get_dim_color(self) -> str:
        """Get current theme's dim color code"""
        dim = self.THEMES[self.current_theme]["dim"]
        return f"\033[38;2;{dim[0]};{dim[1]};{dim[2]}m"


# Global theme instance
_theme = TerminalTheme()


def supports_color():
    if os.environ.get('NO_COLOR'):
        return False
    return hasattr(sys.stdout, 'isatty') and sys.stdout.isatty()


def term_width():
    try:
        return os.get_terminal_size().columns
    except:
        return 80


# ═══════════════════════════════════════════════════════════════════════════════
# GRADIENTS (per model family) - Updated with theme-aware colors
# ═══════════════════════════════════════════════════════════════════════════════

GRADIENTS = {
    # Bert Nano: Sage green (friendly, approachable)
    "nano": [
        (95, 140, 110), (110, 155, 125), (130, 175, 145), (150, 195, 165),
        (170, 210, 180), (150, 195, 165), (130, 175, 145), (110, 155, 125),
    ],
    # Bert Mini: Teal-Mint (fresh, balanced)
    "mini": [
        (16, 163, 127), (30, 175, 140), (50, 190, 155), (75, 205, 170),
        (100, 215, 185), (75, 205, 170), (50, 190, 155), (30, 175, 140),
    ],
    # Bert Main: Sage-Olive (earthy, natural) - thinking model
    "main": [
        (107, 142, 85), (119, 156, 95), (134, 169, 108), (148, 182, 120),
        (162, 195, 132), (148, 182, 120), (134, 169, 108), (119, 156, 95),
    ],
    # Bert Max: Orange-Coral (Claude style - warm, powerful)
    "max": [
        (204, 119, 77), (218, 130, 85), (232, 145, 95), (245, 160, 107),
        (255, 175, 120), (245, 160, 107), (232, 145, 95), (218, 130, 85),
    ],
    # Bert Coder: Cyan (tech, code)
    "coder": [
        (40, 160, 180), (60, 175, 195), (80, 190, 210), (100, 205, 220),
        (120, 215, 230), (100, 205, 220), (80, 190, 210), (60, 175, 195),
    ],
    # Bert Max-Coder: Silver (clean, professional)
    "maxcoder": [
        (140, 150, 160), (160, 170, 180), (180, 190, 200), (200, 210, 220),
        (220, 225, 230), (200, 210, 220), (180, 190, 200), (160, 170, 180),
    ],
}

# Primary colors for each family
FAMILY_COLORS = {
    "nano": (150, 195, 165),   # Sage
    "mini": (75, 205, 170),    # Teal
    "main": (148, 182, 120),   # Sage-Olive
    "max": (245, 160, 107),    # Orange-Coral
    "coder": (100, 205, 220),  # Cyan
    "maxcoder": (200, 210, 220),  # Silver
}

# Theme-aware input bar colors
def get_input_bar_colors() -> dict:
    """Get input bar colors appropriate for current theme."""
    if _theme.current_theme == "light":
        # LIGHT MODE: Dark colors on bone white background
        return {
            "border": (60, 90, 70),          # Dark sage border
            "background": (230, 235, 232),   # Light sage-tinted white
            "text": (20, 30, 25),            # Very dark text for visibility
            "placeholder": (100, 120, 110),  # Medium gray-sage
            "cursor": (40, 70, 50),          # Dark sage cursor
        }
    else:
        # DARK MODE: Light colors on black background
        return {
            "border": (80, 120, 95),         # Medium sage border
            "background": (25, 35, 28),      # Very dark sage bg
            "text": (200, 220, 210),         # Light sage text
            "placeholder": (100, 130, 115),  # Muted sage
            "cursor": (130, 175, 145),       # Bright sage
        }


# ═══════════════════════════════════════════════════════════════════════════════
# ANIMATION FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

BRAILLE = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]


def loading_spinner(message, family, stop_event, speed=0.08):
    """Braille spinner with gradient color"""
    gradient = GRADIENTS.get(family, GRADIENTS["nano"])
    idx = 0
    
    print(Colors.HIDE_CURSOR, end='', flush=True)
    try:
        while not stop_event.is_set():
            frame = BRAILLE[idx % len(BRAILLE)]
            r, g, b = gradient[idx % len(gradient)]
            
            output = f"\033[38;2;{r};{g};{b}m{frame} {message}\033[0m"
            sys.stdout.write(Colors.CLEAR_LINE + output)
            sys.stdout.flush()
            
            idx += 1
            time.sleep(speed)
        
        sys.stdout.write(Colors.CLEAR_LINE)
        sys.stdout.flush()
    finally:
        print(Colors.SHOW_CURSOR, end='', flush=True)


def shimmer_text(text, family, duration=1.2, speed=0.05):
    """Shimmer animation for text"""
    if not supports_color():
        print(text)
        return
    
    gradient = GRADIENTS.get(family, GRADIENTS["nano"])
    start = time.time()
    offset = 0
    
    print(Colors.HIDE_CURSOR, end='', flush=True)
    try:
        while time.time() - start < duration:
            output = ""
            for i, char in enumerate(text):
                r, g, b = gradient[(i + offset) % len(gradient)]
                output += f"\033[38;2;{r};{g};{b}m{char}"
            output += Colors.RESET
            
            sys.stdout.write(Colors.CLEAR_LINE + output)
            sys.stdout.flush()
            
            offset += 1
            time.sleep(speed)
        
        # Final static
        r, g, b = gradient[len(gradient) // 2]
        sys.stdout.write(Colors.CLEAR_LINE + f"\033[38;2;{r};{g};{b}m{text}\033[0m\n")
        sys.stdout.flush()
    finally:
        print(Colors.SHOW_CURSOR, end='', flush=True)


def animate_banner(lines, duration=1.8):
    """Animate BERT CLI banner with flowing sage gradient"""
    if not supports_color():
        for line in lines:
            print(line)
        return
    
    gradient = GRADIENTS["nano"]  # Always sage for banner
    start = time.time()
    offset = 0
    
    # Print initial lines to establish position
    for line in lines:
        print()
    
    print(Colors.HIDE_CURSOR, end='', flush=True)
    
    try:
        while time.time() - start < duration:
            # Move cursor up
            sys.stdout.write(f"\033[{len(lines)}A")
            
            for line_idx, line in enumerate(lines):
                output = ""
                for char_idx, char in enumerate(line):
                    grad_pos = (char_idx + line_idx * 2 + offset) % len(gradient)
                    r, g, b = gradient[grad_pos]
                    output += f"\033[38;2;{r};{g};{b}m{char}"
                output += Colors.RESET
                print(output)
            
            offset = (offset + 1) % len(gradient)
            time.sleep(0.06)
        
        # Final static gradient
        sys.stdout.write(f"\033[{len(lines)}A")
        for line_idx, line in enumerate(lines):
            r, g, b = gradient[min(line_idx, len(gradient) - 1)]
            print(f"\033[38;2;{r};{g};{b}m{line}\033[0m")
    
    except Exception:
        pass
    finally:
        print(Colors.SHOW_CURSOR, end='', flush=True)


def gradient_text(text, family):
    """Apply gradient to text (static)"""
    gradient = GRADIENTS.get(family, GRADIENTS["nano"])
    output = ""
    for i, char in enumerate(text):
        r, g, b = gradient[i % len(gradient)]
        output += f"\033[38;2;{r};{g};{b}m{char}"
    return output + Colors.RESET


# ═══════════════════════════════════════════════════════════════════════════════
# STYLED INPUT (Paste-Friendly Gemini-Style)
# ═══════════════════════════════════════════════════════════════════════════════

def get_styled_input(placeholder: str = "Type your message...", 
                     multiline: bool = False,
                     context_bar: str = "") -> str:
    """
    Get styled input - PASTE-FRIENDLY VERSION.
    
    The box-with-cursor approach breaks paste. This version uses:
    - Decorative header line (visual only)
    - Standard input below (handles any paste length)
    
    This is inspired by Gemini CLI's clean approach.
    """
    
    # Get theme-aware colors
    colors = get_input_bar_colors()
    
    border = colors["border"]
    text_color = colors["text"]
    placeholder_color = colors["placeholder"]
    
    bc = f"\033[38;2;{border[0]};{border[1]};{border[2]}m"
    tc = f"\033[38;2;{text_color[0]};{text_color[1]};{text_color[2]}m"
    plc = f"\033[38;2;{placeholder_color[0]};{placeholder_color[1]};{placeholder_color[2]}m"
    
    # Get prompt icon color (sage - theme aware)
    if _theme.current_theme == "light":
        sage = (60, 100, 75)  # Darker sage for light mode
    else:
        sage = FAMILY_COLORS["nano"]  # Normal sage for dark mode
    prompt_color = f"\033[38;2;{sage[0]};{sage[1]};{sage[2]}m"
    
    # Show context bar if provided
    if context_bar:
        print(context_bar)
    
    if multiline:
        # === MULTILINE MODE ===
        # For pasting large texts - press Enter twice to submit
        
        print(f"\n{bc}─── Paste Mode (press Enter twice to submit) ───{Colors.RESET}")
        sys.stdout.write(f"{prompt_color}❯{Colors.RESET} {tc}")
        sys.stdout.flush()
        
        lines = []
        empty_count = 0
        first_line = True
        
        try:
            while True:
                if first_line:
                    line = input()
                    first_line = False
                else:
                    # Continuation prompt
                    sys.stdout.write(f"{prompt_color}│{Colors.RESET} {tc}")
                    sys.stdout.flush()
                    line = input()
                
                if line == "":
                    empty_count += 1
                    if empty_count >= 2:
                        break
                else:
                    empty_count = 0
                    lines.append(line)
        except (EOFError, KeyboardInterrupt):
            pass
        
        sys.stdout.write(Colors.RESET)
        print(f"{bc}─── End ───{Colors.RESET}\n")
        
        return "\n".join(lines)
    
    else:
        # === SINGLE LINE MODE - PASTE FRIENDLY ===
        # No box around input = paste works perfectly
        
        # Just show a clean sage prompt
        # Input is NOT inside a box, so any length paste works
        print()  # Blank line for spacing
        sys.stdout.write(f"{prompt_color}❯{Colors.RESET} {tc}")
        sys.stdout.flush()
        
        try:
            user_input = input()
        except (EOFError, KeyboardInterrupt):
            user_input = ""
        
        sys.stdout.write(Colors.RESET)
        
        return user_input


# ═══════════════════════════════════════════════════════════════════════════════
# THINKING BOX
# ═══════════════════════════════════════════════════════════════════════════════

def print_thinking_box(thinking_content: str):
    """Print the thinking box with content"""
    lines = thinking_content.strip().split('\n')
    max_width = 58
    
    print(f"\n{Colors.DIM}┌{'─' * max_width}┐{Colors.RESET}")
    print(f"{Colors.DIM}│{Colors.RESET} {Colors.CYAN}🧠 Thinking...{Colors.RESET}{' ' * (max_width - 16)}{Colors.DIM}│{Colors.RESET}")
    print(f"{Colors.DIM}├{'─' * max_width}┤{Colors.RESET}")
    
    # Show last 6 lines of thinking
    display_lines = lines[-6:] if len(lines) > 6 else lines
    
    for line in display_lines:
        if len(line) > max_width - 4:
            line = line[:max_width - 7] + "..."
        padding = max_width - len(line) - 2
        print(f"{Colors.DIM}│{Colors.RESET} {Colors.ITALIC}{line}{Colors.RESET}{' ' * padding}{Colors.DIM}│{Colors.RESET}")
    
    for _ in range(6 - len(display_lines)):
        print(f"{Colors.DIM}│{' ' * max_width}│{Colors.RESET}")
    
    print(f"{Colors.DIM}└{'─' * max_width}┘{Colors.RESET}\n")


# ═══════════════════════════════════════════════════════════════════════════════
# BERT CLI (Enhanced with styled input and themes)
# ═══════════════════════════════════════════════════════════════════════════════

class BertCLI:
    VERSION = "1.2.0"
    VERSION_NAME = "Stable"
    
    # Model info: (display_name, base_model, family, has_thinking)
    MODELS = {
        "nano": ("Bert Nano", "LFM2-700M", "nano", False),
        "mini": ("Bert Mini", "LFM2-1.2B", "mini", False),
        "main": ("Bert Main", "Qwen3-1.7B", "main", True),
        "bert": ("Bert Main", "Qwen3-1.7B", "main", True),
        "1": ("Bert Main", "Qwen3-1.7B", "main", True),
        "max": ("Bert Max", "LFM2-2.6B-Exp", "max", False),
        "coder": ("Bert Coder", "Qwen2.5 coder 1.5B instruct", "coder", False),
        "maxcoder": ("Bert Max-Coder", "Qwen2.5-3B-Instruct", "maxcoder", False),
        "max-coder": ("Bert Max-Coder", "Qwen2.5 coder-3B-Instruct", "maxcoder", False),
    }
    
    PLACEHOLDERS = [
        "Ask Bert how to prepare an omelet",
        "Ask about Python best practices",
        "Need help debugging code?",
        "Want to learn about recursion?",
        "Ask for a code review",
        "Need help with git commands?",
        "Ask about design patterns",
        "Help me write a function...",
        "Explain async/await to me",
        "How do I use Docker?",
    ]
    
    def __init__(self):
        self.engine = None
        self.mode = "nano"
        self.quant = "int4"
        self.debug = False
        self.thinking_mode = False
        self._placeholder_index = 0
        self.multiline_mode = False  # For large text paste
        self.theme = _theme  # Terminal theme manager
        
        if ENGINE_AVAILABLE:
            self.engine = get_engine()
            self.token_manager = get_token_manager()
            self.interrupt_handler = get_interrupt_handler()
        else:
            self.token_manager = None
            self.interrupt_handler = None
    
    def clear(self):
        os.system('cls' if IS_WINDOWS else 'clear')
    
    def banner(self):
        """Print animated banner"""
        print()
        lines = [
            "██████╗ ███████╗██████╗ ████████╗     ██████╗██╗     ██╗",
            "██╔══██╗██╔════╝██╔══██╗╚══██╔══╝    ██╔════╝██║     ██║",
            "██████╔╝█████╗  ██████╔╝   ██║       ██║     ██║     ██║",
            "██╔══██╗██╔══╝  ██╔══██╗   ██║       ██║     ██║     ██║",
            "██████╔╝███████╗██║  ██║   ██║       ╚██████╗███████╗██║",
            "╚═════╝ ╚══════╝╚═╝  ╚═╝   ╚═╝        ╚═════╝╚══════╝╚═╝",
        ]
        
        width = term_width()
        centered_lines = [line.center(width) for line in lines]
        
        animate_banner(centered_lines, duration=1.8)
        
        print()
        print(f"{Colors.DIM}{'by Matias Nisperuza — 2025'.center(width)}{Colors.RESET}")
        print(f"{Colors.DIM}{f'Version {self.VERSION}'.center(width)}{Colors.RESET}")
        print()
    
    def get_placeholder(self) -> str:
        placeholder = self.PLACEHOLDERS[self._placeholder_index % len(self.PLACEHOLDERS)]
        self._placeholder_index += 1
        return placeholder
    
    def prompt(self) -> str:
        """Legacy prompt for fallback mode"""
        sage = FAMILY_COLORS["nano"]
        return f"\033[38;2;{sage[0]};{sage[1]};{sage[2]}m❯{Colors.RESET} "
    
    def context_bar(self) -> str:
        """Generate context bar with gradient model name"""
        model_info = self.MODELS.get(self.mode, self.MODELS["nano"])
        name, base, family, _ = model_info
        
        model_name_colored = gradient_text(name, family)
        
        bar = f"[{model_name_colored} • {Colors.DIM}{base}{Colors.RESET} • {self.quant.upper()}"
        
        if self.token_manager and self.token_manager.has_valid_token():
            remaining = self.token_manager.get_remaining()
            bar += f" │ {Colors.CYAN}{remaining:,}{Colors.RESET} tokens"
        
        bar += "]"
        return bar
    
    def handle_color_command(self, args: str) -> bool:
        """Handle -color light / -color dark commands"""
        args = args.lower().strip()
        
        if args == "light":
            self.theme.set_theme("light")
            print(f"{Colors.GREEN}✓{Colors.RESET} Theme set to Light (bone white)")
            return True
        elif args == "dark":
            self.theme.set_theme("dark")
            print(f"{Colors.GREEN}✓{Colors.RESET} Theme set to Dark (pitch black)")
            return True
        elif args == "reset":
            self.theme.reset_theme()
            print(f"{Colors.GREEN}✓{Colors.RESET} Theme reset to default")
            return True
        else:
            print(f"{Colors.YELLOW}Usage: -color light | -color dark | -color reset{Colors.RESET}")
            return True
    
    def handle_command(self, user_input: str):
        """Handle slash commands and special commands"""
        
        lower = user_input.lower().strip()
        
        # Exit commands
        if lower in ["/*exit", "/*quit", "/*q", "/exit", "/quit"]:
            self.theme.reset_theme()  # Reset theme on exit
            print(f"\n{Colors.DIM}Goodbye! 👋{Colors.RESET}\n")
            return False
        
        # Color theme commands
        if lower.startswith("-color "):
            args = user_input[7:].strip()
            self.handle_color_command(args)
            return True
        
        # Multiline mode toggle
        if lower in ["/*paste", "/paste", "/*multiline", "/multiline"]:
            self.multiline_mode = not self.multiline_mode
            status = "enabled" if self.multiline_mode else "disabled"
            print(f"{Colors.GREEN}✓{Colors.RESET} Multiline/paste mode {status}")
            return True
        
        # Help
        if lower in ["/*help", "/help", "/*h"]:
            self.show_help()
            return True
        
        # Status
        if lower in ["/*status", "/status"]:
            self.show_status()
            return True
        
        # Clear screen
        if lower in ["/*clear", "/clear", "/*cls"]:
            self.clear()
            self.banner()
            return True
        
        # Clear memory
        if lower in ["/*memory", "/memory", "/*mem"]:
            if self.engine:
                self.engine.clear_memory()
            print(f"{Colors.GREEN}✓{Colors.RESET} Memory cleared")
            return True
        
        # Debug toggle
        if lower in ["/*debug", "/debug"]:
            self.debug = not self.debug
            print(f"{Colors.GREEN}✓{Colors.RESET} Debug {'enabled' if self.debug else 'disabled'}")
            return True
        
        # Token commands
        if lower.startswith("/*token ") or lower.startswith("/token "):
            parts = user_input.split(' ', 1)
            if len(parts) > 1:
                token = parts[1].strip()
                if self.token_manager:
                    success, message = self.token_manager.set_token(token)
                    if success:
                        print(f"\n{Colors.GREEN}✓{Colors.RESET} {message}\n")
                    else:
                        print(f"\n{Colors.RED}✗{Colors.RESET} {message}")
                        print(f"Get a valid token at: {Colors.CYAN}https://mnisperuza.github.io/bert-cli/{Colors.RESET}\n")
            return True
        
        if lower in ["/*tokens", "/tokens", "/*token"]:
            if self.token_manager:
                status = self.token_manager.get_status()
                if status.get("has_token"):
                    print(f"\n{Colors.BOLD}Token Status:{Colors.RESET}")
                    print(f"  Remaining: {Colors.CYAN}{status['remaining']:,}{Colors.RESET} / {status['limit']:,}")
                    print(f"  Used: {status['used']:,} ({status['percent_used']}%)")
                    print()
                else:
                    print(f"\n{Colors.YELLOW}{status['message']}{Colors.RESET}\n")
            return True
        
        # Model switching
        for model_key in ["nano", "mini", "main", "bert", "1", "max", "coder", "maxcoder", "max-coder"]:
            if lower == f"bert {model_key}" or lower == model_key:
                actual_key = "main" if model_key in ["bert", "1"] else model_key.replace("-", "")
                self.load_model(actual_key)
                return True
        
        # Quantization switching
        for q in ["int4", "int8", "fp16", "fp32"]:
            if lower == f"bert {q}" or lower == q:
                if self.engine and self.engine.model:
                    self.load_model(self.mode, q, show_picker=False)
                else:
                    self.quant = q
                    print(f"{Colors.GREEN}✓{Colors.RESET} Quantization set to {q.upper()}")
                return True
        
        return None
    
    def pick_quant(self, model_name, family):
        """Quantization picker with polished UI"""
        r, g, b = FAMILY_COLORS.get(family, FAMILY_COLORS["nano"])
        color = f"\033[38;2;{r};{g};{b}m"
        
        print(f"\n{color}┌{'─' * 48}┐{Colors.RESET}")
        print(f"{color}│{Colors.RESET} {Colors.BOLD}🎛️  Select Quantization for {model_name}{Colors.RESET}")
        print(f"{color}└{'─' * 48}┘{Colors.RESET}\n")
        
        print(f"  {Colors.DIM}[1] INT4{Colors.RESET} — Balanced ⭐ (4GB VRAM)")
        print(f"  {Colors.DIM}[2] INT8{Colors.RESET} — High quality (6GB+)")
        print(f"  {Colors.DIM}[3] FP16{Colors.RESET} — Best quality (8GB+)")
        print(f"  {Colors.DIM}[4] FP32{Colors.RESET} — CPU / Full precision")
        print(f"\n  {Colors.DIM}Press Enter for INT4{Colors.RESET}")
        
        try:
            choice = input(f"  {color}Your choice (1-4):{Colors.RESET} ").strip()
            quant_map = {"1": "int4", "2": "int8", "3": "fp16", "4": "fp32", "": "int4"}
            selected = quant_map.get(choice, "int4")
            print(f"\n  {Colors.GREEN}✓ Selected {selected.upper()}{Colors.RESET}")
            return selected
        except:
            return "int4"
    
    def load_model(self, mode: str, quant: str = None, show_picker: bool = True):
        """Load a model with animations"""
        if not ENGINE_AVAILABLE or not self.engine:
            print(f"{Colors.RED}✗ Engine not available{Colors.RESET}")
            return False
        
        model_info = self.MODELS.get(mode, self.MODELS["nano"])
        name, base, family, _ = model_info
        
        print()
        shimmer_text(f"→ {name}", family, duration=1.0)
        
        if quant is None and show_picker:
            quant = self.pick_quant(name, family)
        elif quant is None:
            quant = self.quant
        
        print()
        stop_event = threading.Event()
        
        thread = threading.Thread(
            target=loading_spinner,
            args=(f"Loading {name}...", family, stop_event)
        )
        thread.daemon = True
        thread.start()
        
        success, message = self.engine.load_model(mode, quant)
        
        stop_event.set()
        thread.join(timeout=1.0)
        
        if success:
            self.mode = mode
            self.quant = quant
            print()
            shimmer_text(f"✓ {name} ready!", family, duration=0.8)
            print()
            return True
        else:
            print(f"\n{Colors.RED}✗ {message}{Colors.RESET}\n")
            return False
    
    def init_engine(self):
        """Initialize engine with default model (nano) after banner"""
        if not ENGINE_AVAILABLE:
            print(f"{Colors.DIM}Demo mode (engine not available){Colors.RESET}\n")
            return
        
        if self.engine:
            print(f"  {Colors.DIM}🖥️  {self.engine.get_device_info()}{Colors.RESET}\n")
        
        name, base, family, _ = self.MODELS["nano"]
        quant = self.pick_quant(name, family)
        self.quant = quant
        
        print()
        stop_event = threading.Event()
        
        thread = threading.Thread(
            target=loading_spinner,
            args=("Initializing Bert Nano...", "nano", stop_event)
        )
        thread.daemon = True
        thread.start()
        
        try:
            success, _ = self.engine.load_model(mode="nano", quant=quant)
        except Exception as e:
            if self.debug:
                print(f"\n{Colors.RED}Error: {e}{Colors.RESET}")
            success = False
        
        stop_event.set()
        thread.join(timeout=1.0)
        
        if success:
            print()
            shimmer_text("✓ Bert Nano ready!", "nano", duration=0.8)
        print()
    
    def show_token_required(self):
        """Show token required message"""
        print(f"""
{Colors.DIM}┌──────────────────────────────────────────────────────┐{Colors.RESET}
{Colors.DIM}│{Colors.RESET} {Colors.YELLOW}🔒 Token Required{Colors.RESET}
{Colors.DIM}│{Colors.RESET}
{Colors.DIM}│{Colors.RESET} Get your free token at:
{Colors.DIM}│{Colors.RESET} {Colors.CYAN}https://mnisperuza.github.io/bert-cli/{Colors.RESET}
{Colors.DIM}│{Colors.RESET}
{Colors.DIM}│{Colors.RESET} Then use: {Colors.GREEN}/*token YOUR-TOKEN-HERE{Colors.RESET}
{Colors.DIM}└──────────────────────────────────────────────────────┘{Colors.RESET}
""")
    
    def show_help(self):
        """Show help with gradient model names"""
        
        nano = gradient_text("bert nano", "nano")
        mini = gradient_text("bert mini", "mini")
        main = gradient_text("bert main", "main")
        max_m = gradient_text("bert max", "max")
        coder = gradient_text("bert coder", "coder")
        maxcoder = gradient_text("bert maxcoder", "maxcoder")
        
        print(f"""
{Colors.DIM}{'═' * 60}{Colors.RESET}
  {Colors.BOLD}BERT CLI v{self.VERSION}{Colors.RESET}
{Colors.DIM}{'═' * 60}{Colors.RESET}

  {Colors.BOLD}Models:{Colors.RESET}
    {nano}      {Colors.DIM}LFM2-700M (fastest){Colors.RESET}
    {mini}      {Colors.DIM}LFM2-1.2B (balanced){Colors.RESET}
    {main}      {Colors.DIM}Qwen3-1.7B (thinking 🧠){Colors.RESET}
    {max_m}       {Colors.DIM}LFM2-2.6B(reasoning){Colors.RESET}
    {coder}     {Colors.DIM} Qwen 2.5 coder 1.5B-Instruct (code){Colors.RESET}
    {maxcoder}  {Colors.DIM}Qwen2.5 coder-3B-Instruct (heavy code){Colors.RESET}

  {Colors.BOLD}Quantization:{Colors.RESET}
    bert int4      {Colors.DIM}Balanced ⭐{Colors.RESET}
    bert int8      {Colors.DIM}High quality{Colors.RESET}
    bert fp16      {Colors.DIM}Best quality{Colors.RESET}
    bert fp32      {Colors.DIM}CPU / Full precision{Colors.RESET}

  {Colors.BOLD}Terminal Themes:{Colors.RESET} {Colors.SAGE}NEW!{Colors.RESET}
    -color dark    {Colors.DIM}Pitch black background (default){Colors.RESET}
    -color light   {Colors.DIM}Bone white background{Colors.RESET}
    -color reset   {Colors.DIM}Reset to terminal default{Colors.RESET}

  {Colors.BOLD}Input Modes:{Colors.RESET}
    /*paste        {Colors.DIM}Toggle multiline paste mode{Colors.RESET}

  {Colors.BOLD}Thinking Mode:{Colors.RESET} {Colors.CYAN}(Only bert main){Colors.RESET}
    /*think - your question here
    {Colors.DIM}Shows model's reasoning in a box, counts only response tokens{Colors.RESET}

  {Colors.BOLD}Token Commands:{Colors.RESET}
    /*token XXXX   {Colors.DIM}Set your token key{Colors.RESET}
    /*tokens       {Colors.DIM}Show token status{Colors.RESET}

  {Colors.BOLD}File Commands:{Colors.RESET}
    @path/to/file  {Colors.DIM}Reference a file in your query{Colors.RESET}

  {Colors.BOLD}During Generation:{Colors.RESET}
    ESC            {Colors.DIM}Stop generation{Colors.RESET}
    Ctrl+C         {Colors.DIM}Stop generation{Colors.RESET}

  {Colors.BOLD}Commands:{Colors.RESET}
    /*help         {Colors.DIM}Show this help{Colors.RESET}
    /*status       {Colors.DIM}Show current status{Colors.RESET}
    /*clear        {Colors.DIM}Clear screen{Colors.RESET}
    /*memory       {Colors.DIM}Clear conversation memory{Colors.RESET}
    /*exit         {Colors.DIM}Exit Bert{Colors.RESET}

  {Colors.YELLOW}Get your free token at:{Colors.RESET}
  {Colors.CYAN}https://mnisperuza.github.io/bert-cli/{Colors.RESET}

{Colors.DIM}{'═' * 60}{Colors.RESET}
""")
    
    def show_status(self):
        """Show current status"""
        model_info = self.MODELS.get(self.mode, self.MODELS["nano"])
        name, base, family, has_thinking = model_info
        model_name_colored = gradient_text(name, family)
        
        print(f"\n{Colors.BOLD}Status:{Colors.RESET}")
        print(f"  Model: {model_name_colored}")
        print(f"  Base: {base}")
        print(f"  Quant: {self.quant.upper()}")
        print(f"  Thinking: {'Available 🧠' if has_thinking else 'Not available'}")
        print(f"  Theme: {self.theme.THEMES[self.theme.current_theme]['name']}")
        print(f"  Multiline: {'Enabled' if self.multiline_mode else 'Disabled'}")
        
        if self.engine:
            print(f"  Device: {self.engine.get_device_info()}")
            print(f"  Model loaded: {'Yes' if self.engine.model else 'No'}")
        
        if self.token_manager:
            status = self.token_manager.get_status()
            if status.get("has_token"):
                print(f"  Tokens: {Colors.CYAN}{status['remaining']:,}{Colors.RESET} / {status['limit']:,}")
            else:
                print(f"  Tokens: {Colors.YELLOW}No token{Colors.RESET}")
        
        print()
    
    def query(self, user_input: str, think_mode: bool = False):
        """Send query to model"""
        
        if not self.token_manager or not self.token_manager.has_valid_token():
            self.show_token_required()
            return
        
        if not self.engine or not self.engine.model:
            print(f"\n{Colors.YELLOW}No model loaded. Loading default...{Colors.RESET}")
            if not self.load_model(self.mode, self.quant, show_picker=False):
                return
        
        model_info = self.MODELS.get(self.mode, self.MODELS["nano"])
        name, base, family, has_thinking = model_info
        
        if think_mode and not has_thinking:
            print(f"\n{Colors.YELLOW}⚠️  Thinking mode only works with Bert Main (Qwen3-1.7B){Colors.RESET}")
            print(f"{Colors.DIM}   Current model: {name}{Colors.RESET}")
            print(f"{Colors.DIM}   Switch with: bert main{Colors.RESET}\n")
            return
        
        file_content, enhanced_prompt = self.engine.process_file_request(user_input)
        
        if file_content:
            paths = self.engine.file_handler.extract_paths(user_input)
            for path_str in paths:
                path = self.engine.file_handler.resolve_path(path_str)
                if path:
                    print(f"{Colors.DIM}📂 Found: {path.name}{Colors.RESET}")
            print()
        
        self.interrupt_handler.reset()
        
        r, g, b = FAMILY_COLORS.get(family, FAMILY_COLORS["nano"])
        model_color = f"\033[38;2;{r};{g};{b}m"
        
        thinking_content = ""
        response_content = ""
        response_tokens = 0
        in_thinking = False
        
        bert_label = gradient_text("Bert", family)
        print(f"{bert_label}: ", end='', flush=True)
        
        try:
            for chunk in self.engine.generate_stream(enhanced_prompt):
                if check_for_esc():
                    self.engine.stop_generation()
                    print(f"\n{Colors.DIM}[Stopped]{Colors.RESET}")
                    break
                
                chunk_type = chunk.get("type", "")
                content = chunk.get("content", "")
                
                if chunk_type == "thinking":
                    thinking_content += content
                    in_thinking = True
                
                elif chunk_type == "token":
                    if content.strip() in ['assistant', 'user', 'system']:
                        continue
                    
                    response_content += content
                    response_tokens += 1
                    
                    print(content, end='', flush=True)
                    time.sleep(0.008)
                
                elif chunk_type == "status":
                    print(f"\n{Colors.DIM}{content}{Colors.RESET}", flush=True)
                
                elif chunk_type == "done":
                    print()
                    
                    thinking = chunk.get("thinking", "") or thinking_content
                    if think_mode and thinking.strip():
                        print_thinking_box(thinking)
                    
                    resp_tokens = chunk.get("response_tokens", response_tokens)
                    remaining = chunk.get("tokens_remaining", 0)
                    
                    print(f"\n{Colors.DIM}─── {resp_tokens} tokens │ {remaining:,} remaining ───{Colors.RESET}")
                
                elif chunk_type == "error":
                    print(f"\n{Colors.RED}Error: {content}{Colors.RESET}")
        
        except KeyboardInterrupt:
            self.engine.stop_generation()
            print(f"\n{Colors.DIM}[Interrupted]{Colors.RESET}")
        except Exception as e:
            print(f"\n{Colors.RED}Error: {e}{Colors.RESET}")
            if self.debug:
                import traceback
                traceback.print_exc()
    
    def run(self):
        """Main CLI loop with styled input"""
        
        self.clear()
        self.banner()
        
        self.init_engine()
        
        # Tips box with sage accent
        r_nano = FAMILY_COLORS["nano"]
        r_main = FAMILY_COLORS["main"]
        r_coder = FAMILY_COLORS["coder"]
        
        print(f"{Colors.DIM}┌─────────────────────────────────────────────────────────┐{Colors.RESET}")
        print(f"{Colors.DIM}│{Colors.RESET} Models: \033[38;2;{r_nano[0]};{r_nano[1]};{r_nano[2]}mnano\033[0m • mini • \033[38;2;{r_main[0]};{r_main[1]};{r_main[2]}mmain 🧠\033[0m • max • \033[38;2;{r_coder[0]};{r_coder[1]};{r_coder[2]}mcoder\033[0m   {Colors.DIM}│{Colors.RESET}")
        print(f"{Colors.DIM}│{Colors.RESET} Quant:  bert int4 / int8 / fp16 / fp32                  {Colors.DIM}│{Colors.RESET}")
        print(f"{Colors.DIM}│{Colors.RESET} Theme:  -color light / -color dark                     {Colors.DIM}│{Colors.RESET}")
        print(f"{Colors.DIM}│{Colors.RESET} Think:  /*think - question (bert main only) 🧠         {Colors.DIM}│{Colors.RESET}")
        print(f"{Colors.DIM}│{Colors.RESET} Paste:  /*paste to toggle multiline mode               {Colors.DIM}│{Colors.RESET}")
        print(f"{Colors.DIM}│{Colors.RESET} Stop:   Press ESC during generation                    {Colors.DIM}│{Colors.RESET}")
        print(f"{Colors.DIM}│{Colors.RESET} Help:   /*help                                         {Colors.DIM}│{Colors.RESET}")
        print(f"{Colors.DIM}└─────────────────────────────────────────────────────────┘{Colors.RESET}\n")
        
        print(f"{Colors.DIM}{'─' * 60}{Colors.RESET}")
        
        while True:
            try:
                # Show context bar
                ctx = self.context_bar()
                
                # Get input using styled input bar
                placeholder = self.get_placeholder()
                user_input = get_styled_input(
                    placeholder=placeholder,
                    multiline=self.multiline_mode,
                    context_bar=ctx
                )
                
                if not user_input:
                    hint = self.get_placeholder()
                    print(f"{Colors.DIM}💡 {hint}{Colors.RESET}\n")
                    continue
                
                # Check for /*think command
                think_mode = False
                if user_input.lower().startswith("/*think ") or user_input.lower().startswith("/think "):
                    parts = user_input.split(' ', 1)
                    if len(parts) > 1:
                        user_input = parts[1].strip()
                        think_mode = True
                    else:
                        print(f"{Colors.YELLOW}Usage: /*think - your question here{Colors.RESET}")
                        continue
                
                # Handle commands
                result = self.handle_command(user_input)
                
                if result is False:
                    break
                elif result is True:
                    print(f"{Colors.DIM}{'─' * 60}{Colors.RESET}")
                    continue
                
                # Query model
                self.query(user_input, think_mode=think_mode)
                
                print(f"{Colors.DIM}{'─' * 60}{Colors.RESET}")
                
            except KeyboardInterrupt:
                self.theme.reset_theme()
                print(f"\n\n{Colors.DIM}Goodbye! 👋{Colors.RESET}\n")
                break
            except EOFError:
                self.theme.reset_theme()
                print(f"\n\n{Colors.DIM}Goodbye! 👋{Colors.RESET}\n")
                break


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    cli = BertCLI()
    cli.run()


if __name__ == "__main__":
    main()
