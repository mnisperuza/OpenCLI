"""
BERT CLI v1.0.0b (Beta)
══════════════════════════════════════════════════════════════════════════════
✅ 4 main models + 1 coder (all Qwen)
✅ Gradient system: Sage, Teal, Sage-Olive, Orange-Coral, Silver
✅ Animated BERT CLI banner
✅ Fast braille spinner
✅ Quantization picker
✅ Error logging to ~/.bert/errors.log
✅ Uninstall command: bert --del

Models:
  bert nano  → Qwen2.5-0.5B-Instruct (default, fastest)
  bert mini  → Qwen2.5-1.5B-Instruct
  bert / bert 1 → Qwen3-1.7B
  bert max   → Qwen3-4B (most capable)
  bert coder → Qwen2.5-Coder-1.5B-Instruct

By Biwa Industries — 2025
══════════════════════════════════════════════════════════════════════════════
"""

import os
import sys
import time
import threading
from pathlib import Path
from datetime import datetime
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'

# Add script directory to path
_script_dir = Path(__file__).parent.resolve()
if str(_script_dir) not in sys.path:
    sys.path.insert(0, str(_script_dir))


# ═══════════════════════════════════════════════════════════════════════════════
# ERROR LOGGER
# ═══════════════════════════════════════════════════════════════════════════════

LOGGER_AVAILABLE = False
try:
    from bert.logger import get_logger, log_error, setup_crash_handler
    LOGGER_AVAILABLE = True
    setup_crash_handler()
except ImportError:
    try:
        from logger import get_logger, log_error, setup_crash_handler
        LOGGER_AVAILABLE = True
        setup_crash_handler()
    except ImportError:
        # Define dummy functions if logger not available
        def log_error(*args, **kwargs):
            pass
        def get_logger():
            return None


# ═══════════════════════════════════════════════════════════════════════════════
# ENGINE IMPORT
# ═══════════════════════════════════════════════════════════════════════════════

ENGINE_AVAILABLE = False
try:
    from engine import get_engine, BertEngine
    ENGINE_AVAILABLE = True
except ImportError:
    pass

CODE_AVAILABLE = False
try:
    from code import get_code_session, end_code_session
    CODE_AVAILABLE = True
except ImportError:
    pass


# ═══════════════════════════════════════════════════════════════════════════════
# COLORS
# ═══════════════════════════════════════════════════════════════════════════════

class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
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
    
    @staticmethod
    def rgb(r, g, b):
        return f"\033[38;2;{max(0,min(255,int(r)))};{max(0,min(255,int(g)))};{max(0,min(255,int(b)))}m"


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
# GRADIENTS (per model family)
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
    # Bert 1: Sage-Olive (earthy, natural)
    "bert": [
        (107, 142, 85), (119, 156, 95), (134, 169, 108), (148, 182, 120),
        (162, 195, 132), (148, 182, 120), (134, 169, 108), (119, 156, 95),
    ],
    # Bert Max: Orange-Coral (Claude style - warm, powerful)
    "max": [
        (204, 119, 77), (218, 130, 85), (232, 145, 95), (245, 160, 107),
        (255, 175, 120), (245, 160, 107), (232, 145, 95), (218, 130, 85),
    ],
    # Bert Coder: Silver (clean, professional)
    "coder": [
        (140, 150, 160), (160, 170, 180), (180, 190, 200), (200, 210, 220),
        (220, 225, 230), (200, 210, 220), (180, 190, 200), (160, 170, 180),
    ],
}

# Primary colors for each family
FAMILY_COLORS = {
    "nano": (150, 195, 165),   # Sage
    "mini": (75, 205, 170),    # Teal
    "bert": (148, 182, 120),   # Sage-Olive
    "max": (245, 160, 107),    # Orange-Coral (Claude)
    "coder": (200, 210, 220),  # Silver
}


# ═══════════════════════════════════════════════════════════════════════════════
# SPINNER
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


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

class BertCLI:
    
    VERSION = "1.0.0b"
    VERSION_NAME = "Beta"
    
    # Model info
    MODELS = {
        "nano": ("Bert Nano", "Qwen2.5-0.5B", "nano"),
        "mini": ("Bert Mini", "Qwen2.5-1.5B", "mini"),
        "bert": ("Bert 1", "Qwen3-1.7B", "bert"),
        "1": ("Bert 1", "Qwen3-1.7B", "bert"), 
        "max": ("Bert Max", "Qwen3-4B", "max"),
        "coder": ("Bert Coder", "Qwen2.5-Coder-1.5B", "coder"),
    }
    
    def __init__(self):
        self.current_dir = Path.cwd()
        self.engine = None
        self.mode = "nano"  # Default
        self.quant = "int4"
        self.debug = False
        self.code_mode = False
        self.code_session = None
    
    def clear(self):
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def banner(self):
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
        
        # Animate the banner!
        animate_banner(centered_lines, duration=1.8)
        
        print()
        print(f"{Colors.DIM}{'by Biwa Industries — 2025'.center(width)}{Colors.RESET}")
        print(f"{Colors.DIM}{f'Version {self.VERSION} ({self.VERSION_NAME})'.center(width)}{Colors.RESET}")
        print()
    
    def pick_quant(self, model_name, family):
        """Quantization picker with polished UI"""
        r, g, b = FAMILY_COLORS.get(family, FAMILY_COLORS["nano"])
        color = f"\033[38;2;{r};{g};{b}m"
        
        print(f"\n{color}┌{'─' * 48}┐{Colors.RESET}")
        print(f"{color}│{Colors.RESET} {Colors.BOLD}🎛️  Select Quantization for {model_name}{Colors.RESET}")
        print(f"{color}└{'─' * 48}┘{Colors.RESET}\n")
        
        options = [
            ("1", "int2", "INT2", "Most compressed (3GB VRAM)"),
            ("2", "int4", "INT4", "Balanced ⭐"),
            ("3", "int8", "INT8", "High quality (6GB+)"),
            ("4", "fp16", "FP16", "Best quality (8GB+)"),
        ]
        
        for num, key, label, desc in options:
            if key == "int4":
                print(f"  {Colors.GREEN}[{num}] {label}{Colors.RESET} — {desc}")
            else:
                print(f"  {Colors.DIM}[{num}] {label} — {desc}{Colors.RESET}")
        
        print(f"\n  {Colors.DIM}Press Enter for INT4{Colors.RESET}")
        
        try:
            choice = input(f"  {color}Your choice (1-4):{Colors.RESET} ").strip()
            
            quant_map = {"1": "int2", "2": "int4", "3": "int8", "4": "fp16", "": "int4"}
            selected = quant_map.get(choice, "int4")
            
            print(f"\n  {Colors.GREEN}✓ Selected {selected.upper()}{Colors.RESET}")
            return selected
        except:
            return "int4"
    
    def switch_model(self, model_key):
        """Switch to a different model"""
        key = model_key.lower().strip()
        
        if key not in self.MODELS:
            print(f"\n{Colors.RED}Unknown model: {model_key}{Colors.RESET}")
            print(f"{Colors.BLACK}Available: nano, mini, bert/1, max, coder{Colors.RESET}\n")
            return
        
        name, desc, family = self.MODELS[key]
        
        # Shimmer the model name
        print()
        shimmer_text(f"→ {name}", family, duration=1.0)
        
        # Pick quantization
        quant = self.pick_quant(name, family)
        self.quant = quant
        self.mode = key if key != "1" else "bert"
        
        # Load with spinner
        print()
        stop_event = threading.Event()
        
        thread = threading.Thread(
            target=loading_spinner,
            args=(f"Loading {name}...", family, stop_event)
        )
        thread.daemon = True
        thread.start()
        
        if self.engine:
            success = self.engine.load_model(mode=self.mode, quant=quant)
        else:
            success = False
        
        stop_event.set()
        thread.join(timeout=1.0)
        
        if success:
            print()
            shimmer_text(f"✓ {name} ready!", family, duration=0.8)
        else:
            print(f"\n{Colors.RED}✗ Failed to load {name}{Colors.RESET}")
        
        print()
    
    def change_quant(self, quant_arg):
        """Change quantization on current model"""
        quant_map = {
            "1": "int1", "int1": "int1", "2": "int2", "int2": "int2",
            "4": "int4", "int4": "int4", "6": "int6", "int6": "int6",
            "8": "int8", "int8": "int8", "16": "fp16", "fp16": "fp16",
            "32": "fp32", "fp32": "fp32",
        }
        
        q = quant_arg.lower().replace("-", "").replace("_", "")
        
        if q not in quant_map:
            print(f"\n{Colors.RED}Unknown: {quant_arg}{Colors.RESET}")
            print(f"{Colors.BLACK}Use: int2, int4, int8, fp16, fp32{Colors.RESET}\n")
            return
        
        new_quant = quant_map[q]
        
        if new_quant == self.quant:
            print(f"\n{Colors.BLACK}Already using {new_quant.upper()}{Colors.RESET}\n")
            return
        
        name, _, family = self.MODELS.get(self.mode, self.MODELS["nano"])
        
        print()
        shimmer_text(f"→ {name} @ {new_quant.upper()}", family, duration=0.8)
        
        self.quant = new_quant
        
        # Reload
        print()
        stop_event = threading.Event()
        
        thread = threading.Thread(
            target=loading_spinner,
            args=(f"Reloading with {new_quant.upper()}...", family, stop_event)
        )
        thread.daemon = True
        thread.start()
        
        if self.engine:
            success = self.engine.load_model(mode=self.mode, quant=new_quant)
        else:
            success = False
        
        stop_event.set()
        thread.join(timeout=1.0)
        
        if success:
            print()
            shimmer_text(f"✓ Now running {new_quant.upper()}!", family, duration=0.6)
        else:
            print(f"\n{Colors.RED}✗ Failed{Colors.RESET}")
        
        print()
    
    def toggle_coder(self):
        """Toggle coder mode"""
        if self.mode == "coder":
            # Switch back to nano
            self.switch_model("nano")
        else:
            self.switch_model("coder")
    
    def handle_command(self, inp):
        """Handle commands. Returns: None=not a command, True=continue, False=exit"""
        lower = inp.lower().strip()
        
        # Exit
        if lower in ["/*exit", "/*quit", "/*q", "exit", "quit"]:
            print(f"\n{Colors.DIM}Goodbye! 👋{Colors.RESET}\n")
            return False
        
        # Clear
        if lower == "/*clear":
            self.clear()
            self.banner()
            return True
        
        # Help
        if lower == "/*help":
            self.show_help()
            return True
        
        # Status
        if lower == "/*status":
            self.show_status()
            return True
        
        # Debug
        if lower == "/*debug":
            self.debug = not self.debug
            print(f"{Colors.GREEN}✓ Debug: {'ON' if self.debug else 'OFF'}{Colors.RESET}\n")
            return True
        
        # Model switching: "bert nano", "bert mini", "bert", "bert 1", "bert max", "bert coder"
        if lower.startswith("bert "):
            arg = lower[5:].strip()
            
            # Quantization change: "bert int2", "bert fp16"
            if arg.startswith("int") or arg.startswith("fp") or arg.isdigit():
                self.change_quant(arg)
                return True
            
            # Model switch
            if arg in ["nano", "mini", "1", "max", "coder"]:
                self.switch_model(arg)
                return True
            
            print(f"\n{Colors.RED}Unknown: bert {arg}{Colors.RESET}")
            print(f"{Colors.BLACK}Models: nano, mini, 1, max, coder{Colors.RESET}")
            print(f"{Colors.BLACK}Quant: int2, int4, int8, fp16{Colors.RESET}\n")
            return True
        
        # Just "bert" = bert 1
        if lower == "bert":
            self.switch_model("bert")
            return True
        
        # Navigation
        if lower.startswith("/*cd "):
            path = inp[5:].strip()
            try:
                new_path = Path(path).expanduser().resolve()
                if new_path.is_dir():
                    self.current_dir = new_path
                    os.chdir(new_path)
                    print(f"{Colors.GREEN}✓ {new_path}{Colors.RESET}\n")
                else:
                    print(f"{Colors.RED}Not found: {path}{Colors.RESET}\n")
            except Exception as e:
                print(f"{Colors.RED}Error: {e}{Colors.RESET}\n")
            return True
        
        if lower == "/*ls":
            self.list_dir()
            return True
        
        return None  # Not a command
    
    def list_dir(self):
        try:
            items = sorted(self.current_dir.iterdir())
            dirs = [i for i in items if i.is_dir() and not i.name.startswith('.')]
            files = [i for i in items if i.is_file() and not i.name.startswith('.')]
            
            print(f"\n{Colors.DIM}{self.current_dir}{Colors.RESET}\n")
            
            for d in dirs[:15]:
                print(f"  {Colors.BLUE}📁 {d.name}/{Colors.RESET}")
            for f in files[:15]:
                size = f.stat().st_size
                s = f"{size}B" if size < 1024 else f"{size/1024:.1f}K"
                print(f"  {Colors.BLACK}📄 {f.name} ({s}){Colors.RESET}")
            
            if len(dirs) + len(files) > 30:
                print(f"\n  {Colors.BLACK}... and more{Colors.RESET}")
            print()
        except Exception as e:
            print(f"{Colors.RED}Error: {e}{Colors.RESET}\n")
    
    def show_help(self):
        r_nano = FAMILY_COLORS["nano"]
        r_mini = FAMILY_COLORS["mini"]
        r_bert = FAMILY_COLORS["bert"]
        r_max = FAMILY_COLORS["max"]
        r_coder = FAMILY_COLORS["coder"]
        
        print(f"\n{Colors.DIM}{'═' * 55}{Colors.RESET}")
        print(f"  {Colors.BOLD}BERT CLI — Command Reference{Colors.RESET}")
        print(f"{Colors.DIM}{'═' * 55}{Colors.RESET}\n")
        
        print(f"  {Colors.BOLD}Models:{Colors.RESET}")
        print(f"    \033[38;2;{r_nano[0]};{r_nano[1]};{r_nano[2]}m●\033[0m bert nano     Qwen2.5-0.5B  {Colors.DIM}(fastest){Colors.RESET}")
        print(f"    \033[38;2;{r_mini[0]};{r_mini[1]};{r_mini[2]}m●\033[0m bert mini     Qwen2.5-1.5B  {Colors.DIM}(balanced){Colors.RESET}")
        print(f"    \033[38;2;{r_bert[0]};{r_bert[1]};{r_bert[2]}m●\033[0m bert          Qwen3-1.7B    {Colors.DIM}(flagship){Colors.RESET}")
        print(f"    \033[38;2;{r_max[0]};{r_max[1]};{r_max[2]}m●\033[0m bert max      Qwen3-4B      {Colors.DIM}(smartest){Colors.RESET}")
        print(f"    \033[38;2;{r_coder[0]};{r_coder[1]};{r_coder[2]}m●\033[0m bert coder    Qwen2.5-Coder {Colors.DIM}(code){Colors.RESET}\n")
        
        print(f"  {Colors.BOLD}Quantization:{Colors.RESET}")
        print(f"    bert int2     {Colors.DIM}Most compressed (3GB){Colors.RESET}")
        print(f"    bert int4     {Colors.DIM}Balanced ⭐{Colors.RESET}")
        print(f"    bert int8     {Colors.DIM}High quality (6GB+){Colors.RESET}")
        print(f"    bert fp16     {Colors.DIM}Best quality (8GB+){Colors.RESET}\n")
        
        print(f"  {Colors.BOLD}Navigation:{Colors.RESET}")
        print(f"    /*cd <path>   {Colors.DIM}Change directory{Colors.RESET}")
        print(f"    /*ls          {Colors.DIM}List files{Colors.RESET}\n")
        
        print(f"  {Colors.BOLD}Commands:{Colors.RESET}")
        print(f"    /*clear       {Colors.DIM}Clear screen{Colors.RESET}")
        print(f"    /*status      {Colors.DIM}Show status{Colors.RESET}")
        print(f"    /*help        {Colors.DIM}This help{Colors.RESET}")
        print(f"    /*exit        {Colors.DIM}Exit Bert{Colors.RESET}\n")
        
        print(f"{Colors.DIM}{'═' * 55}{Colors.RESET}\n")
    
    def show_status(self):
        name, desc, family = self.MODELS.get(self.mode, self.MODELS["nano"])
        r, g, b = FAMILY_COLORS.get(family, FAMILY_COLORS["nano"])
        
        print(f"\n{Colors.DIM}{'─' * 45}{Colors.RESET}")
        print(f"  {Colors.BOLD}Status{Colors.RESET}")
        print(f"{Colors.DIM}{'─' * 45}{Colors.RESET}")
        print(f"  Model:   \033[38;2;{r};{g};{b}m● {name}\033[0m")
        print(f"  Base:    {Colors.DIM}{desc}{Colors.RESET}")
        print(f"  Quant:   {self.quant.upper()}")
        print(f"  Engine:  {'✓ Loaded' if self.engine and self.engine.model else '✗ Not loaded'}")
        print(f"  Dir:     {Colors.DIM}{self.current_dir}{Colors.RESET}")
        print(f"{Colors.DIM}{'─' * 45}{Colors.RESET}\n")
    
    def context_bar(self):
        """Show context bar after each response"""
        name, _, family = self.MODELS.get(self.mode, self.MODELS["nano"])
        r, g, b = FAMILY_COLORS.get(family, FAMILY_COLORS["nano"])
        
        left = f"\033[38;2;{r};{g};{b}m{name}\033[0m [{self.quant.upper()}]"
        right = str(self.current_dir).replace(str(Path.home()), '~')
        
        width = term_width()
        # Approximate visible length (strip ANSI)
        left_visible = len(name) + len(self.quant) + 4
        spacing = max(1, width - left_visible - len(right) - 2)
        
        print(f"{left}{' ' * spacing}{Colors.DIM}{right}{Colors.RESET}")
    
    def prompt(self):
        return f"{Colors.DIM}>{Colors.RESET} "
    
    def query(self, user_input):
        """Send query to model"""
        _, _, family = self.MODELS.get(self.mode, self.MODELS["nano"])
        
        # Show spinner while generating
        stop_event = threading.Event()
        
        thread = threading.Thread(
            target=loading_spinner,
            args=("", family, stop_event, 0.1)
        )
        thread.daemon = True
        thread.start()
        
        response = None
        if self.engine:
            try:
                response = self.engine.generate(user_input)
            except Exception as e:
                response = f"Error: {e}"
                # Log the error
                log_error("Generation", str(e), {
                    "model": self.mode,
                    "quant": self.quant,
                    "input_length": len(user_input)
                })
                if self.debug:
                    import traceback
                    traceback.print_exc()
        else:
            time.sleep(0.5)
            response = "[Demo] Engine not loaded"
        
        stop_event.set()
        thread.join(timeout=1.0)
        
        # Print response
        r, g, b = FAMILY_COLORS.get(family, FAMILY_COLORS["nano"])
        print(f"\033[38;2;{r};{g};{b}m{response}\033[0m\n")
    
    def init_engine(self):
        """Initialize engine with quantization picker"""
        if not ENGINE_AVAILABLE:
            print(f"{Colors.BLACK}Demo mode (engine not available){Colors.RESET}\n")
            log_error("Import", "Engine not available", {"ENGINE_AVAILABLE": False})
            return
        
        # Pick initial quantization
        name, _, family = self.MODELS["nano"]
        quant = self.pick_quant(name, family)
        self.quant = quant
        
        # Load with spinner
        print()
        stop_event = threading.Event()
        
        thread = threading.Thread(
            target=loading_spinner,
            args=("Initializing Bert Nano...", "nano", stop_event)
        )
        thread.daemon = True
        thread.start()
        
        try:
            self.engine = get_engine()
            self.engine.load_model(mode="nano", quant=quant)
        except Exception as e:
            log_error("ModelLoad", str(e), {"model": "nano", "quant": quant})
            if self.debug:
                print(f"\n{Colors.RED}Error: {e}{Colors.RESET}")
        
        stop_event.set()
        thread.join(timeout=1.0)
        print()
    
    def run(self):
        self.clear()
        self.banner()
        self.init_engine()
        
        # Tips with colored model names
        r_nano = FAMILY_COLORS["nano"]
        r_max = FAMILY_COLORS["max"]
        r_coder = FAMILY_COLORS["coder"]
        
        print(f"{Colors.DIM}┌─────────────────────────────────────────────────────┐{Colors.RESET}")
        print(f"{Colors.DIM}│{Colors.RESET} Models: \033[38;2;{r_nano[0]};{r_nano[1]};{r_nano[2]}mnano\033[0m {Colors.DIM}•{Colors.RESET} mini {Colors.DIM}•{Colors.RESET} bert {Colors.DIM}•{Colors.RESET} \033[38;2;{r_max[0]};{r_max[1]};{r_max[2]}mmax\033[0m {Colors.DIM}•{Colors.RESET} \033[38;2;{r_coder[0]};{r_coder[1]};{r_coder[2]}mcoder\033[0m        {Colors.DIM}│{Colors.RESET}")
        print(f"{Colors.DIM}│{Colors.RESET} Quant:  bert int2 / int4 / int8 / fp16            {Colors.DIM}│{Colors.RESET}")
        print(f"{Colors.DIM}│{Colors.RESET} Help:   /*help                                    {Colors.DIM}│{Colors.RESET}")
        print(f"{Colors.DIM}└─────────────────────────────────────────────────────┘{Colors.RESET}\n")
        
        print(f"{Colors.DIM}{'─' * 60}{Colors.RESET}")
        
        while True:
            try:
                user_input = input(self.prompt()).strip()
                
                self.context_bar()
                
                if not user_input:
                    continue
                
                result = self.handle_command(user_input)
                
                if result is False:
                    break
                elif result is True:
                    continue
                
                # Query model
                self.query(user_input)
                
                print(f"{Colors.DIM}{'─' * 60}{Colors.RESET}")
                
            except KeyboardInterrupt:
                print(f"\n\n{Colors.DIM}Goodbye! 👋{Colors.RESET}\n")
                break
            except EOFError:
                print(f"\n\n{Colors.DIM}Goodbye! 👋{Colors.RESET}\n")
                break


# ═══════════════════════════════════════════════════════════════════════════════
# UNINSTALL FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

def uninstall_bert():
    """Remove Bert data directory (~/.bert)"""
    import shutil
    
    bert_dir = Path.home() / ".bert"
    
    print()
    print(f"{Colors.YELLOW}┌{'─' * 50}┐{Colors.RESET}")
    print(f"{Colors.YELLOW}│{Colors.RESET} {Colors.BOLD}⚠️  Uninstall Bert{Colors.RESET}")
    print(f"{Colors.YELLOW}└{'─' * 50}┘{Colors.RESET}")
    print()
    
    if not bert_dir.exists():
        print(f"{Colors.DIM}Nothing to remove. Bert data directory not found.{Colors.RESET}")
        print(f"{Colors.DIM}Location: {bert_dir}{Colors.RESET}\n")
        return
    
    # Show what will be deleted
    print(f"{Colors.DIM}This will remove:{Colors.RESET}")
    print(f"  • {bert_dir}")
    
    try:
        items = list(bert_dir.iterdir())
        for item in items[:10]:
            print(f"    └─ {item.name}")
        if len(items) > 10:
            print(f"    └─ ... and {len(items) - 10} more")
    except:
        pass
    
    print()
    print(f"{Colors.DIM}This includes: memory, config, error logs, backups{Colors.RESET}")
    print()
    
    try:
        confirm = input(f"{Colors.YELLOW}Are you sure? [y/N]:{Colors.RESET} ").strip().lower()
        
        if confirm in ['y', 'yes']:
            try:
                shutil.rmtree(bert_dir)
                print(f"\n{Colors.GREEN}✓ Bert data removed successfully.{Colors.RESET}")
                print(f"{Colors.DIM}To fully uninstall, also run: pip uninstall bert-cli{Colors.RESET}\n")
            except Exception as e:
                print(f"\n{Colors.RED}✗ Failed to remove: {e}{Colors.RESET}\n")
        else:
            print(f"\n{Colors.DIM}Cancelled.{Colors.RESET}\n")
    
    except (KeyboardInterrupt, EOFError):
        print(f"\n{Colors.DIM}Cancelled.{Colors.RESET}\n")


def show_version():
    """Show version info"""
    print(f"\n{Colors.DIM}Bert CLI{Colors.RESET}")
    print(f"Version: {BertCLI.VERSION} ({BertCLI.VERSION_NAME})")
    print(f"By Biwa Industries — 2025\n")


def show_info():
    """Show info about Bert"""
    print(f"""
{Colors.DIM}{'═' * 50}{Colors.RESET}
  {Colors.BOLD}BERT CLI{Colors.RESET}
  A calm, local AI assistant by Biwa Industries
{Colors.DIM}{'═' * 50}{Colors.RESET}

  Version:  {BertCLI.VERSION} ({BertCLI.VERSION_NAME})
  Models:   Qwen 2.5 & Qwen 3 family
  License:  Proprietary (Biwa Industries)
  
  GitHub:   github.com/mnisperuza/bert-cli
  Support:  contact: biwaindustries@gmail.com
  
{Colors.DIM}{'═' * 50}{Colors.RESET}
""")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    """Entry point - handles CLI args or starts interactive mode"""
    import sys
    
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        
        if arg in ['--del', '--delete', '--uninstall', '--remove']:
            uninstall_bert()
            return
        
        elif arg in ['--ver', '--version', '-v']:
            show_version()
            return
        
        elif arg in ['--info', '-i']:
            show_info()
            return
        
        elif arg in ['--help', '-h']:
            print(f"""
{Colors.BOLD}Bert CLI{Colors.RESET} — A calm, local AI assistant

{Colors.DIM}Usage:{Colors.RESET}
  bert              Start Bert (interactive mode)
  bert --ver        Show version
  bert --info       Show info about Bert
  bert --del        Remove Bert data (~/.bert)
  bert --help       Show this help

{Colors.DIM}In-session commands:{Colors.RESET}
  bert nano/mini/max/coder    Switch model
  bert int2/int4/int8/fp16    Change quantization
  /*help                      Show all commands
  /*exit                      Exit Bert
""")
            return
        
        else:
            print(f"{Colors.RED}Unknown option: {arg}{Colors.RESET}")
            print(f"{Colors.DIM}Use 'bert --help' for usage info{Colors.RESET}\n")
            return
    
    # No args - start interactive mode
    cli = BertCLI()
    cli.run()


if __name__ == "__main__":
    main()
