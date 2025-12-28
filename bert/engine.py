"""
Bert Core Engine
══════════════════════════════════════════════════════════════════════════════
Models:
  - Bert Nano:  Qwen2.5-0.5B-Instruct (fastest)
  - Bert Mini:  Qwen2.5-1.5B-Instruct
  - Bert 1:     Qwen3-1.7B
  - Bert Max:   Qwen3-4B (most capable)
  - Bert Coder: Qwen2.5-Coder-1.5B-Instruct

By Biwa — 2025
══════════════════════════════════════════════════════════════════════════════
"""

import os
import sys
import json
import gc
import re
import warnings
from pathlib import Path
from datetime import datetime

# ═══════════════════════════════════════════════════════════════════════════════
# SUPPRESS ALL WARNINGS
# ═══════════════════════════════════════════════════════════════════════════════

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['PYTHONWARNINGS'] = 'ignore'
os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'OFF'
os.environ['TORCH_SHOW_CPP_STACKTRACES'] = '0'

warnings.filterwarnings('ignore')

import logging
logging.getLogger().setLevel(logging.ERROR)
for name in ['torch', 'transformers', 'accelerate', 'bitsandbytes', 
             'torch.distributed', 'torch.distributed.elastic']:
    try:
        logging.getLogger(name).setLevel(logging.CRITICAL)
        logging.getLogger(name).disabled = True
    except:
        pass

# ═══════════════════════════════════════════════════════════════════════════════
# ML IMPORTS
# ═══════════════════════════════════════════════════════════════════════════════

import platform

ML_AVAILABLE = False
BNB_AVAILABLE = False
torch = None
IS_WINDOWS = platform.system() == "Windows"

try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        from transformers import logging as tf_logging
        tf_logging.set_verbosity_error()
        ML_AVAILABLE = True
        
        # Check if bitsandbytes is available (mainly Linux)
        try:
            import bitsandbytes
            BNB_AVAILABLE = True
        except ImportError:
            BNB_AVAILABLE = False
            
except ImportError:
    pass


# ═══════════════════════════════════════════════════════════════════════════════
# CUSTOM SYSTEM PROMPTS PER MODEL
# ═══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPTS = {
    "nano": """You are Bert Nano, a always reliable a fast and friendly AI assistant by Biwa.
Keep responses super short and helpful. Be direct and conversational.
For greetings, just say hi back briefly.""",

    "mini": """You are Bert Mini, a always reliable , balanced AI assistant by Biwa.
You provide helpful, clear responses with good detail when needed.
Be friendly and conversational. Match your response length to the question.""",

    "bert": """You are Bert, the always reliable flagship AI assistant by Biwa.
You are knowledgeable, thoughtful, and articulate.
Provide well-reasoned responses. Be warm but professional.
Take time to explain complex topics clearly.""",

    "max": """You are Bert Max, the always reliable most capable AI assistant by Biwa.
You excel at complex reasoning, analysis, and detailed explanations.
You are thorough, insightful, and can handle nuanced topics.
Provide comprehensive answers while remaining clear and organized.""",

    "coder": """You are Bert Coder, a always reliable, specialized coding assistant by Biwa.
You excel at programming, debugging, and technical explanations.
Write clean, well-commented code. Explain your solutions.
Be precise and technical. Use code blocks for examples.""",
}

# Fallback for unknown models
DEFAULT_PROMPT = "You are Bert,a always reliable helpful AI assistant. Be concise and direct."


# ═══════════════════════════════════════════════════════════════════════════════
# MEMORY CLEANUP
# ═══════════════════════════════════════════════════════════════════════════════

def cleanup_memory():
    """Clean GPU/CPU memory"""
    try:
        if torch and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        gc.collect()
        gc.collect()
    except:
        pass


# ═══════════════════════════════════════════════════════════════════════════════
# SHARED MEMORY
# ═══════════════════════════════════════════════════════════════════════════════

class SharedMemory:
    def __init__(self, name, max_turns=50):
        self.name = name
        self.max_turns = max_turns
        self.file = Path.home() / ".bert" / f"memory_{name}.json"
        self.history = []
        self._load()
    
    def _load(self):
        try:
            if self.file.exists():
                with open(self.file, 'r') as f:
                    data = json.load(f)
                    self.history = data.get('history', [])[-self.max_turns:]
        except:
            self.history = []
    
    def _save(self):
        try:
            self.file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.file, 'w') as f:
                json.dump({'history': self.history}, f)
        except:
            pass
    
    def add(self, user, assistant):
        self.history.append({'user': user, 'assistant': assistant})
        if len(self.history) > self.max_turns:
            self.history = self.history[-self.max_turns:]
        self._save()
    
    def get(self, n=5):
        return self.history[-n:] if self.history else []
    
    def clear(self):
        self.history = []
        self._save()


# ═══════════════════════════════════════════════════════════════════════════════
# BERT ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class BertEngine:
    
    # Model definitions
    MODELS = {
        "nano": {
            "path": "Qwen/Qwen2.5-0.5B-Instruct",
            "name": "Bert Nano",
            "family": "nano",
            "max_tokens": 2000,
            "temp": 0.85,
        },
        "mini": {
            "path": "Qwen/Qwen2.5-1.5B-Instruct",
            "name": "Bert Mini", 
            "family": "mini",
            "max_tokens": 3500,
            "temp": 0.8,
        },
        "bert": {
            "path": "Qwen/Qwen3-1.7B",
            "name": "Bert 1",
            "family": "bert",
            "max_tokens": 6200,
            "temp": 0.8,
        },
        "max": {
            "path": "Qwen/Qwen3-4B",
            "name": "Bert Max",
            "family": "max",
            "max_tokens": 12000,
            "temp": 0.6,
        },
        "coder": {
            "path": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
            "name": "Bert Coder",
            "family": "coder",
            "max_tokens": 7000,
            "temp": 0.7,
        },
    }
    
    def __init__(self):
        self.device = "cuda" if ML_AVAILABLE and torch and torch.cuda.is_available() else "cpu"
        self.model = None
        self.tokenizer = None
        self.current_mode = "nano"
        self.current_quant = "int4"
        
        # Shared memory per family
        self.memories = {
            "nano": SharedMemory("nano"),
            "mini": SharedMemory("mini"),
            "bert": SharedMemory("bert"),
            "max": SharedMemory("max"),
            "coder": SharedMemory("coder"),
        }
        self.memory = self.memories["nano"]
    
    def load_model(self, mode=None, quant=None):
        """Load model with specified quantization (cross-platform)"""
        if not ML_AVAILABLE:
            print("❌ ML libraries not available")
            return False
        
        if mode:
            self.current_mode = mode
        if quant:
            self.current_quant = quant
        
        config = self.MODELS.get(self.current_mode, self.MODELS["nano"])
        model_path = config["path"]
        q = self.current_quant.lower()
        
        # Switch memory
        self.memory = self.memories.get(self.current_mode, self.memories["nano"])
        
        # Cleanup
        if self.model:
            del self.model
            self.model = None
        if self.tokenizer:
            del self.tokenizer
            self.tokenizer = None
        cleanup_memory()
        
        # Cross-platform check: bitsandbytes not available on Windows by default
        if q in ["int1", "int2", "int4", "int6", "int8"] and not BNB_AVAILABLE:
            if IS_WINDOWS:
                print(f"⚠️  {q.upper()} quantization requires bitsandbytes (not available on Windows)")
                print(f"   Falling back to FP16...")
                q = "fp16"
                self.current_quant = "fp16"
            else:
                print(f"⚠️  bitsandbytes not installed. Install with: pip install bitsandbytes")
                print(f"   Falling back to FP16...")
                q = "fp16"
                self.current_quant = "fp16"
        
        print(f"📦 Loading {config['name']} with {q.upper()}...")
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path, trust_remote_code=True
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Quantization config
            bnb_config = None
            
            if self.device == "cuda":
                if BNB_AVAILABLE and q in ["int1", "int2", "int4", "4bit"]:
                    bnb_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4",
                    )
                elif BNB_AVAILABLE and q in ["int6", "int8", "8bit"]:
                    bnb_config = BitsAndBytesConfig(load_in_8bit=True)
                elif q in ["fp16", "float16"]:
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_path,
                        torch_dtype=torch.float16,
                        device_map="auto",
                        trust_remote_code=True,
                    )
                    print(f"✓ {config['name']} loaded ({q.upper()})")
                    return True
                else:  # fp32 or fallback
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_path,
                        torch_dtype=torch.float32,
                        device_map="auto",
                        trust_remote_code=True,
                    )
                    print(f"✓ {config['name']} loaded (FP32)")
                    return True
                
                # Load with quantization config
                if bnb_config:
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_path,
                        quantization_config=bnb_config,
                        device_map="auto",
                        trust_remote_code=True,
                    )
            else:
                # CPU - always use float32
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.float32,
                    trust_remote_code=True,
                )
            
            print(f"✓ {config['name']} loaded ({q.upper()})")
            return True
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return False
    
    def _clean_response(self, text, user_msg):
        """
        Clean model output - IMPROVED think tag handling.
        The key is to extract content AFTER </think> when present.
        """
        original = text
        
        # ═══════════════════════════════════════════════════════════════════════
        # STEP 1: Handle <think>...</think> blocks properly
        # ═══════════════════════════════════════════════════════════════════════
        
        # Check if there's a closing </think> tag - extract everything AFTER it
        think_end_patterns = ['</think>', '</thinking>', '</thought>']
        
        for pattern in think_end_patterns:
            if pattern in text.lower():
                # Find the LAST occurrence of the closing tag
                idx = text.lower().rfind(pattern)
                text = text[idx + len(pattern):]
                break
        
        # Also remove any remaining opening tags without closing
        text = re.sub(r'<think>.*', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<thinking>.*', '', text, flags=re.DOTALL | re.IGNORECASE)
        
        # ═══════════════════════════════════════════════════════════════════════
        # STEP 2: Remove thinking-style content (bold headers, analysis blocks)
        # ═══════════════════════════════════════════════════════════════════════
        
        # Remove **Bold Analysis Headers**
        text = re.sub(r'\*\*[A-Z][^*]{3,60}\*\*\s*\n?', '', text)
        
        # Remove lines that look like internal thinking
        thinking_patterns = [
            r'^My (current |)thinking.*$',
            r'^I\'m (now |currently |)focusing.*$',
            r'^(Let me |I\'ll |I will )think.*$',
            r'^(Analyzing|Processing|Considering).*$',
            r'^The (goal|objective|aim) is to.*$',
        ]
        for pattern in thinking_patterns:
            text = re.sub(pattern, '', text, flags=re.MULTILINE | re.IGNORECASE)
        
        # ═══════════════════════════════════════════════════════════════════════
        # STEP 3: Find actual response start
        # ═══════════════════════════════════════════════════════════════════════
        
        # If user message appears, get content after it
        if user_msg and len(user_msg) > 3:
            user_lower = user_msg.lower().strip()
            if user_lower in text.lower():
                idx = text.lower().rfind(user_lower)
                text = text[idx + len(user_msg):]
        
        # Remove role prefixes
        text = re.sub(r'^[\s\n]*(system|user|assistant|bert|human|AI)[\s:]+', '', text, flags=re.IGNORECASE)
        
        # ═══════════════════════════════════════════════════════════════════════
        # STEP 4: Clean up formatting
        # ═══════════════════════════════════════════════════════════════════════
        
        # Remove stray XML-like tags
        text = re.sub(r'<[a-z_]+>', '', text, flags=re.IGNORECASE)
        text = re.sub(r'</[a-z_]+>', '', text, flags=re.IGNORECASE)
        
        # Clean excessive whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'^\s+', '', text)  # Leading whitespace
        text = text.strip()
        
        # ═══════════════════════════════════════════════════════════════════════
        # STEP 5: Validate - if we cleaned too much, try to salvage
        # ═══════════════════════════════════════════════════════════════════════
        
        if len(text) < 2:
            # Try to find any sentence-like content in original
            sentences = re.findall(r'[A-Z][^.!?]*[.!?]', original)
            # Filter out thinking sentences
            good_sentences = [s for s in sentences if not any(
                kw in s.lower() for kw in ['thinking', 'analyzing', 'my current', 'focusing', 'objective']
            )]
            if good_sentences:
                text = ' '.join(good_sentences[-3:])  # Take last few sentences
        
        return text.strip()
    
    def generate(self, user_msg):
        """Generate response with custom system prompt per model"""
        if not self.model:
            return "Model not loaded."
        
        try:
            config = self.MODELS.get(self.current_mode, self.MODELS["nano"])
            msg_lower = user_msg.lower().strip()
            words = len(user_msg.split())
            
            # Get custom system prompt for this model
            system_prompt = SYSTEM_PROMPTS.get(self.current_mode, DEFAULT_PROMPT)
            
            # Quick patterns - for fast greetings
            quick = ['hi', 'hey', 'hello', 'hola', 'sup', 'yo', 'thanks', 
                    'thank you', 'bye', 'ok', 'okay', 'cool', 'gm', 'gn']
            is_quick = any(msg_lower.rstrip('!?., ') == p for p in quick)
            
            # Token limits based on query + model
            base_tokens = config["max_tokens"]
            if is_quick:
                max_tokens = 40
            elif words <= 3:
                max_tokens = min(100, base_tokens)
            elif words <= 8:
                max_tokens = min(300, base_tokens)
            elif words <= 20:
                max_tokens = min(800, base_tokens)
            else:
                max_tokens = base_tokens
            
            # Build messages
            messages = [{"role": "system", "content": system_prompt}]
            
            # Add memory (skip for quick queries)
            if not is_quick:
                for turn in self.memory.get(3):
                    messages.append({"role": "user", "content": turn["user"]})
                    messages.append({"role": "assistant", "content": turn["assistant"]})
            
            messages.append({"role": "user", "content": user_msg})
            
            # Format prompt
            if hasattr(self.tokenizer, 'apply_chat_template') and self.tokenizer.chat_template:
                prompt = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            else:
                # Fallback format
                prompt = f"{system_prompt}\n\nUser: {user_msg}\nAssistant:"
            
            # Generate
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=config["temp"],
                    top_p=0.9,
                    repetition_penalty=1.1,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    use_cache=True,
                )
            
            full = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = self._clean_response(full, user_msg)
            
            # Cleanup
            del inputs, outputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Validate
            if len(response) < 2:
                response = "Hey! How can I help?"
            
            # Save to memory
            self.memory.add(user_msg, response)
            
            return response
            
        except Exception as e:
            return f"Error: {e}"


# ═══════════════════════════════════════════════════════════════════════════════
# SINGLETON
# ═══════════════════════════════════════════════════════════════════════════════

_engine = None

def get_engine():
    global _engine
    if _engine is None:
        _engine = BertEngine()
    return _engine
