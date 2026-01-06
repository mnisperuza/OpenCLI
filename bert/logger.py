"""
Bert Error Logger
══════════════════════════════════════════════════════════════════════════════
Logs errors to ~/.bert/errors.log for debugging and support.
You can share this file manually when reporting issues.

By Matias Nisperuza — 2026
══════════════════════════════════════════════════════════════════════════════
"""

import os
import sys
import json
import traceback
import platform
from pathlib import Path
from datetime import datetime


class BertLogger:
    """Simple error logger for Bert CLI"""
    
    def __init__(self):
        self.bert_dir = Path.home() / ".bert"
        self.log_file = self.bert_dir / "errors.log"
        self.max_log_size = 1024 * 1024  # 1MB max
        self._ensure_dir()
    
    def _ensure_dir(self):
        """Create .bert directory if needed"""
        try:
            self.bert_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
    
    def _rotate_if_needed(self):
        """Rotate log file if too large"""
        try:
            if self.log_file.exists() and self.log_file.stat().st_size > self.max_log_size:
                # Keep last 500KB
                content = self.log_file.read_text(encoding='utf-8', errors='ignore')
                self.log_file.write_text(content[-500000:], encoding='utf-8')
        except Exception:
            pass
    
    def _get_system_info(self):
        """Get basic system info for debugging"""
        try:
            return {
                "os": platform.system(),
                "os_version": platform.version(),
                "python": platform.python_version(),
                "arch": platform.machine(),
            }
        except Exception:
            return {"os": "unknown"}
    
    def log_error(self, error_type: str, error_msg: str, 
                  context: dict = None, include_traceback: bool = True):
        """
        Log an error to the error file.
        
        Args:
            error_type: Category of error (e.g., "ModelLoad", "Generation", "Import")
            error_msg: The error message
            context: Optional dict with additional context (model, quant, etc.)
            include_traceback: Whether to include full traceback
        """
        try:
            self._rotate_if_needed()
            
            entry = {
                "timestamp": datetime.now().isoformat(),
                "type": error_type,
                "message": str(error_msg)[:500],  # Limit message length
                "system": self._get_system_info(),
            }
            
            if context:
                entry["context"] = {k: str(v)[:100] for k, v in context.items()}
            
            if include_traceback:
                tb = traceback.format_exc()
                if tb and tb != "NoneType: None\n":
                    entry["traceback"] = tb[-2000:]  # Last 2000 chars of traceback
            
            # Write to log file
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write("\n" + "=" * 60 + "\n")
                f.write(f"[{entry['timestamp']}] {error_type}\n")
                f.write(f"Message: {entry['message']}\n")
                
                if context:
                    f.write(f"Context: {json.dumps(entry.get('context', {}))}\n")
                
                f.write(f"System: {entry['system']['os']} / Python {entry['system']['python']}\n")
                
                if 'traceback' in entry:
                    f.write(f"\nTraceback:\n{entry['traceback']}\n")
                
                f.write("=" * 60 + "\n")
            
            return True
            
        except Exception:
            # Silently fail - logging should never crash the app
            return False
    
    def log_crash(self, exc_type, exc_value, exc_tb):
        """Log unhandled exceptions - can be used as sys.excepthook"""
        try:
            tb_str = ''.join(traceback.format_exception(exc_type, exc_value, exc_tb))
            self.log_error(
                "Crash",
                str(exc_value),
                context={"exception_type": str(exc_type.__name__)},
                include_traceback=False  # We have it already
            )
            
            # Also write the full traceback
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(f"\nFull Traceback:\n{tb_str}\n")
        except Exception:
            pass
    
    def get_log_path(self) -> str:
        """Get the path to the log file"""
        return str(self.log_file)
    
    def get_recent_errors(self, n: int = 5) -> str:
        """Get the last n errors from the log"""
        try:
            if not self.log_file.exists():
                return "No errors logged."
            
            content = self.log_file.read_text(encoding='utf-8', errors='ignore')
            entries = content.split("=" * 60)
            
            # Get last n non-empty entries
            recent = [e.strip() for e in entries if e.strip()][-n:]
            
            if not recent:
                return "No errors logged."
            
            return "\n---\n".join(recent)
            
        except Exception:
            return "Could not read error log."
    
    def clear_log(self):
        """Clear the error log"""
        try:
            if self.log_file.exists():
                self.log_file.unlink()
            return True
        except Exception:
            return False


# Global logger instance
_logger = None


def get_logger() -> BertLogger:
    """Get or create the global logger instance"""
    global _logger
    if _logger is None:
        _logger = BertLogger()
    return _logger


def log_error(error_type: str, error_msg: str, context: dict = None):
    """Convenience function to log an error"""
    return get_logger().log_error(error_type, error_msg, context)


def setup_crash_handler():
    """Set up global crash handler"""
    logger = get_logger()
    
    def crash_handler(exc_type, exc_value, exc_tb):
        logger.log_crash(exc_type, exc_value, exc_tb)
        # Call default handler
        sys.__excepthook__(exc_type, exc_value, exc_tb)
    
    sys.excepthook = crash_handler
