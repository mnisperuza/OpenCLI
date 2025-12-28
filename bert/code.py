"""
BERT Code Module — Beta
══════════════════════════════════════════════════════════════════════════════
"""

import os
import sys
import difflib
import subprocess
import shutil
import json
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Tuple, Any


# ═══════════════════════════════════════════════════════════════════════════════
# ANSI COLORS
# ═══════════════════════════════════════════════════════════════════════════════

class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GRAY = "\033[90m"
    
    SAGE = "\033[38;2;119;135;124m"
    LIGHT_SAGE = "\033[38;2;143;167;154m"


# ═══════════════════════════════════════════════════════════════════════════════
# OPERATION RESULT
# ═══════════════════════════════════════════════════════════════════════════════

class OperationResult:
    """Standardized result for all file operations"""
    
    def __init__(self, success: bool, message: str, data: Any = None):
        self.success = success
        self.message = message
        self.data = data
    
    def __bool__(self):
        return self.success


# ═══════════════════════════════════════════════════════════════════════════════
# BERT CODE SESSION
# ═══════════════════════════════════════════════════════════════════════════════

class BertCodeSession:
    """Code editing session with file safety, backups, and approval workflow."""
    
    MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB
    
    DANGEROUS_EXTENSIONS = {'.exe', '.dll', '.so', '.dylib', '.bin', '.sys'}
    
    TEXT_EXTENSIONS = {
        '.py', '.js', '.ts', '.jsx', '.tsx', '.html', '.css', '.scss',
        '.json', '.yaml', '.yml', '.toml', '.xml', '.md', '.txt', '.rst',
        '.sh', '.bash', '.zsh', '.c', '.cpp', '.h', '.hpp', '.java',
        '.go', '.rs', '.rb', '.php', '.sql', '.vue', '.svelte',
        '.gitignore', '.dockerignore', '.env'
    }
    
    def __init__(self, workspace_dir: Optional[str] = None):
        self.workspace_dir = Path(workspace_dir).resolve() if workspace_dir else Path.cwd().resolve()
        self.file_access_granted = False
        self.pending_changes: List[Dict] = []
        self.applied_changes: List[Dict] = []
        self.backup_dir = self.workspace_dir / ".bert_backup"
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.command_history: List[Dict] = []
    
    def request_file_access(self) -> bool:
        """Request user permission for file system access"""
        print(f"\n{Colors.SAGE}{'═' * 70}{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.LIGHT_SAGE}  BERT CODE MODE — File Access Request{Colors.RESET}")
        print(f"{Colors.SAGE}{'═' * 70}{Colors.RESET}\n")
        
        print(f"  Workspace: {Colors.CYAN}{self.workspace_dir}{Colors.RESET}\n")
        print(f"  {Colors.LIGHT_SAGE}This will allow Bert to:{Colors.RESET}")
        print(f"    • Read files in the workspace")
        print(f"    • Create new files")
        print(f"    • Edit existing files (with automatic backup)")
        print(f"    • Run shell commands (with your approval)\n")
        print(f"  {Colors.YELLOW}⚠ All file changes require your confirmation{Colors.RESET}\n")
        print(f"{Colors.SAGE}{'─' * 70}{Colors.RESET}")
        
        try:
            response = input(f"\n  Grant file access? {Colors.GRAY}[Y/n]{Colors.RESET} ").strip().lower()
            
            if response in ['', 'y', 'yes']:
                self.file_access_granted = True
                self.backup_dir.mkdir(parents=True, exist_ok=True)
                print(f"\n  {Colors.GREEN}✓ File access granted{Colors.RESET}\n")
                return True
            else:
                print(f"\n  {Colors.RED}✗ File access denied{Colors.RESET}\n")
                return False
                
        except (EOFError, KeyboardInterrupt):
            print(f"\n  {Colors.RED}✗ File access denied{Colors.RESET}\n")
            return False
    
    def _is_safe_path(self, filepath: str) -> bool:
        """Check if path is within workspace"""
        try:
            full_path = (self.workspace_dir / filepath).resolve()
            return str(full_path).startswith(str(self.workspace_dir))
        except Exception:
            return False
    
    def _is_text_file(self, filepath: Path) -> bool:
        """Check if file is likely a text file"""
        if filepath.suffix.lower() in self.TEXT_EXTENSIONS:
            return True
        if filepath.name in self.TEXT_EXTENSIONS:
            return True
        try:
            with open(filepath, 'rb') as f:
                chunk = f.read(1024)
                if b'\x00' in chunk:
                    return False
                try:
                    chunk.decode('utf-8')
                    return True
                except UnicodeDecodeError:
                    return False
        except Exception:
            return False
    
    def read_file(self, filepath: str) -> OperationResult:
        """Read a file from workspace"""
        if not self.file_access_granted:
            return OperationResult(False, "File access not granted")
        
        if not self._is_safe_path(filepath):
            return OperationResult(False, f"Path outside workspace: {filepath}")
        
        full_path = (self.workspace_dir / filepath).resolve()
        
        if not full_path.exists():
            return OperationResult(False, f"File not found: {filepath}")
        
        if not full_path.is_file():
            return OperationResult(False, f"Not a file: {filepath}")
        
        size = full_path.stat().st_size
        if size > self.MAX_FILE_SIZE:
            return OperationResult(False, f"File too large ({size / 1024 / 1024:.1f}MB)")
        
        if not self._is_text_file(full_path):
            return OperationResult(False, f"Binary file: {filepath}")
        
        try:
            with open(full_path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
            
            return OperationResult(True, f"Read {len(content)} chars", {
                'content': content,
                'path': str(full_path),
                'size': size,
                'lines': content.count('\n') + 1
            })
        except Exception as e:
            return OperationResult(False, f"Error reading: {e}")
    
    def list_files(self, pattern: str = "*", recursive: bool = False) -> OperationResult:
        """List files in workspace"""
        if not self.file_access_granted:
            return OperationResult(False, "File access not granted")
        
        try:
            files = []
            dirs = []
            
            glob_func = self.workspace_dir.rglob if recursive else self.workspace_dir.glob
            
            for path in glob_func(pattern):
                if any(p.startswith('.') for p in path.parts):
                    continue
                
                rel_path = path.relative_to(self.workspace_dir)
                
                if path.is_file():
                    files.append({
                        'path': str(rel_path),
                        'size': path.stat().st_size
                    })
                elif path.is_dir():
                    dirs.append(str(rel_path))
            
            return OperationResult(True, f"{len(files)} files, {len(dirs)} dirs", {
                'files': sorted(files, key=lambda x: x['path']),
                'directories': sorted(dirs)
            })
        except Exception as e:
            return OperationResult(False, f"Error: {e}")
    
    def tree(self, max_depth: int = 3) -> OperationResult:
        """Generate directory tree view"""
        if not self.file_access_granted:
            return OperationResult(False, "File access not granted")
        
        try:
            lines = [f"{Colors.CYAN}{self.workspace_dir.name}/{Colors.RESET}"]
            self._tree_recursive(self.workspace_dir, "", lines, 0, max_depth)
            return OperationResult(True, "Tree generated", {'tree': "\n".join(lines)})
        except Exception as e:
            return OperationResult(False, f"Error: {e}")
    
    def _tree_recursive(self, path: Path, prefix: str, lines: List[str], depth: int, max_depth: int):
        if depth >= max_depth:
            return
        
        try:
            items = sorted(path.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower()))
            items = [i for i in items if not i.name.startswith('.')]
            
            for i, item in enumerate(items):
                is_last = (i == len(items) - 1)
                connector = "└── " if is_last else "├── "
                
                if item.is_dir():
                    lines.append(f"{prefix}{connector}{Colors.BLUE}{item.name}/{Colors.RESET}")
                    new_prefix = prefix + ("    " if is_last else "│   ")
                    self._tree_recursive(item, new_prefix, lines, depth + 1, max_depth)
                else:
                    size = item.stat().st_size
                    size_str = f"{size}B" if size < 1024 else f"{size/1024:.1f}K"
                    lines.append(f"{prefix}{connector}{item.name} {Colors.GRAY}({size_str}){Colors.RESET}")
        except PermissionError:
            pass
    
    def _create_backup(self, filepath: Path) -> Optional[str]:
        """Create backup of existing file"""
        if not filepath.exists():
            return None
        
        try:
            content_hash = hashlib.md5(filepath.read_bytes()).hexdigest()[:8]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"{filepath.name}_{timestamp}_{content_hash}.bak"
            backup_path = self.backup_dir / backup_name
            shutil.copy2(filepath, backup_path)
            return str(backup_path)
        except Exception:
            return None
    
    def propose_change(self, filepath: str, new_content: str, description: str = "") -> OperationResult:
        """Propose a file change for approval"""
        if not self.file_access_granted:
            return OperationResult(False, "File access not granted")
        
        if not self._is_safe_path(filepath):
            return OperationResult(False, f"Path outside workspace")
        
        full_path = (self.workspace_dir / filepath).resolve()
        is_new = not full_path.exists()
        
        if full_path.suffix.lower() in self.DANGEROUS_EXTENSIONS:
            return OperationResult(False, f"Cannot modify: {full_path.suffix}")
        
        old_content = None
        if not is_new:
            read_result = self.read_file(filepath)
            if read_result.success:
                old_content = read_result.data['content']
        
        change = {
            'id': len(self.pending_changes),
            'filepath': filepath,
            'full_path': full_path,
            'is_new': is_new,
            'old_content': old_content,
            'new_content': new_content,
            'description': description
        }
        
        self.pending_changes.append(change)
        action = "CREATE" if is_new else "EDIT"
        return OperationResult(True, f"Proposed {action}: {filepath}", {'id': change['id']})
    
    def show_diff(self, change_id: int) -> OperationResult:
        """Show diff for a pending change"""
        if change_id >= len(self.pending_changes):
            return OperationResult(False, f"Invalid change ID")
        
        change = self.pending_changes[change_id]
        
        print(f"\n{Colors.SAGE}{'─' * 70}{Colors.RESET}")
        
        if change['is_new']:
            print(f"{Colors.GREEN}+ NEW FILE: {change['filepath']}{Colors.RESET}")
            print(f"{Colors.SAGE}{'─' * 70}{Colors.RESET}\n")
            
            lines = change['new_content'].splitlines()
            for line in lines[:30]:
                print(f"{Colors.GREEN}+ {line}{Colors.RESET}")
            if len(lines) > 30:
                print(f"{Colors.GRAY}... ({len(lines) - 30} more lines){Colors.RESET}")
        else:
            print(f"{Colors.YELLOW}~ EDIT: {change['filepath']}{Colors.RESET}")
            print(f"{Colors.SAGE}{'─' * 70}{Colors.RESET}\n")
            
            old_lines = (change['old_content'] or '').splitlines(keepends=True)
            new_lines = change['new_content'].splitlines(keepends=True)
            
            diff = list(difflib.unified_diff(
                old_lines, new_lines,
                fromfile=f"a/{change['filepath']}",
                tofile=f"b/{change['filepath']}"
            ))
            
            for line in diff[:60]:
                if line.startswith('+') and not line.startswith('+++'):
                    print(f"{Colors.GREEN}{line.rstrip()}{Colors.RESET}")
                elif line.startswith('-') and not line.startswith('---'):
                    print(f"{Colors.RED}{line.rstrip()}{Colors.RESET}")
                elif line.startswith('@@'):
                    print(f"{Colors.CYAN}{line.rstrip()}{Colors.RESET}")
                else:
                    print(line.rstrip())
            
            if len(diff) > 60:
                print(f"\n{Colors.GRAY}... ({len(diff) - 60} more lines){Colors.RESET}")
        
        print(f"\n{Colors.SAGE}{'─' * 70}{Colors.RESET}\n")
        return OperationResult(True, "Diff displayed")
    
    def show_all_pending(self) -> OperationResult:
        """Show summary of pending changes"""
        if not self.pending_changes:
            print(f"\n{Colors.GRAY}No pending changes{Colors.RESET}\n")
            return OperationResult(True, "No pending changes")
        
        print(f"\n{Colors.SAGE}{'═' * 70}{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.LIGHT_SAGE}  Pending Changes ({len(self.pending_changes)}){Colors.RESET}")
        print(f"{Colors.SAGE}{'═' * 70}{Colors.RESET}\n")
        
        for change in self.pending_changes:
            action = f"{Colors.GREEN}CREATE{Colors.RESET}" if change['is_new'] else f"{Colors.YELLOW}EDIT{Colors.RESET}"
            print(f"  [{change['id']}] {action}: {change['filepath']}")
        
        print(f"\n{Colors.SAGE}{'─' * 70}{Colors.RESET}")
        print(f"  {Colors.GRAY}[Y] Apply all  [N] Reject all  [D] Show diffs  [#] Show specific{Colors.RESET}")
        print(f"{Colors.SAGE}{'─' * 70}{Colors.RESET}\n")
        
        return OperationResult(True, f"{len(self.pending_changes)} pending")
    
    def confirm_and_apply(self) -> OperationResult:
        """Interactive confirmation and apply"""
        if not self.pending_changes:
            return OperationResult(True, "No changes to apply")
        
        self.show_all_pending()
        
        while True:
            try:
                choice = input(f"  Your choice: ").strip().upper()
                
                if choice == 'Y':
                    return self._apply_all_changes()
                elif choice == 'N':
                    self.pending_changes = []
                    return OperationResult(False, "Changes rejected")
                elif choice == 'D':
                    for change in self.pending_changes:
                        self.show_diff(change['id'])
                elif choice.isdigit() and int(choice) < len(self.pending_changes):
                    self.show_diff(int(choice))
                else:
                    print(f"  {Colors.GRAY}Invalid option{Colors.RESET}")
            except (EOFError, KeyboardInterrupt):
                self.pending_changes = []
                return OperationResult(False, "Cancelled")
    
    def _apply_all_changes(self) -> OperationResult:
        """Apply all pending changes"""
        applied = 0
        failed = 0
        
        for change in self.pending_changes:
            try:
                change['full_path'].parent.mkdir(parents=True, exist_ok=True)
                
                backup_path = None
                if not change['is_new']:
                    backup_path = self._create_backup(change['full_path'])
                
                with open(change['full_path'], 'w', encoding='utf-8') as f:
                    f.write(change['new_content'])
                
                self.applied_changes.append({**change, 'backup_path': backup_path})
                
                action = "Created" if change['is_new'] else "Modified"
                backup_info = f" (backup saved)" if backup_path else ""
                print(f"  {Colors.GREEN}✓ {action}: {change['filepath']}{backup_info}{Colors.RESET}")
                applied += 1
            except Exception as e:
                print(f"  {Colors.RED}✗ Failed: {change['filepath']} - {e}{Colors.RESET}")
                failed += 1
        
        self.pending_changes = []
        
        if failed == 0:
            return OperationResult(True, f"Applied {applied} change(s)")
        return OperationResult(False, f"Applied {applied}, failed {failed}")
    
    def run_command(self, command: str, description: str = "", timeout: int = 60) -> OperationResult:
        """Run a shell command with approval"""
        if not self.file_access_granted:
            return OperationResult(False, "File access not granted")
        
        print(f"\n{Colors.SAGE}{'═' * 70}{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.YELLOW}  ⚡ Command Execution Request{Colors.RESET}")
        print(f"{Colors.SAGE}{'═' * 70}{Colors.RESET}\n")
        
        print(f"  Command: {Colors.CYAN}{command}{Colors.RESET}")
        if description:
            print(f"  Purpose: {description}")
        print(f"  Dir:     {self.workspace_dir}\n")
        print(f"  {Colors.YELLOW}⚠ This will execute on your system{Colors.RESET}")
        print(f"{Colors.SAGE}{'─' * 70}{Colors.RESET}")
        
        try:
            approval = input(f"\n  Allow? {Colors.GRAY}[y/N]{Colors.RESET} ").strip().lower()
            if approval not in ['y', 'yes']:
                return OperationResult(False, "Command rejected")
        except (EOFError, KeyboardInterrupt):
            return OperationResult(False, "Cancelled")
        
        print(f"\n  {Colors.GRAY}Executing...{Colors.RESET}\n")
        
        try:
            result = subprocess.run(
                command, shell=True, cwd=self.workspace_dir,
                capture_output=True, text=True, timeout=timeout
            )
            
            self.command_history.append({
                'command': command,
                'returncode': result.returncode,
                'timestamp': datetime.now().isoformat()
            })
            
            output = result.stdout
            if result.stderr:
                output += f"\n{result.stderr}"
            
            if result.returncode == 0:
                print(f"  {Colors.GREEN}✓ Command completed{Colors.RESET}")
                if output.strip():
                    lines = output.strip().splitlines()
                    for line in lines[:30]:
                        print(f"  {line}")
                    if len(lines) > 30:
                        print(f"  {Colors.GRAY}... ({len(lines) - 30} more){Colors.RESET}")
                return OperationResult(True, "Success", {'output': output})
            else:
                print(f"  {Colors.RED}✗ Failed (exit {result.returncode}){Colors.RESET}")
                return OperationResult(False, f"Exit {result.returncode}")
                
        except subprocess.TimeoutExpired:
            return OperationResult(False, f"Timed out after {timeout}s")
        except Exception as e:
            return OperationResult(False, f"Error: {e}")
    
    def get_summary(self) -> str:
        """Get session summary"""
        parts = []
        if self.applied_changes:
            creates = sum(1 for c in self.applied_changes if c['is_new'])
            edits = len(self.applied_changes) - creates
            if creates:
                parts.append(f"{creates} created")
            if edits:
                parts.append(f"{edits} modified")
        if self.command_history:
            parts.append(f"{len(self.command_history)} commands")
        return ", ".join(parts) if parts else "No changes"
    
    def end_session(self):
        """End the code session"""
        if self.pending_changes:
            print(f"\n{Colors.YELLOW}Warning: {len(self.pending_changes)} pending changes discarded{Colors.RESET}")
        print(f"\n{Colors.GREEN}✓ Code session ended. {self.get_summary()}{Colors.RESET}\n")


# ═══════════════════════════════════════════════════════════════════════════════
# GLOBAL SESSION
# ═══════════════════════════════════════════════════════════════════════════════

_code_session: Optional[BertCodeSession] = None


def get_code_session(workspace_dir: Optional[str] = None) -> BertCodeSession:
    """Get or create the global code session"""
    global _code_session
    if _code_session is None or workspace_dir:
        _code_session = BertCodeSession(workspace_dir)
    return _code_session


def end_code_session():
    """End the current code session"""
    global _code_session
    if _code_session:
        _code_session.end_session()
        _code_session = None


if __name__ == "__main__":
    print(f"\n{Colors.LIGHT_SAGE}BERT Code Module — Test{Colors.RESET}\n")
    session = get_code_session()
    if session.request_file_access():
        result = session.tree(max_depth=2)
        if result.success:
            print(result.data['tree'])
    end_code_session()
