# BERT CLI

**A Friendly, local AI assistant by Biwa**

![Biwa](https://img.shields.io/badge/Biwa-5F6F64?style=for-the-badge&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA4AAAAOCAYAAAAfSC3RAAABCGlDQ1BJQ0MgUHJvZmlsZQAAeJxjYGA8wQAELAYMDLl5JUVB7k4KEZFRCuwPGBiBEAwSk4sLGHADoKpv1yBqL+viUYcLcKakFicD6Q9ArFIEtBxopAiQLZIOYWuA2EkQtg2IXV5SUAJkB4DYRSFBzkB2CpCtkY7ETkJiJxcUgdT3ANk2uTmlyQh3M/Ck5oUGA2kOIJZhKGYIYnBncAL5H6IkfxEDg8VXBgbmCQixpJkMDNtbGRgkbiHEVBYwMPC3MDBsO48QQ4RJQWJRIliIBYiZ0tIYGD4tZ2DgjWRgEL7AwMAVDQsIHG5TALvNnSEfCNMZchhSgSKeDHkMyQx6QJYRgwGDIYMZAKbWPz9HbOBQAAAA6ElEQVR42o3SMS8FURAF4G/3PiIReWhVGhGJPJVKSCSC8iUKCoUoxB/QiFbtf+g14gdQUiF50YiSTkHQzMqz1q7T3JMz98zcmbnJT2TIMYoDrOEZTxGrRXFhDPu4wkTojeZ+zGA8eK5MSujXN3EY/KPG8x1o4wj3uMYJhiOWsor+PjGJFSzjDC9Yx1sk6KWSKUMHC/GsV9zgISb9iGm0WhXGLkaQMBhnG3O4xSo6WcUaBrCLbbyjh6FIMoVTHNetYQN3WMIsLrFXKvJrQEXvW7jAOXaKiTZ9hKL/eSyWtEakP/i/kFeZvgBS0h/GkIM2hAAAAABJRU5ErkJggg==&logoColor=white)
![BERT CLI](https://img.shields.io/badge/Bert-CLI-598556?style=for-the-badge)
![Version](https://img.shields.io/badge/version-1.0.0-96C3A3?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Windows](https://img.shields.io/badge/Windows-0078D6?style=for-the-badge&logo=windows&logoColor=white)
![Linux](https://img.shields.io/badge/Linux-FCC624?style=for-the-badge&logo=linux&logoColor=black)
![HuggingFace](https://img.shields.io/badge/HuggingFace-%23FFD21E.svg?style=for-the-badge&logo=huggingface&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey)
![License](https://img.shields.io/badge/License-Apache2.0-blue)

---

## UPDATES

**¡BERT CLI is now officially out of Beta phase!**

Visit berts Official GitHub Page here: [Bert CLI official Page](https://mnisperuza.github.io/bert-cli/)


## ⚖️ Legal & Licensing
This is **Software Licensed Under Apache 2.0**.  
- For licensing terms, see [LICENSE](./LICENSE).

---

## Overview

### About Bert:
![Bert CLI](https://img.shields.io/badge/Bert-598556?style=for-the-badge&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA4AAAAOCAYAAAAfSC3RAAABCGlDQ1BJQ0MgUHJvZmlsZQAAeJxjYGA8wQAELAYMDLl5JUVB7k4KEZFRCuwPGBiBEAwSk4sLGHADoKpv1yBqL+viUYcLcKakFicD6Q9ArFIEtBxopAiQLZIOYWuA2EkQtg2IXV5SUAJkB4DYRSFBzkB2CpCtkY7ETkJiJxcUgdT3ANk2uTmlyQh3M/Ck5oUGA2kOIJZhKGYIYnBncAL5H6IkfxEDg8VXBgbmCQixpJkMDNtbGRgkbiHEVBYwMPC3MDBsO48QQ4RJQWJRIliIBYiZ0tIYGD4tZ2DgjWRgEL7AwMAVDQsIHG5TALvNnSEfCNMZchhSgSKeDHkMyQx6QJYRgwGDIYMZAKbWPz9HbOBQAAACHUlEQVR42kXSy2pTURjF8f+3zzHmcpLehCalFCvWWooIVsFBoRcRh6IOWtA3cGAfoI+gL+DMF3DQiaCCDhQc2JaqFaSIsbV3m8TE5CTpOftzsKOO9mbDWmuwfzI6VVDPExBBBIwRRARVRRVEQBUAVBXUnUaMC7kANJohjWaIonieAUBcDuncBMEX3EoURyBwaXQMATa2tyhVq3QHWay1iOLmO02+iBDHMelkioVb8xyVKhzVKly7cJmNvS2evnlNJpkishYAXwxWFWOMUAtD7s7e4O3KKotPHvPpR5Fn79+RTaRYuDlHLQzpy+XoSmeIbIwAJopj+nI58kGWF2vLPLr/gInT55i5OEElrFOq/WJuchbf87k3fZ1Wu40Yg7GqJBMJXq0sMzI0xHGzzff9XVa/fOZsYZCvu9ucyec5LJfxPUNPOksUR/ie8ag3Q16urzHU34/vCSODBXrTAeIZKtXfWGsJTqbwBephE5Mw+O5jhO5shuL+Du2oRaVeY32zSNSyDBcGKO7sMjk+zubOHuVGg3yqCxmbGVDjCYKgKGJg8c48QTpJq3VM1I74dnBA2vd5uLREkE1hBGRs2gXpqImtJYoibl+5ynDfKbZLZTYPf/L8w0dyuRQnPINVkPNTBbfYISdGAKXSaKAxGBFiq/QEaccOwCr+X4f/Xq2CQG8m6EBxbq21dAw4CABqAVHESAe0W3H1+r/USUcV/gAs9ewGK9JGLAAAAABJRU5ErkJggg==&logoColor=white)

Bert is designed as a reliable AI assistant, something you can always rely on, you can see Bert as a Friend, as a service, as a companion, or as something else that you could value.



<p align="center">
  <img src="assets/preview.png" alt="Bert CLI Preview" width="500">
</p>


---
## Quick Start


### PyPI

```bash
# Directly from PyPI Package:

pip install bert-cli

```

---

## Usage

### Start Bert
```bash
bert
```
#### Claim Your weekly token to start using Bert CLI

Go to [Bert CLI official Page](https://mnisperuza.github.io/bert-cli/).

Claim yout token -

Then run bert and use:
```
/*token YOUR-TOKEN-HERE
```

## Key Features for Out of Beta 1.0.0

- ✅ **ESC Interrupt** — Stop generation mid-stream
- ✅ **Token-by-token Streaming** — See responses appear naturally
- ✅ **Thinking Display** — See model's reasoning with `/*think`
- ✅ **Path-aware File Inspection** — Use `@path/to/file` in queries
- ✅ **Automatic Context Compression** — Never lose context
- ✅ **Weekly Token System** — 20K tokens free per week
- ✅ **Gradient UI** — Beautiful color-coded model names

>For more info visit:  [Bert CLI official Page](https://mnisperuza.github.io/bert-cli/)

---

## Models

| Model | Base | VRAM | Features |
|-------|------|------|----------|
| **Bert Nano** | LiquidAI/LFM2-700M | ~2GB | Ultra-fast |
| **Bert Mini** | LiquidAI/LFM2-1.2B | ~4GB | Balanced |
| **Bert Main** | Qwen/Qwen3-1.7B | ~5GB | Thinking 🧠 |
| **Bert Max** | LiquidAI/LFM2-2.6B | ~8GB | Reasoning |
| **Bert Coder** | Qwen/Qwen2.5-Coder-1.5B-Instruct | ~4GB | Code |
| **Bert Max-Coder** | Qwen/Qwen2.5-Coder-3B-Instruct | ~8GB | Heavy Code |

### Command Line Options
```bash
bert --ver      # Show version
bert --info     # Show info
bert --del      # Remove Bert data (~/.bert)
bert --help     # Show help
```

### In-Session Commands

**Switch Models:**
```
bert nano       # Fastest (0.7B)
bert mini       # Balanced (1.2B)
bert main       # Flagship (1.7B)
bert max        # Most capable (2.6B)
bert coder      # Code-optimized (1.5B)
bert maxcoder   # The best for Code (3B)
```

**Change Quantization:**
```
bert int4       # Balanced ⭐ 
bert int8       # High quality 
bert fp16       # Best quality (all platforms)
bert fp32       # Full precision / CPU
```

**Other Commands:**
```
/*help          # Show all commands
/*status        # Show current status
/*clear         # Clear screen
/*exit          # Exit Bert
```

---

## System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| RAM | 8GB | 16GB+ |
| VRAM | 3GB | 6GB+ |
| Python | 3.8 | 3.10+ |
| Storage | 30GB | 40GB |

### Quantization Support

| Platform | INT4/INT8 | FP16 | FP32 |
|----------|-----------|------|------|
| Linux | ✅ | ✅ | ✅ |
| Windows | ✅* | ✅ | ✅ |
| macOS | ❌ | ✅ | ✅ |



---

## Uninstall

Remove Bert data:
```bash
bert --del
```

---

## Support

- **GitHub Issues**: [github.com/mnisperuza/bert-cli/issues](https://github.com/mnisperuza/bert-cli/issues)

- **Email**: biwaindustries@gmail.com

When reporting issues, include:
1. Bert version (`bert --ver`)
2. Your OS (Windows/Linux/macOS)
3. Error message



---
Thanks for Using Bert CLI ❤️

**Built with restraint and long-term intent.**  
**Biwa — 2025**

---


