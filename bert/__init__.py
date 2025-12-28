"""
Bert CLI — A calm, local AI assistant by Biwa Industries
Version 1.0.0b (Beta)
"""

__version__ = "1.0.0b"
__author__ = "Biwa Industries"
__email__ = "contact@biwaindustries.com"

from bert.cli import main, BertCLI
from bert.engine import get_engine, BertEngine

__all__ = ['main', 'BertCLI', 'get_engine', 'BertEngine', '__version__']
