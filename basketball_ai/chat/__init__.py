"""Deprecated legacy chat package.

It is intentionally not mounted by FastAPI or the operations console. Chat V3
in WordPress is the only conversational layer; this package remains for one
release solely to allow old serialized sessions to be inspected.
"""
import warnings
warnings.warn("basketball_ai.chat is deprecated; use WordPress Chat V3", DeprecationWarning, stacklevel=2)
