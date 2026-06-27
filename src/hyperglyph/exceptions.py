"""Exceptions for the Hyper Glyph package."""


class HyperGlyphError(Exception):
    """Base exception for Hyper Glyph."""


class OptionalDependencyError(HyperGlyphError):
    """Raised when optional dependencies are missing."""
