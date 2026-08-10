"""Vercel entrypoint for the SupportFlow FastAPI application."""

from supportflow.api import app

__all__ = ["app"]
