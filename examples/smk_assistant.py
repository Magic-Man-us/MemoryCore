"""Compatibility launcher for the current typed-memory example.

The former SMK implementation and fixed taxonomy have been removed.
"""
import asyncio

from typed_memory import main

if __name__ == "__main__":
    asyncio.run(main())
