#!/usr/bin/env python3
"""
Simple launcher script for the Error Detection GUI

This script can be used to launch the GUI application.
"""

import subprocess
import sys
import os
from pathlib import Path

def main():
    """Launch the GUI application"""
    gui_script = Path(__file__).parent / "gui.py"

    if not gui_script.exists():
        print(f"Error: GUI script not found at {gui_script}")
        sys.exit(1)

    try:
        print("Launching Error Detection GUI...")
        print(f"Script location: {gui_script}")
        print("Note: Close the GUI window to return to the terminal")

        # Launch the GUI
        subprocess.run([sys.executable, str(gui_script)], check=True)

    except subprocess.CalledProcessError as e:
        print(f"Error launching GUI: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nLauncher interrupted")
        sys.exit(0)

if __name__ == "__main__":
    main()