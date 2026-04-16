"""
app.py
------
One-command launcher for the UCB Bank RAG Chatbot.

Starts the FastAPI backend and opens the chat UI in your browser.

Usage:
  python app.py
"""

import subprocess
import sys
import time
import webbrowser
import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).parent
load_dotenv(PROJECT_ROOT / ".env")

HOST = os.getenv("APP_HOST", "127.0.0.1")
PORT = int(os.getenv("APP_PORT", "8080"))
OPEN_BROWSER = os.getenv("OPEN_BROWSER", "true").lower() == "true"
URL = os.getenv("APP_URL", f"http://127.0.0.1:{PORT}")

def main():
    print("=" * 52)
    print("  UCB Bank RAG Chatbot")
    print(f"  Starting server at {URL}")
    print("  Press Ctrl+C to stop")
    print("=" * 52)

    # Start uvicorn as a subprocess
    server = subprocess.Popen(
        [
            sys.executable, "-m", "uvicorn",
            "api.main:app",
            "--host", HOST,
            "--port", str(PORT),
            "--log-level", "warning",   # suppress info noise
        ],
        cwd=str(PROJECT_ROOT),
    )

    # Wait for server to be ready — just check TCP connection, not /health
    import socket
    print("\nWaiting for server", end="", flush=True)
    for _ in range(30):
        try:
            s = socket.create_connection((HOST, PORT), timeout=1)
            s.close()
            break
        except OSError:
            print(".", end="", flush=True)
            time.sleep(1)
    print(" ready!\n")

    # Open browser only when running on a desktop session.
    if OPEN_BROWSER:
        print(f"Opening chatbot at {URL} ...")
        webbrowser.open(URL)
    else:
        print(f"Browser auto-open disabled. Open {URL} from your own machine or tunnel the port.")

    print("Chat UI is open. Press Ctrl+C here to shut down.\n")

    try:
        server.wait()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.terminate()
        server.wait()
        print("Goodbye!")


if __name__ == "__main__":
    main()
