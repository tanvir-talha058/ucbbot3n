"""
chat.py
-------
Terminal chat interface for the UCB Bank RAG chatbot.

Usage:
  python chat.py
"""

import logging
import os
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# Silence all warnings and progress bars
warnings.filterwarnings("ignore")
os.environ["TQDM_DISABLE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.disable(logging.CRITICAL)

# Suppress stderr entirely — kills tqdm progress bars from sentence_transformers
sys.stderr = open(os.devnull, "w")

from pipeline.rag_pipeline import UCBRagPipeline

def main():
    print("=" * 50)
    print("  UCB Bank RAG Chatbot — Terminal Mode")
    print("  Type 'quit' to exit")
    print("=" * 50)
    print()

    print("Loading pipeline...")
    pipeline = UCBRagPipeline()

    print()
    print("Bot: Welcome to UCB Bank! I'm your virtual assistant.")
    print("     I can help you with loans, cards, accounts, and all UCB Bank services.")
    print("     How can I assist you today?")
    print()

    session_id = "terminal_session"

    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        result = pipeline.chat(query=query, session_id=session_id)

        print(f"\nBot: {result['answer']}")
        print()


if __name__ == "__main__":
    main()
