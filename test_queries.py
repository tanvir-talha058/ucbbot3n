"""Run all test queries through the pipeline and print results."""
import sys, os, warnings, logging
sys.path.insert(0, '.')
warnings.filterwarnings("ignore")
os.environ["TQDM_DISABLE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.disable(logging.CRITICAL)

from pipeline.rag_pipeline import UCBRagPipeline

pipeline = UCBRagPipeline()

QUERIES = [
    # 1. Account Services
    ("What documents are required to open a savings account?", "english"),
    ("How can I close my account?", "english"),
    ("What is the minimum balance requirement?", "english"),
    # 2. Loans & Credit
    ("How do I apply for a personal loan?", "english"),
    ("Can I increase my credit card limit?", "english"),
    ("Do you offer student loans?", "english"),
    # 3. Deposits & Transactions
    ("How do I open a fixed deposit?", "english"),
    ("What is the process to deposit a cheque?", "english"),
    ("What is the daily online transfer limit?", "english"),
    # 4. Digital Banking
    ("How do I register for mobile banking?", "english"),
    ("How can I reset my internet banking password?", "english"),
    ("How do I block my debit card if it's lost?", "english"),
    # 5. Customer Support
    ("What should I do if I lose my ATM card?", "english"),
    ("What is the customer care helpline number?", "english"),
    ("What are the branch operating hours?", "english"),
    # 6. Compliance & Security
    ("How do I update my KYC information?", "english"),
    ("What security measures do you use for online banking?", "english"),
    ("How do I report suspicious transactions?", "english"),
    # 7. Edge Cases
    ("Tell me about accounts.", "english"),
    ("Can I open an account and apply for a loan together?", "english"),
    ("Savings account opening process please", "english"),
    ("How to open savings account?", "english"),
]

CATEGORIES = [
    "1. Account Services",
    "1. Account Services",
    "1. Account Services",
    "2. Loans & Credit",
    "2. Loans & Credit",
    "2. Loans & Credit",
    "3. Deposits & Transactions",
    "3. Deposits & Transactions",
    "3. Deposits & Transactions",
    "4. Digital Banking",
    "4. Digital Banking",
    "4. Digital Banking",
    "5. Customer Support",
    "5. Customer Support",
    "5. Customer Support",
    "6. Compliance & Security",
    "6. Compliance & Security",
    "6. Compliance & Security",
    "7. Edge Cases",
    "7. Edge Cases",
    "7. Edge Cases",
    "7. Edge Cases",
]

current_cat = ""
for i, ((query, lang), cat) in enumerate(zip(QUERIES, CATEGORIES)):
    if cat != current_cat:
        print(f"\n{'='*70}")
        print(f"  {cat}")
        print(f"{'='*70}")
        current_cat = cat

    result = pipeline.chat(query=query, session_id=f"test_{i}", language=lang)
    fallback_tag = " [FALLBACK]" if result["fallback"] else ""
    print(f"\nQ: {query}")
    print(f"A:{fallback_tag} {result['answer']}")
    print(f"   Retrieved: {result['retrieved_count']} chunks")
