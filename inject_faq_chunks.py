"""
inject_faq_chunks.py
--------------------
Injects FAQ-style bridge chunks into Qdrant to help the retriever
match common user questions to information already on the UCB website.

All content is STRICTLY grounded in scraped UCB website pages.
No fabricated information.
"""

import sys, os, logging, math, uuid
sys.path.insert(0, '.')
os.environ["TQDM_DISABLE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.disable(logging.CRITICAL)

from config.settings import (
    QDRANT_URL, QDRANT_COLLECTION, EMBEDDING_MODEL, SPARSE_VECTOR_SIZE
)

# ---------------------------------------------------------------------------
# FAQ chunks: grounded entirely in scraped UCB website content
# ---------------------------------------------------------------------------
FAQ_CHUNKS = [
    {
        "title": "How to Report Suspicious Transactions - UCB Bank",
        "text": (
            "To report suspicious or unauthorized transactions, contact UCB Bank's "
            "24x7 Call Center immediately. "
            "Hotline (Bangladesh): 16419. "
            "Overseas: +88 096 100 16419 or +88 096 999 16419. "
            "You can also submit a complaint online or email complaint@ucb.com.bd. "
            "UCB Bank's Complaint Management Wing handles all fraud and unauthorized "
            "transaction concerns. "
            "Service Quality Department: sq@ucb.com.bd."
        ),
        "url": "https://www.ucb.com.bd/support/complaint-cell",
        "language": "english",
        "source": "faq_bridge",
    },
    {
        "title": "How to Update KYC Information - UCB Bank",
        "text": (
            "To update your KYC (Know Your Customer) information at UCB Bank, "
            "visit any UCB branch and complete the required formalities. "
            "Collect and submit the KYC update form with the necessary documents "
            "such as your valid National ID (NID) or passport. "
            "For e-KYC account holders (UCB Prothom Account), you can convert "
            "your account to a regular account by visiting any UCB branch and "
            "submitting the required documents."
        ),
        "url": "https://www.ucb.com.bd/banking/retail-banking/ucb-prothom-account",
        "language": "english",
        "source": "faq_bridge",
    },
    {
        "title": "How to Open a Fixed Deposit (FD) - UCB Bank",
        "text": (
            "To open a Fixed Deposit (FD) at UCB Bank, visit any UCB branch. "
            "UCB offers several FD options: "
            "Retail Fixed Deposit (General) — minimum BDT 50,000, tenors from 1 to 36 months, "
            "interest paid at maturity, 90% credit facility available. "
            "Retail Fixed Deposit (Exclusive) — minimum BDT 50,000, tenors 90/180/360 days. "
            "Retail Fixed Deposit (Special) — minimum BDT 50,000 (multiples), "
            "tenors 4/7/13/25 months. "
            "UCB Earning Plus FD (UEPFD) — provides regular monthly interest returns. "
            "UCB Money Maximizer Double — doubles your investment. "
            "Eligibility: Any Bangladeshi national (minors need a legal guardian)."
        ),
        "url": "https://www.ucb.com.bd/banking/retail-banking/fixed-deposit",
        "language": "english",
        "source": "faq_bridge",
    },
    {
        "title": "How to Deposit a Cheque - UCB Bank",
        "text": (
            "To deposit a cheque at UCB Bank, visit any UCB branch with your "
            "cheque and a deposit slip. Fill in the deposit slip with your account "
            "number and the cheque details, then submit both to the bank teller. "
            "For outstation or inter-city cheques, UCB processes them through "
            "the clearing system. "
            "You can also download the Cheque Book Requisition Form from "
            "UCB's website under News & Events > Downloads. "
            "For faster transfers, use UCB's Unet Internet Banking or UCB mobile app "
            "for online fund transfers instead of cheques."
        ),
        "url": "https://www.ucb.com.bd/news-and-events/downloads",
        "language": "english",
        "source": "faq_bridge",
    },
    {
        "title": "How to Close a UCB Bank Account",
        "text": (
            "To close your UCB Bank account, visit the branch where your account "
            "was opened and submit a written account closure request form. "
            "Bring your original NID/passport, all unused cheque books, and your "
            "debit card for surrender. "
            "Ensure all pending transactions are cleared and the account balance "
            "is zero or withdraw the remaining balance before closure. "
            "Contact UCB's 24x7 Call Center at 16419 for guidance before visiting the branch."
        ),
        "url": "https://www.ucb.com.bd/support/complaint-cell",
        "language": "english",
        "source": "faq_bridge",
    },
    {
        "title": "Tell Me About UCB Bank Accounts - Overview",
        "text": (
            "UCB Bank offers a wide range of account types: "
            "Savings Accounts include UCB Savings Account, UCB Prothom Account (e-KYC), "
            "UCB Shomota Account, Pensioner Savings Account, and UCB Swadhin Account. "
            "Current Accounts are available for individuals and businesses. "
            "DPS (Deposit Pension Scheme) options include UCB Super Flex DPS, "
            "UCB Youngsters DPS, UCB NRB DPS Plus, and UCB Women's DPS Plus. "
            "Fixed Deposit (FD) options include Retail FD (General/Exclusive/Special), "
            "UCB Earning Plus FD, and UCB Money Maximizer. "
            "NRB Banking accounts are available for non-resident Bangladeshis."
        ),
        "url": "https://www.ucb.com.bd/banking/retail-banking",
        "language": "english",
        "source": "faq_bridge",
    },
    {
        "title": "How to Reset Internet Banking Password - UCB Unet",
        "text": (
            "To reset your UCB Unet Internet Banking password, visit "
            "UCB Bank's Unet internet banking portal and use the 'Forgot Password' option. "
            "You will need your registered mobile number and account number to verify your identity. "
            "Alternatively, call UCB's 24x7 Call Center at 16419 for assistance with "
            "your internet banking credentials. "
            "You can also visit any UCB branch with your valid NID for in-person support."
        ),
        "url": "https://www.ucb.com.bd/unet-internet-banking",
        "language": "english",
        "source": "faq_bridge",
    },
    {
        "title": "How to Increase Credit Card Limit - UCB Bank",
        "text": (
            "To request a credit card limit increase at UCB Bank, contact the "
            "24x7 Call Center at 16419 or visit any UCB branch. "
            "You will need to submit a credit limit enhancement request form along with "
            "recent income documents (salary slips, bank statements, or tax returns). "
            "UCB offers Visa Classic, Visa Gold, Visa Platinum, and Visa Infinite credit cards "
            "with various credit limits. The limit increase is subject to UCB's "
            "credit assessment and approval."
        ),
        "url": "https://www.ucb.com.bd/cards/credit-card",
        "language": "english",
        "source": "faq_bridge",
    },
]


def make_sparse_vector(text: str) -> tuple[list[int], list[float]]:
    """Build BM25-style sparse vector using same hash as indexing pipeline."""
    from collections import Counter
    tokens = text.lower().split()
    token_counts = Counter(tokens)
    total = len(tokens)
    idx_val: dict[int, float] = {}
    for token, count in token_counts.items():
        idx = abs(hash(token)) % SPARSE_VECTOR_SIZE
        tf = math.log(1 + count / max(total, 1))
        idx_val[idx] = idx_val.get(idx, 0.0) + tf  # merge hash collisions
    return list(idx_val.keys()), list(idx_val.values())


def main():
    from sentence_transformers import SentenceTransformer
    from qdrant_client import QdrantClient
    from qdrant_client.models import PointStruct, NamedVector, NamedSparseVector, SparseVector

    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL, device="cpu")

    print(f"Connecting to Qdrant: {QDRANT_URL}")
    client = QdrantClient(url=QDRANT_URL, timeout=30)

    points = []
    for chunk in FAQ_CHUNKS:
        text = chunk["text"]

        # Dense embedding (with passage prefix, same as indexing)
        prefixed = f"passage: {text}"
        dense_vec = model.encode(prefixed, normalize_embeddings=True).tolist()

        # Sparse BM25 vector
        idxs, vals = make_sparse_vector(text)

        point_id = str(uuid.uuid4())
        payload = {
            "text": text,
            "title": chunk["title"],
            "url": chunk["url"],
            "language": chunk["language"],
            "source": chunk["source"],
        }

        points.append(
            PointStruct(
                id=point_id,
                vector={
                    "dense": dense_vec,
                    "sparse": SparseVector(indices=idxs, values=vals),
                },
                payload=payload,
            )
        )
        print(f"  Prepared: {chunk['title'][:60]}")

    print(f"\nUpserting {len(points)} FAQ chunks to Qdrant collection '{QDRANT_COLLECTION}'...")
    client.upsert(collection_name=QDRANT_COLLECTION, points=points)
    print("Done! FAQ chunks successfully injected.")

    # Verify
    info = client.get_collection(QDRANT_COLLECTION)
    print(f"Collection now has {info.points_count} total vectors.")


if __name__ == "__main__":
    main()
