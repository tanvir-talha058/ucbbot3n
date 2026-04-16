"""
inject_cards_faq.py
-------------------
Inject credit card and debit card FAQ chunks into Qdrant.
Content is grounded in scraped UCB website pages.
"""

import sys, os, logging, math, uuid
sys.path.insert(0, '.')
os.environ["TQDM_DISABLE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.disable(logging.CRITICAL)

from config.settings import (
    QDRANT_URL, QDRANT_COLLECTION, EMBEDDING_MODEL, SPARSE_VECTOR_SIZE
)

CARD_FAQ_CHUNKS = [
    {
        "title": "UCB Credit Cards Overview - Types and Features",
        "text": (
            "UCB Bank offers the following credit cards: "
            "1. VISA Classic Card - unsecured credit limit BDT 20,000 to 49,000 (global limit), "
            "EMV contactless dual currency, 0% interest up to 24 months on EMI at 450+ merchants, "
            "maximum 45 days interest-free period. "
            "2. Master Classic Card - same features as Visa Classic with Mastercard network. "
            "3. VISA Gold Card - unsecured credit limit BDT 50,000 to 1,49,000, "
            "unlimited free access to UCB Imperial Airport Lounges. "
            "4. Master Gold Card - same as Visa Gold on Mastercard network. "
            "5. VISA Platinum Card - credit limit BDT 1,50,000 to 10,00,000 (secured up to BDT 25,00,000), "
            "Priority Pass membership for 1,800+ airport lounges worldwide, "
            "BOGO deals at major 5-star hotels. "
            "6. Master Platinum Card - same as Visa Platinum on Mastercard network. "
            "7. VISA Signature Card - premium card with exclusive benefits. "
            "8. Master World Card - premium Mastercard with exclusive benefits. "
            "All cards: annual fee waiver with 18 POS/e-commerce transactions, "
            "free supplementary card, free UPAY mobile wallet top-up."
        ),
        "url": "https://www.ucb.com.bd/cards",
        "language": "english",
        "source": "faq_bridge",
    },
    {
        "title": "UCB Credit Card - How to Apply and Requirements",
        "text": (
            "To apply for a UCB credit card, visit any UCB branch or apply online. "
            "Required documents for credit card application: "
            "valid National ID (NID) or passport, recent passport-size photograph, "
            "proof of income (salary certificate or bank statement), "
            "utility bill for address proof. "
            "UCB offers Classic, Gold, Platinum, Signature, and World credit cards "
            "on both Visa and Mastercard networks. "
            "All UCB credit cards come with: "
            "0% interest EMI for up to 24 months at 450+ partner merchants, "
            "45 days interest-free period, free add money to UPAY wallet, "
            "annual fee waiver by 18 POS or e-commerce transactions."
        ),
        "url": "https://www.ucb.com.bd/cards/documentation-requirements",
        "language": "english",
        "source": "faq_bridge",
    },
    {
        "title": "UCB Debit Card - Features and How to Get One",
        "text": (
            "UCB Bank offers the UCB Dual Currency Debit Card (Visa and Mastercard). "
            "Features: accepted at all ATMs and POS terminals displaying Visa and Mastercard logos worldwide. "
            "Wide network of UCB, VISA and Mastercard ATMs across Bangladesh and internationally. "
            "USD endorsement required on valid passport for international use. "
            "To get a UCB debit card, open a savings or current account at any UCB branch. "
            "The debit card is linked directly to your UCB bank account. "
            "UCB also offers a Business Debit Card and Payroll Card for corporate customers. "
            "If your debit card is lost or stolen, immediately call UCB 24x7 helpline: 16419."
        ),
        "url": "https://www.ucb.com.bd/banking/retail-banking/debit-card",
        "language": "english",
        "source": "faq_bridge",
    },
    {
        "title": "UCB Customer Care Helpline and Contact Information",
        "text": (
            "UCB Bank Customer Care Contact: "
            "24x7 Call Center Hotline (Bangladesh): 16419. "
            "Overseas: +88 096 100 16419 or +88 096 999 16419. "
            "Complaint email: complaint@ucb.com.bd. "
            "Service Quality email: sq@ucb.com.bd. "
            "Complaint Management Wing PABX: +880-02 55668070 Ext-5554 or Ext-2535. "
            "Zonal Office Chittagong PABX: +880-02 55668070 Ext.6250. "
            "Zonal Office Rajshahi: +88-0721-774030, email: rjh.zonal@ucb.com.bd. "
            "Online complaint form available at UCB website: ucb.com.bd/support/complaint-cell."
        ),
        "url": "https://www.ucb.com.bd/support/complaint-cell",
        "language": "english",
        "source": "faq_bridge",
    },
    {
        "title": "UCB Credit Card Security and Safety Tips",
        "text": (
            "UCB credit and debit card security features: "
            "EMV chip technology for enhanced security. "
            "OTP (One-Time Password) sent to registered mobile number for online transactions. "
            "3D Secure (3DS) authentication for e-commerce purchases. "
            "To protect your card: never share your PIN or OTP with anyone, "
            "check your statement regularly for unauthorized transactions, "
            "report lost or stolen cards immediately by calling 16419. "
            "To block your card if lost or stolen, call UCB 24x7 Call Center at 16419 immediately."
        ),
        "url": "https://www.ucb.com.bd/cards/card-security",
        "language": "english",
        "source": "faq_bridge",
    },
    {
        "title": "UCB Prepaid Card - Features",
        "text": (
            "UCB Bank offers a Prepaid Card for customers who want a prepaid payment solution. "
            "The prepaid card can be used at ATMs and POS terminals. "
            "It is available under retail banking products. "
            "For the latest offers on prepaid cards, visit UCB's prepaid card page."
        ),
        "url": "https://www.ucb.com.bd/banking/retail-banking/prepaid-card",
        "language": "english",
        "source": "faq_bridge",
    },
]


def make_sparse_vector(text: str) -> tuple[list[int], list[float]]:
    from collections import Counter
    tokens = text.lower().split()
    token_counts = Counter(tokens)
    total = len(tokens)
    idx_val: dict[int, float] = {}
    for token, count in token_counts.items():
        idx = abs(hash(token)) % SPARSE_VECTOR_SIZE
        tf = math.log(1 + count / max(total, 1))
        idx_val[idx] = idx_val.get(idx, 0.0) + tf
    return list(idx_val.keys()), list(idx_val.values())


def main():
    from sentence_transformers import SentenceTransformer
    from qdrant_client import QdrantClient
    from qdrant_client.models import PointStruct, SparseVector

    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL, device="cpu")

    print(f"Connecting to Qdrant: {QDRANT_URL}")
    client = QdrantClient(url=QDRANT_URL, timeout=30)

    points = []
    for chunk in CARD_FAQ_CHUNKS:
        text = chunk["text"]
        prefixed = f"passage: {text}"
        dense_vec = model.encode(prefixed, normalize_embeddings=True).tolist()
        idxs, vals = make_sparse_vector(text)

        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector={
                    "dense": dense_vec,
                    "sparse": SparseVector(indices=idxs, values=vals),
                },
                payload={
                    "text": text,
                    "title": chunk["title"],
                    "url": chunk["url"],
                    "language": chunk["language"],
                    "source": chunk["source"],
                },
            )
        )
        print(f"  Prepared: {chunk['title'][:60]}")

    print(f"\nUpserting {len(points)} card FAQ chunks...")
    client.upsert(collection_name=QDRANT_COLLECTION, points=points)
    print("Done!")

    info = client.get_collection(QDRANT_COLLECTION)
    print(f"Collection now has {info.points_count} total vectors.")


if __name__ == "__main__":
    main()
