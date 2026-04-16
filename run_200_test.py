"""
run_200_test.py
---------------
Runs all 200 test questions against the chunk DB using BM25 retrieval.
Does NOT require Qdrant or FastAPI to be running.
Outputs a full diagnostic report.
"""

import json
import math
import re
import textwrap
from collections import Counter
from rank_bm25 import BM25Okapi

CHUNKS_PATH = "data/processed/ucb_chunks.json"
REPORT_PATH = "test_report_200.txt"
TOP_K = 10

# Chunks whose URLs are high-frequency "noise" — they match everything
# because they contain boilerplate menus + T&C text with all UCB keywords
NOISE_URLS = {
    "https://www.ucb.com.bd/cards/reward-program-and-partners",
    "https://www.ucb.com.bd/cards/terms-and-conditions",
    "https://www.ucb.com.bd/banking/retail-banking/sms-banking",
    "https://www.ucb.com.bd/banking/retail-banking/24-7-customer-service",
}

# Map each question number to URL keyword(s) the correct chunk should contain
# (used to detect WRONG CHUNK vs CORRECT CHUNK)
EXPECTED_URL_HINTS = {
    # General
    1:  ["profile", "corporate", "about"],
    2:  ["profile", "corporate", "about"],
    3:  ["profile", "corporate", "about", "contact"],
    4:  ["profile", "corporate", "about", "branch"],
    5:  ["profile", "corporate", "about", "atm"],
    6:  ["profile", "corporate", "about"],
    7:  ["profile", "corporate", "about"],
    8:  ["profile", "corporate", "about"],
    9:  ["profile", "corporate", "about", "swift", "offshore"],
    10: ["profile", "corporate", "about"],
    11: ["contact", "customer-service", "complaint"],
    12: ["contact", "customer-service", "complaint"],
    13: ["profile", "corporate", "about"],
    14: ["profile", "corporate", "about"],
    15: ["profile", "corporate", "about"],
    # Savings/Deposits
    16: ["savings", "account", "retail-banking"],
    17: ["savings", "account", "retail-banking", "nrb"],
    18: ["earning-plus", "fixed-deposit", "uepfd", "fd"],
    19: ["earning-plus", "fixed-deposit", "uepfd"],
    20: ["earning-plus", "fixed-deposit", "uepfd"],
    21: ["earning-plus", "fixed-deposit", "uepfd"],
    22: ["earning-plus", "fixed-deposit", "uepfd"],
    23: ["earning-plus", "fixed-deposit", "uepfd"],
    24: ["earning-plus", "fixed-deposit", "senior"],
    25: ["earning-plus", "fixed-deposit", "senior"],
    26: ["earning-plus", "fixed-deposit", "uepfd"],
    27: ["earning-plus", "fixed-deposit", "uepfd"],
    28: ["earning-plus", "fixed-deposit", "uepfd"],
    29: ["earning-plus", "fixed-deposit", "uepfd"],
    30: ["super-flex", "dps", "superflex"],
    31: ["super-flex", "dps", "superflex"],
    32: ["super-flex", "dps", "superflex"],
    33: ["super-flex", "dps", "superflex"],
    34: ["super-flex", "dps", "superflex"],
    35: ["dps", "deposit"],
    # Credit Cards
    36: ["cards", "credit-card"],
    37: ["cards", "classic"],
    38: ["cards", "credit-card", "features"],
    39: ["cards", "emi", "features"],
    40: ["cards", "emi", "features"],
    41: ["cards", "emi", "features"],
    42: ["cards", "classic", "features"],
    43: ["cards", "gold", "features"],
    44: ["cards", "platinum", "lounge", "features"],
    45: ["cards", "lounge", "priority-pass"],
    46: ["cards", "lounge", "priority-pass"],
    47: ["cards", "platinum", "hotel", "features"],
    48: ["cards", "signature", "cashback", "features"],
    49: ["cards", "signature", "features"],
    50: ["cards", "signature", "lounge", "features"],
    51: ["cards", "signature", "features"],
    52: ["cards", "world", "mastercard", "features"],
    53: ["cards", "world", "mastercard", "features"],
    54: ["cards", "business", "features"],
    55: ["cards", "business", "debit"],
    56: ["cards", "business", "debit"],
    57: ["upay", "cards"],
    # Debit Cards
    58: ["debit-card", "dual-currency"],
    59: ["debit-card", "schedule", "charges", "fees"],
    60: ["debit-card", "schedule", "charges", "fees"],
    61: ["debit-card", "youngster", "charges"],
    62: ["debit-card", "ayma", "charges"],
    63: ["debit-card", "prothom", "charges"],
    64: ["debit-card", "charges", "atm"],
    65: ["debit-card", "charges", "npsb"],
    66: ["debit-card", "charges", "international"],
    67: ["debit-card", "charges", "markup"],
    68: ["debit-card", "takapay"],
    69: ["debit-card", "takapay"],
    70: ["debit-card", "charges", "pin"],
    71: ["debit-card", "charges", "balance"],
    72: ["debit-card", "contact", "helpline"],
    # Prepaid Cards
    73: ["prepaid"],
    74: ["prepaid"],
    75: ["prepaid"],
    76: ["prepaid"],
    77: ["prepaid"],
    78: ["prepaid"],
    79: ["prepaid"],
    80: ["prepaid", "upay"],
    # Loans
    81: ["personal-loan"],
    82: ["personal-loan"],
    83: ["personal-loan"],
    84: ["personal-loan"],
    85: ["personal-loan"],
    86: ["personal-loan"],
    87: ["personal-loan"],
    88: ["personal-loan", "nrb"],
    89: ["personal-loan", "retail-banking"],
    90: ["sme", "loan"],
    91: ["arfl", "agri", "rural"],
    92: ["arfl", "agri", "rural"],
    93: ["arfl", "agri", "rural"],
    94: ["arfl", "agri", "rural"],
    95: ["arfl", "agri", "rural"],
    96: ["corporate", "loan"],
    97: ["corporate", "export", "trade"],
    98: ["corporate", "export", "pre-shipment", "trade-finance"],
    99: ["corporate", "export", "post-shipment", "trade-finance"],
    100: ["corporate", "syndication", "loan"],
    # NRB
    101: ["nrb"],
    102: ["nrb", "savings"],
    103: ["nrb", "savings"],
    104: ["nrb", "savings"],
    105: ["nrb"],
    106: ["nrb", "dps"],
    107: ["nrb", "dps"],
    108: ["nrb", "dps"],
    109: ["nrb", "dps"],
    110: ["nrb"],
    # Lounge
    111: ["lounge", "cards"],
    112: ["lounge", "signature"],
    113: ["lounge", "world"],
    114: ["lounge", "platinum"],
    115: ["lounge", "gold"],
    116: ["lounge", "classic"],
    117: ["lounge", "business"],
    118: ["lounge", "platinum"],
    119: ["lounge", "rfcd", "erq"],
    120: ["lounge", "prepaid", "asset", "privilege"],
    121: ["lounge", "imperial", "debit"],
    # Imperial
    122: ["imperial"],
    123: ["imperial"],
    124: ["imperial"],
    125: ["imperial"],
    126: ["imperial"],
    127: ["imperial"],
    128: ["imperial"],
    129: ["imperial"],
    130: ["imperial", "signature"],
    # Islamic
    131: ["taqwa", "islamic"],
    132: ["taqwa", "islamic"],
    133: ["taqwa", "al-wadia", "current"],
    134: ["taqwa", "al-wadia"],
    135: ["taqwa", "al-wadia"],
    136: ["taqwa", "mudaraba", "savings"],
    137: ["taqwa", "mudaraba", "term"],
    138: ["taqwa", "mudaraba", "profit"],
    139: ["taqwa", "mudaraba", "profit", "isr"],
    140: ["taqwa", "islamic", "loan", "finance"],
    # AYMA
    141: ["ayma"],
    142: ["ayma", "savings"],
    143: ["ayma", "savings"],
    144: ["ayma", "savings"],
    145: ["ayma", "lending"],
    146: ["ayma", "dipti"],
    147: ["ayma", "jyoti"],
    148: ["ayma"],
    149: ["ayma", "locker"],
    150: ["ayma", "sanchay"],
    # SME/Agent
    151: ["sme", "sonirvor", "current"],
    152: ["sme", "sonirvor", "current"],
    153: ["sme", "sonirvor"],
    154: ["agent-banking"],
    155: ["agent-banking"],
    156: ["agent-banking"],
    157: ["agent-banking", "contact"],
    158: ["agent-banking", "school"],
    # Digital
    159: ["unet", "internet-banking"],
    160: ["unet", "internet-banking", "corporate"],
    161: ["upay"],
    162: ["unet", "internet-banking", "transfer"],
    163: ["unet", "internet-banking", "digital"],
    164: ["pos", "acquiring"],
    165: ["pos", "acquiring", "pci"],
    166: ["pos", "nfc"],
    # Rates/Contact
    167: ["rates", "deposit"],
    168: ["rates", "lending"],
    169: ["rates", "fx", "exchange"],
    170: ["rates", "schedule", "charges"],
    171: ["contact", "customer-service", "complaint"],
    172: ["contact", "customer-service", "complaint", "card-features"],
    173: ["contact", "customer-service", "complaint"],
    174: ["contact", "customer-service", "cards"],
    175: ["contact", "customer-service", "complaint"],
    176: ["locker"],
    177: ["locker"],
    178: ["green", "csr", "sustainability"],
    179: ["csr", "sustainability"],
    180: ["csr", "sustainability"],
    181: ["csr", "sustainability", "annual-report"],
    # Tricky
    182: ["cards", "classic", "gold"],
    183: ["cards", "business", "debit"],
    184: ["earning-plus", "fixed-deposit", "uepfd"],
    185: ["super-flex", "dps"],
    186: ["prepaid"],
    187: ["taqwa", "mudaraba", "al-wadia"],
    188: ["lounge", "cards"],
    189: ["nrb", "dps", "super-flex"],
    190: ["personal-loan"],
    191: ["personal-loan"],
    # Banglish
    192: ["cards", "classic", "annual"],
    193: ["cards", "signature", "lounge"],
    194: ["personal-loan"],
    195: ["dps", "loan"],
    196: ["nrb"],
    197: ["taqwa", "islamic"],
    198: ["debit-card", "takapay", "npsb"],
    199: ["agent-banking"],
    200: ["ayma", "savings"],
}

# ── 200 test questions ───────────────────────────────────────────────────────

QUESTIONS = [
    # 1. GENERAL / ABOUT UCB
    (1,  "General", "What is UCB Bank?"),
    (2,  "General", "When was UCB Bank established?"),
    (3,  "General", "Where is UCB's head office located?"),
    (4,  "General", "How many branches does UCB have?"),
    (5,  "General", "How many ATMs does UCB operate?"),
    (6,  "General", "Who is the CEO of UCB Bank?"),
    (7,  "General", "Who is the Chairman of UCB?"),
    (8,  "General", "Is UCB listed on any stock exchange?"),
    (9,  "General", "What is UCB's SWIFT code?"),
    (10, "General", "What is UCB's official website?"),
    (11, "General", "What is UCB's general helpline number?"),
    (12, "General", "What email can I use to contact UCB?"),
    (13, "General", "How many employees does UCB have?"),
    (14, "General", "What banking segments does UCB operate in?"),
    (15, "General", "What is DSE30 Index and is UCB part of it?"),

    # 2. SAVINGS & DEPOSIT ACCOUNTS
    (16, "Savings/Deposits", "How do I open a savings account at UCB?"),
    (17, "Savings/Deposits", "What documents are needed to open a UCB savings account?"),
    (18, "Savings/Deposits", "What is UCB Earning Plus Fixed Deposit?"),
    (19, "Savings/Deposits", "What is the minimum deposit for UEPFD?"),
    (20, "Savings/Deposits", "What is the interest rate for a 1-year UEPFD?"),
    (21, "Savings/Deposits", "What is the interest rate for a 2-year UEPFD?"),
    (22, "Savings/Deposits", "What is the interest rate for a 3-year UEPFD?"),
    (23, "Savings/Deposits", "What happens to UEPFD interest rates for 4-10 year tenors?"),
    (24, "Savings/Deposits", "Do senior citizens get special interest rates on FD?"),
    (25, "Savings/Deposits", "What is the senior citizen age threshold at UCB?"),
    (26, "Savings/Deposits", "Can I do partial encashment of UEPFD?"),
    (27, "Savings/Deposits", "Does UEPFD auto-renew?"),
    (28, "Savings/Deposits", "Can I get a loan against my UEPFD?"),
    (29, "Savings/Deposits", "What is the credit facility percentage on UEPFD?"),
    (30, "Savings/Deposits", "What is UCB Super Flex DPS?"),
    (31, "Savings/Deposits", "What is the minimum installment for a 1-year Super Flex DPS?"),
    (32, "Savings/Deposits", "What is the maturity value of a 1-year Super Flex DPS at BDT 5000/month?"),
    (33, "Savings/Deposits", "What is the maturity value of a 10-year Super Flex DPS at BDT 1000/month?"),
    (34, "Savings/Deposits", "Does Super Flex DPS have a grace period for late installments?"),
    (35, "Savings/Deposits", "What other DPS products does UCB offer?"),

    # 3. CREDIT CARDS
    (36, "Credit Cards", "What credit cards does UCB offer?"),
    (37, "Credit Cards", "What is the credit limit for UCB Classic card?"),
    (38, "Credit Cards", "What is the interest-free period on UCB credit cards?"),
    (39, "Credit Cards", "Which UCB credit cards offer EMI?"),
    (40, "Credit Cards", "How many months can I get 0% EMI?"),
    (41, "Credit Cards", "How many merchants support UCB EMI?"),
    (42, "Credit Cards", "How can I get annual fee waiver on Classic card?"),
    (43, "Credit Cards", "What extra benefits does UCB Gold card offer over Classic?"),
    (44, "Credit Cards", "What lounge access does UCB Platinum card include?"),
    (45, "Credit Cards", "What is Priority Pass / Lounge Key?"),
    (46, "Credit Cards", "How many lounges does Priority Pass cover?"),
    (47, "Credit Cards", "What hotel benefits does UCB Platinum card have?"),
    (48, "Credit Cards", "What cashback does UCB Signature card offer?"),
    (49, "Credit Cards", "How many signing bonus points does Signature card give?"),
    (50, "Credit Cards", "How many free Priority Pass entries does Signature card include per year?"),
    (51, "Credit Cards", "Does Signature card offer Meet and Greet at the airport?"),
    (52, "Credit Cards", "What is the cashback rate on UCB World Mastercard for international hotels?"),
    (53, "Credit Cards", "What gift voucher does World card give on annual spend?"),
    (54, "Credit Cards", "What are the benefits of UCB Business Credit Card?"),
    (55, "Credit Cards", "Does UCB offer a Business Debit Card?"),
    (56, "Credit Cards", "What is the cash withdrawal limit on Business Debit Card?"),
    (57, "Credit Cards", "Can I use upay wallet with UCB credit cards?"),

    # 4. DEBIT CARDS
    (58, "Debit Cards", "What is UCB Dual Currency Debit Card?"),
    (59, "Debit Cards", "What is the annual fee for UCB debit card on retail savings account?"),
    (60, "Debit Cards", "Is the debit card free for salary/payroll accounts?"),
    (61, "Debit Cards", "What is the debit card fee for UCB Youngsters account?"),
    (62, "Debit Cards", "What is the debit card fee for UCB AYMA account?"),
    (63, "Debit Cards", "What is the debit card fee for UCB Prothom account?"),
    (64, "Debit Cards", "What is the cash withdrawal fee at UCB ATMs?"),
    (65, "Debit Cards", "What is the fee for withdrawing cash at NPSB network ATMs?"),
    (66, "Debit Cards", "What is the fee for international ATM cash withdrawal?"),
    (67, "Debit Cards", "What is the international transaction markup fee?"),
    (68, "Debit Cards", "What is the Takapay debit card and when was it launched?"),
    (69, "Debit Cards", "What is the issuance fee for Takapay debit card?"),
    (70, "Debit Cards", "What is the PIN replacement fee at branch?"),
    (71, "Debit Cards", "Is there a fee for balance inquiry at UCB ATMs?"),
    (72, "Debit Cards", "What is the debit card helpline number?"),

    # 5. PREPAID CARDS
    (73, "Prepaid Cards", "What is UCB International Prepaid Card?"),
    (74, "Prepaid Cards", "Do I need a bank account for UCB prepaid card?"),
    (75, "Prepaid Cards", "What is the maximum annual limit on UCB prepaid card?"),
    (76, "Prepaid Cards", "How long is the UCB prepaid card valid?"),
    (77, "Prepaid Cards", "Who is eligible for UCB prepaid card?"),
    (78, "Prepaid Cards", "What documents are needed for prepaid card?"),
    (79, "Prepaid Cards", "What other prepaid card products does UCB offer?"),
    (80, "Prepaid Cards", "What is UPAY-UCB Co-Branded Prepaid Card?"),

    # 6. LOANS
    (81,  "Loans", "What personal loan products does UCB offer?"),
    (82,  "Loans", "What is the maximum UCB personal loan amount?"),
    (83,  "Loans", "What is the repayment tenure for personal loan?"),
    (84,  "Loans", "Is UCB personal loan secured or unsecured?"),
    (85,  "Loans", "What is the minimum monthly income required for personal loan?"),
    (86,  "Loans", "What is the minimum income for self-employed applicants?"),
    (87,  "Loans", "What is the minimum age for personal loan?"),
    (88,  "Loans", "Can NRBs apply for UCB personal loan?"),
    (89,  "Loans", "What other retail loan products does UCB offer?"),
    (90,  "Loans", "What SME loan products does UCB have?"),
    (91,  "Loans", "What is UCB Agri and Rural Fast Loan?"),
    (92,  "Loans", "What is the loan limit for UCB ARFL?"),
    (93,  "Loans", "What is the maximum tenure for ARFL?"),
    (94,  "Loans", "Does ARFL require collateral?"),
    (95,  "Loans", "How quickly is ARFL processed?"),
    (96,  "Loans", "What corporate loan products does UCB offer?"),
    (97,  "Loans", "Does UCB offer export finance?"),
    (98,  "Loans", "What is pre-shipment finance?"),
    (99,  "Loans", "What is post-shipment finance?"),
    (100, "Loans", "Does UCB offer loan syndication?"),

    # 7. NRB BANKING
    (101, "NRB Banking", "What NRB banking products does UCB offer?"),
    (102, "NRB Banking", "Who is eligible for NRB savings account?"),
    (103, "NRB Banking", "What is the minimum opening balance for NRB savings account in urban areas?"),
    (104, "NRB Banking", "What is the minimum opening balance for NRB savings account in rural areas?"),
    (105, "NRB Banking", "Is account maintenance free for NRB accounts?"),
    (106, "NRB Banking", "What is UCB NRB DPS Plus?"),
    (107, "NRB Banking", "What is the minimum installment for NRB DPS Plus?"),
    (108, "NRB Banking", "What is the maturity value of 5-year NRB DPS Plus?"),
    (109, "NRB Banking", "What is the maturity value of 10-year NRB DPS Plus?"),
    (110, "NRB Banking", "What documents are needed for NRB account opening?"),

    # 8. AIRPORT LOUNGE ACCESS
    (111, "Lounge Access", "Which UCB cards give airport lounge access?"),
    (112, "Lounge Access", "How many companions can Signature card bring to international lounge?"),
    (113, "Lounge Access", "How many companions can World card bring to lounge?"),
    (114, "Lounge Access", "How many companions can Platinum card bring to international lounge?"),
    (115, "Lounge Access", "How many companions can Gold card bring to international lounge?"),
    (116, "Lounge Access", "Does Classic card give lounge access?"),
    (117, "Lounge Access", "How many companions can Business card bring to international lounge?"),
    (118, "Lounge Access", "What is the domestic lounge access for Platinum card?"),
    (119, "Lounge Access", "Does ERQ/RFCD card give lounge access?"),
    (120, "Lounge Access", "What is Asset Privilege Prepaid Card lounge benefit?"),
    (121, "Lounge Access", "What is Imperial Debit Card lounge access?"),

    # 9. IMPERIAL BANKING
    (122, "Imperial Banking", "What is UCB Imperial Banking?"),
    (123, "Imperial Banking", "What is the eligibility for UCB Imperial Banking option 1?"),
    (124, "Imperial Banking", "What is option 2 for Imperial Banking eligibility?"),
    (125, "Imperial Banking", "What is option 3 for Imperial Banking eligibility?"),
    (126, "Imperial Banking", "What are the benefits of UCB Imperial Banking?"),
    (127, "Imperial Banking", "Do Imperial customers get dedicated relationship managers?"),
    (128, "Imperial Banking", "What credit card fee waiver do Imperial customers get?"),
    (129, "Imperial Banking", "What lounge access do Imperial customers get?"),
    (130, "Imperial Banking", "How many free overseas lounge entries does Signature card give Imperial customers?"),

    # 10. ISLAMIC BANKING
    (131, "Islamic Banking", "What is UCB Islamic Banking?"),
    (132, "Islamic Banking", "What is the brand name for UCB Islamic Banking?"),
    (133, "Islamic Banking", "What is Al-Wadia Current Account?"),
    (134, "Islamic Banking", "What is the Shariah mode of Al-Wadia account?"),
    (135, "Islamic Banking", "Who is eligible for Al-Wadia account?"),
    (136, "Islamic Banking", "What is Mudaraba Savings Account?"),
    (137, "Islamic Banking", "What is Mudaraba Term Deposit?"),
    (138, "Islamic Banking", "How is profit calculated in Mudaraba accounts?"),
    (139, "Islamic Banking", "What is ISR in Islamic banking?"),
    (140, "Islamic Banking", "What Islamic loan products does UCB offer?"),

    # 11. AYMA WOMEN BANKING
    (141, "AYMA Women Banking", "What is UCB AYMA?"),
    (142, "AYMA Women Banking", "Who is eligible for AYMA savings account?"),
    (143, "AYMA Women Banking", "What is the minimum opening balance for AYMA savings in urban areas?"),
    (144, "AYMA Women Banking", "What is the minimum opening balance for AYMA savings in rural areas?"),
    (145, "AYMA Women Banking", "What loan products are available under AYMA?"),
    (146, "AYMA Women Banking", "What is UCB AYMA Dipti?"),
    (147, "AYMA Women Banking", "What is UCB AYMA Jyoti?"),
    (148, "AYMA Women Banking", "How many branches offer AYMA one-stop banking?"),
    (149, "AYMA Women Banking", "How many branches offer locker service under AYMA?"),
    (150, "AYMA Women Banking", "Does AYMA offer Sanchay Patra issuance?"),

    # 12. SME & AGENT BANKING
    (151, "SME/Agent Banking", "What is Sonirvor Current Account?"),
    (152, "SME/Agent Banking", "Who can open Sonirvor Current Account?"),
    (153, "SME/Agent Banking", "What interest does Sonirvor offer?"),
    (154, "SME/Agent Banking", "What agent banking services does UCB provide?"),
    (155, "SME/Agent Banking", "What are the daily cash deposit limits for savings account at agent banking?"),
    (156, "SME/Agent Banking", "What are the daily cash withdrawal limits for current account at agent banking?"),
    (157, "SME/Agent Banking", "What is the agent banking contact email?"),
    (158, "SME/Agent Banking", "What school banking services are available through agent banking?"),

    # 13. DIGITAL & INTERNET BANKING
    (159, "Digital Banking", "What is UCB Unet?"),
    (160, "Digital Banking", "Does UCB offer corporate internet banking?"),
    (161, "Digital Banking", "What is UCB upay?"),
    (162, "Digital Banking", "Can I do online transfers through UCB?"),
    (163, "Digital Banking", "What digital banking features does UCB offer?"),
    (164, "Digital Banking", "What is UCB's POS service?"),
    (165, "Digital Banking", "Is UCB's POS PCI-DSS certified?"),
    (166, "Digital Banking", "Does UCB support NFC payments?"),

    # 14. RATES, CHARGES & CONTACT
    (167, "Rates/Charges/Contact", "Where can I find UCB's current deposit rates?"),
    (168, "Rates/Charges/Contact", "Where can I find UCB's lending rates?"),
    (169, "Rates/Charges/Contact", "How often are UCB's foreign exchange rates updated?"),
    (170, "Rates/Charges/Contact", "Where can I find UCB's schedule of charges?"),
    (171, "Rates/Charges/Contact", "What is UCB's 24/7 call center number?"),
    (172, "Rates/Charges/Contact", "What is the overseas call center number for UCB?"),
    (173, "Rates/Charges/Contact", "What email can I use to send complaints to UCB?"),
    (174, "Rates/Charges/Contact", "What is UCB cards division email?"),
    (175, "Rates/Charges/Contact", "Where is UCB's Chittagong zonal office contact?"),
    (176, "Rates/Charges/Contact", "Does UCB have locker services?"),
    (177, "Rates/Charges/Contact", "In how many branches is UCB locker service available?"),
    (178, "Rates/Charges/Contact", "What is UCB's green banking initiative?"),
    (179, "Rates/Charges/Contact", "What are UCB's CSR focus areas?"),
    (180, "Rates/Charges/Contact", "How many CSR focus areas does UCB have?"),
    (181, "Rates/Charges/Contact", "Does UCB publish sustainability reports?"),

    # 15. TRICKY / EDGE CASES
    (182, "Tricky/Edge Cases", "What is the difference between UCB Classic and UCB Gold credit card?"),
    (183, "Tricky/Edge Cases", "Can I withdraw cash from a Business Debit Card and how much?"),
    (184, "Tricky/Edge Cases", "Is UEPFD partial encashment allowed?"),
    (185, "Tricky/Edge Cases", "What happens if I miss a Super Flex DPS installment?"),
    (186, "Tricky/Edge Cases", "Can I open a prepaid card without a UCB bank account?"),
    (187, "Tricky/Edge Cases", "What is the difference between Mudaraba and Al-Wadia accounts?"),
    (188, "Tricky/Edge Cases", "Which card gives the most lounge companions internationally?"),
    (189, "Tricky/Edge Cases", "What is the difference between NRB DPS Plus and Super Flex DPS?"),
    (190, "Tricky/Edge Cases", "Can a self-employed person get a UCB personal loan?"),
    (191, "Tricky/Edge Cases", "What is the minimum income for doctors applying for personal loan?"),

    # BONUS BANGLISH
    (192, "Banglish", "UCB er credit card er annual fee kivabe waive hobe?"),
    (193, "Banglish", "Signature card e koto jon guest niye lounge e jawa jay?"),
    (194, "Banglish", "Personal loan er jonno ki ki documents lagbe?"),
    (195, "Banglish", "UCB DPS e ki loan pawa jay?"),
    (196, "Banglish", "NRB account khulte ki ki documents lagbe?"),
    (197, "Banglish", "UCB Islamic banking ki halal?"),
    (198, "Banglish", "Takapay card ki NPSB e use kora jay?"),
    (199, "Banglish", "Agent banking e ki account khola jay?"),
    (200, "Banglish", "UCB AYMA account er minimum balance koto?"),
]


# ── helpers ──────────────────────────────────────────────────────────────────

def tokenize(text: str) -> list[str]:
    return re.sub(r"[^a-z0-9\s]", " ", text.lower()).split()


def snippet(text: str, max_chars: int = 220) -> str:
    text = text.strip().replace("\n", " ")
    return text[:max_chars] + "..." if len(text) > max_chars else text


def smart_verdict(qnum: int, top_url: str, top_score: float,
                   all_urls: list[str], all_scores: list[float]) -> tuple[str, str]:
    """
    Return (verdict_emoji_string, note) based on:
    - Whether top chunk URL is a known noise URL
    - Whether any of top-10 URLs match expected hint keywords
    - BM25 score of the best relevant chunk
    """
    hints = EXPECTED_URL_HINTS.get(qnum, [])

    # Find best non-noise chunk that matches expected URL hints
    best_relevant_score = 0.0
    best_relevant_url = ""
    for url, score in zip(all_urls, all_scores):
        if url in NOISE_URLS:
            continue
        url_lower = url.lower()
        if any(h in url_lower for h in hints):
            if score > best_relevant_score:
                best_relevant_score = score
                best_relevant_url = url

    top_is_noise = top_url in NOISE_URLS
    top_matches_hint = any(h in top_url.lower() for h in hints)

    if best_relevant_score == 0 and top_score < 3.0:
        return "❌ DATA MISSING", "No relevant chunk found in DB"

    if best_relevant_score == 0 and not top_matches_hint:
        return "❌ WRONG CHUNK", f"Top chunk unrelated. Best score: {top_score:.2f}"

    if top_is_noise:
        if best_relevant_score > 3.0:
            return "⚠️  NOISE TOP-1", f"Top is noise chunk. Relevant chunk exists at score {best_relevant_score:.2f}: {best_relevant_url}"
        else:
            return "❌ NOISE+WEAK", f"Top is noise and no strong relevant chunk found"

    if top_matches_hint and top_score >= 5.0:
        return "✅ CORRECT", f"Top chunk is relevant (score {top_score:.2f})"

    if top_matches_hint and top_score < 5.0:
        return "⚠️  WEAK CORRECT", f"Relevant chunk found but low score ({top_score:.2f})"

    if best_relevant_score > 5.0:
        return "⚠️  WRONG TOP-1", f"Top chunk off-topic but relevant exists at score {best_relevant_score:.2f}: {best_relevant_url}"

    return "❌ WRONG CHUNK", f"Top URL {top_url} doesn't match expected topic {hints}"


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    # Load chunks
    with open(CHUNKS_PATH, encoding="utf-8") as f:
        chunks = json.load(f)

    texts = [c.get("text", "") for c in chunks]
    tokenized = [tokenize(t) for t in texts]
    bm25 = BM25Okapi(tokenized)

    lines = []
    cat_stats: dict[str, dict] = {}

    header = "=" * 90
    lines.append(header)
    lines.append("UCB BANK RAG CHATBOT — 200-QUESTION TEST REPORT")
    lines.append(f"Total chunks in DB: {len(chunks)}")
    lines.append("NOTE: BM25 retrieval only (Qdrant offline). Verdicts based on URL relevance.")
    lines.append(header)

    current_category = None
    pass_count = fail_count = weak_count = 0

    for qnum, category, question in QUESTIONS:
        # BM25 retrieval
        query_tokens = tokenize(question)
        scores = bm25.get_scores(query_tokens)

        # Top-K indices sorted by score
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:TOP_K]
        top_score = scores[top_indices[0]]
        top_chunk = chunks[top_indices[0]]
        top_text = texts[top_indices[0]]
        top_url = top_chunk.get("url", "")

        all_urls = [chunks[i].get("url", "") for i in top_indices]
        all_scores = [scores[i] for i in top_indices]

        v, note = smart_verdict(qnum, top_url, top_score, all_urls, all_scores)

        # Tally
        if "✅" in v:
            pass_count += 1
        elif "⚠️" in v:
            weak_count += 1
        else:
            fail_count += 1

        # Category stats
        if category not in cat_stats:
            cat_stats[category] = {"pass": 0, "weak": 0, "fail": 0, "total": 0}
        cat_stats[category]["total"] += 1
        if "✅" in v:
            cat_stats[category]["pass"] += 1
        elif "⚠️" in v:
            cat_stats[category]["weak"] += 1
        else:
            cat_stats[category]["fail"] += 1

        # Section header
        if category != current_category:
            current_category = category
            lines.append("")
            lines.append("─" * 90)
            lines.append(f"  CATEGORY: {category.upper()}")
            lines.append("─" * 90)

        lines.append("")
        lines.append(f"[Q{qnum:03d}] {question}")
        lines.append(f"  Verdict        : {v}")
        lines.append(f"  Note           : {note}")
        lines.append(f"  Top chunk URL  : {top_url}")
        lines.append(f"  BM25 top score : {top_score:.3f}")

        # What the bot would actually answer
        if "✅" in v:
            lines.append(f"  What bot GETS  : \"{snippet(top_text, 200)}\"")
            lines.append(f"  What DB HAS    : (same — correct chunk retrieved)")
        elif "NOISE" in v or "WRONG" in v:
            # Find the best relevant chunk and show that instead
            hints = EXPECTED_URL_HINTS.get(qnum, [])
            relevant_text = ""
            relevant_url = ""
            relevant_score = 0.0
            for url, score, idx in zip(all_urls, all_scores, top_indices):
                if url not in NOISE_URLS and any(h in url.lower() for h in hints):
                    if score > relevant_score:
                        relevant_score = score
                        relevant_text = texts[idx]
                        relevant_url = url
            lines.append(f"  What bot GETS  : \"{snippet(top_text, 180)}\"  <-- WRONG/IRRELEVANT")
            if relevant_text:
                lines.append(f"  What DB HAS    : \"{snippet(relevant_text, 200)}\"")
                lines.append(f"  Correct URL    : {relevant_url}  (score {relevant_score:.2f})")
            else:
                lines.append(f"  What DB HAS    : NOT FOUND IN DB — data likely not scraped")
        elif "DATA MISSING" in v:
            lines.append(f"  What bot GETS  : FALLBACK RESPONSE (no relevant chunk)")
            lines.append(f"  What DB HAS    : DATA NOT IN DB")
        else:
            lines.append(f"  What bot GETS  : \"{snippet(top_text, 200)}\"")
            lines.append(f"  What DB HAS    : (same chunk, but low confidence)")

    # ── Summary ──────────────────────────────────────────────────────────────
    lines.append("")
    lines.append("=" * 90)
    lines.append("SUMMARY")
    lines.append("=" * 90)
    lines.append(f"  Total questions : 200")
    lines.append(f"  PASS  ✅ CORRECT      : {pass_count}  ({pass_count/2:.0f}%)")
    lines.append(f"  WARN  ⚠️  PARTIAL/WEAK : {weak_count}  ({weak_count/2:.0f}%)")
    lines.append(f"  FAIL  ❌ WRONG/MISSING: {fail_count}  ({fail_count/2:.0f}%)")
    lines.append("")
    lines.append("  BY CATEGORY:")
    lines.append(f"  {'Category':<30} {'Total':>5} {'✅':>5} {'⚠️':>5} {'❌':>5}")
    lines.append(f"  {'-'*55}")
    for cat, s in cat_stats.items():
        lines.append(
            f"  {cat:<30} {s['total']:>5} {s['pass']:>5} {s['weak']:>5} {s['fail']:>5}"
        )
    lines.append("=" * 90)

    report = "\n".join(lines)

    # Save to file
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report)

    # Print to console (encode safely for Windows terminal)
    safe_report = report.encode("ascii", errors="replace").decode("ascii")
    print(safe_report)
    print(f"\n\nReport saved to: {REPORT_PATH} (full Unicode version)")


if __name__ == "__main__":
    main()
