"""Scrape all missing URLs from UCB sitemap and embed into Qdrant."""
import requests, json, math, uuid, time
from bs4 import BeautifulSoup
from collections import Counter
import sys
sys.path.insert(0, '.')
from preprocessing.chunker import split_into_chunks, clean_text, detect_language
from config.settings import CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL, QDRANT_URL, QDRANT_COLLECTION, SPARSE_VECTOR_SIZE
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, SparseVector

missing_urls = [
    'https://www.ucb.com.bd/auction-notice',
    'https://www.ucb.com.bd/banking/agent-banking/about-agent-banking',
    'https://www.ucb.com.bd/banking/agent-banking/changes-and-transection-limits',
    'https://www.ucb.com.bd/banking/agent-banking/list-of-agents-terminated-on-basis-of=complaints-irregularities-or-non-compliance',
    'https://www.ucb.com.bd/banking/corporate-banking/house-building-finance',
    'https://www.ucb.com.bd/banking/corporate-banking/import-finance',
    'https://www.ucb.com.bd/banking/corporate-banking/industrial-loan',
    'https://www.ucb.com.bd/banking/corporate-banking/letter-of-credit',
    'https://www.ucb.com.bd/banking/corporate-banking/loan-syndication',
    'https://www.ucb.com.bd/banking/corporate-banking/long-term-debt-instrument',
    'https://www.ucb.com.bd/banking/corporate-banking/more-corporate-banking-loans',
    'https://www.ucb.com.bd/banking/corporate-banking/more-structured-finance-products-services',
    'https://www.ucb.com.bd/banking/corporate-banking/project-finance',
    'https://www.ucb.com.bd/banking/corporate-banking/syndicated-structured-finance',
    'https://www.ucb.com.bd/banking/corporate-banking/wealth-management',
    'https://www.ucb.com.bd/banking/corporate-banking/working-capital-finance',
    'https://www.ucb.com.bd/banking/nrb-banking/nrb-sms-banking',
    'https://www.ucb.com.bd/banking/nrb-banking/remittance-services',
    'https://www.ucb.com.bd/banking/sme-banking/prottoyi-monthly-deposit',
    'https://www.ucb.com.bd/banking/sme-banking/sabolombi-easy-account',
    'https://www.ucb.com.bd/banking/sme-banking/sms-banking-services',
    'https://www.ucb.com.bd/banking/sme-banking/ucb-dhrubo',
    'https://www.ucb.com.bd/banking/sme-banking/ucb-dipti',
    'https://www.ucb.com.bd/banking/sme-banking/ucb-durjoy',
    'https://www.ucb.com.bd/banking/sme-banking/ucb-jyoti',
    'https://www.ucb.com.bd/banking/sme-banking/ucb-onkur',
    'https://www.ucb.com.bd/banking/sme-banking/ucb-shopno',
    'https://www.ucb.com.bd/banking/sme-banking/ucb-sme-installment-loan-usil',
    'https://www.ucb.com.bd/banking/sme-banking/uddomi-fixed-deposit',
    'https://www.ucb.com.bd/cards/card-features/protect-your-card',
    'https://www.ucb.com.bd/cards/card-privileges',
    'https://www.ucb.com.bd/cards/equal-installment-plan',
    'https://www.ucb.com.bd/cards/master-classic-card',
    'https://www.ucb.com.bd/cards/master-gold-card',
    'https://www.ucb.com.bd/cards/master-platinum-card',
    'https://www.ucb.com.bd/cards/visa-classic-card',
    'https://www.ucb.com.bd/cards/visa-gold-card',
    'https://www.ucb.com.bd/cards/visa-platinum-card',
    'https://www.ucb.com.bd/career',
    'https://www.ucb.com.bd/career/get-in-touch',
    'https://www.ucb.com.bd/imperial/imperial-discount',
    'https://www.ucb.com.bd/know-ucb/about-us/audit-committee/akhter-matin-chaudhury-2',
    'https://www.ucb.com.bd/know-ucb/about-us/audit-committee/mr-atm-tahmiduzzaman-fcs-1',
    'https://www.ucb.com.bd/know-ucb/about-us/audit-committee/mr-touhid-shipar-rafiquzzaman-1',
    'https://www.ucb.com.bd/know-ucb/about-us/audit-committee/muhammed-shah-alam-1',
    'https://www.ucb.com.bd/know-ucb/about-us/audit-committee/profile',
    'https://www.ucb.com.bd/know-ucb/about-us/audit-committee/syed-kamruzzaman-1',
    'https://www.ucb.com.bd/know-ucb/about-us/audit-committee/syed-mohammed-nuruddin-1',
    'https://www.ucb.com.bd/know-ucb/about-us/board-of-directors/afroza-zaman',
    'https://www.ucb.com.bd/know-ucb/about-us/board-of-directors/akhter-matin-chaudhury',
    'https://www.ucb.com.bd/know-ucb/about-us/board-of-directors/asifuzzaman-chowdhury',
    'https://www.ucb.com.bd/know-ucb/about-us/board-of-directors/bashir-ahmed',
    'https://www.ucb.com.bd/know-ucb/about-us/board-of-directors/dr-aparup-chowdhury',
    'https://www.ucb.com.bd/know-ucb/about-us/board-of-directors/hajee-m-a-kalam',
    'https://www.ucb.com.bd/know-ucb/about-us/board-of-directors/hajee-yunus-ahmed',
    'https://www.ucb.com.bd/know-ucb/about-us/board-of-directors/mr-anisuzzaman-chowdhury',
    'https://www.ucb.com.bd/know-ucb/about-us/board-of-directors/mr-atm-tahmiduzzaman-fcs',
    'https://www.ucb.com.bd/know-ucb/about-us/board-of-directors/mr-bazal-ahmed-1',
    'https://www.ucb.com.bd/know-ucb/about-us/board-of-directors/mr-kanak-kanti-sen',
    'https://www.ucb.com.bd/know-ucb/about-us/board-of-directors/mr-m-a-sabur-2',
    'https://www.ucb.com.bd/know-ucb/about-us/board-of-directors/mr-mohammed-shawkat-jamil-1',
    'https://www.ucb.com.bd/know-ucb/about-us/board-of-directors/mr-touhid-shipar-rafiquzzaman',
    'https://www.ucb.com.bd/know-ucb/about-us/board-of-directors/mrs-rukhmila-zaman',
    'https://www.ucb.com.bd/know-ucb/about-us/board-of-directors/muhammed-shah-alam',
    'https://www.ucb.com.bd/know-ucb/about-us/board-of-directors/nurul-islam-chowdhury',
    'https://www.ucb.com.bd/know-ucb/about-us/board-of-directors/professor-dr-md-jonaid-shafiq',
    'https://www.ucb.com.bd/know-ucb/about-us/board-of-directors/profile?profile=1-mrs-rukhmila-zaman',
    'https://www.ucb.com.bd/know-ucb/about-us/board-of-directors/roxana-zaman',
    'https://www.ucb.com.bd/know-ucb/about-us/board-of-directors/syed-kamruzzaman',
    'https://www.ucb.com.bd/know-ucb/about-us/board-of-directors/syed-mohammed-nuruddin',
    'https://www.ucb.com.bd/know-ucb/about-us/executive-committee/asifuzzaman-chowdhury-1',
    'https://www.ucb.com.bd/know-ucb/about-us/executive-committee/bashir-ahmed-1',
    'https://www.ucb.com.bd/know-ucb/about-us/executive-committee/hajee-yunus-ahmed-1',
    'https://www.ucb.com.bd/know-ucb/about-us/executive-committee/mr-anisuzzaman-chowdhury-2',
    'https://www.ucb.com.bd/know-ucb/about-us/executive-committee/mr-atm-tahmiduzzaman-fcs-2',
    'https://www.ucb.com.bd/know-ucb/about-us/executive-committee/mr-bazal-ahmed',
    'https://www.ucb.com.bd/know-ucb/about-us/executive-committee/mr-m-a-sabur',
    'https://www.ucb.com.bd/know-ucb/about-us/executive-committee/nurul-islam-chowdhury-1',
    'https://www.ucb.com.bd/know-ucb/about-us/executive-committee/profile',
    'https://www.ucb.com.bd/know-ucb/about-us/nomination-and-remuneration-committee',
    'https://www.ucb.com.bd/know-ucb/about-us/nomination-and-remuneration-committee/dr-aparup-chowdhury-1',
    'https://www.ucb.com.bd/know-ucb/about-us/nomination-and-remuneration-committee/mr-asifuzzaman-chowdhury',
    'https://www.ucb.com.bd/know-ucb/about-us/nomination-and-remuneration-committee/mr-nurul-islam-chowdhury',
    'https://www.ucb.com.bd/know-ucb/about-us/nomination-and-remuneration-committee/mrs-roxana-zaman-1',
    'https://www.ucb.com.bd/know-ucb/about-us/nomination-and-remuneration-committee/professor-dr-md-jonaid-shafiq-1',
    'https://www.ucb.com.bd/know-ucb/about-us/risk-management-committee/mr-anisuzzaman-chowdhury-2',
    'https://www.ucb.com.bd/know-ucb/about-us/risk-management-committee/mr-atm-tahmiduzzaman-fcs-3',
    'https://www.ucb.com.bd/know-ucb/about-us/risk-management-committee/mr-kanak-kanti-sen-1',
    'https://www.ucb.com.bd/know-ucb/about-us/risk-management-committee/mr-m-a-sabur-2',
    'https://www.ucb.com.bd/know-ucb/about-us/risk-management-committee/mrs-afroza-zaman',
    'https://www.ucb.com.bd/know-ucb/about-us/risk-management-committee/mrs-roxana-zaman',
    'https://www.ucb.com.bd/know-ucb/about-us/risk-management-committee/profile',
    'https://www.ucb.com.bd/know-ucb/about-us/senior-management/mr-abul-alam-ferdous',
    'https://www.ucb.com.bd/know-ucb/about-us/senior-management/mr-arif-quadri',
    'https://www.ucb.com.bd/know-ucb/about-us/senior-management/mr-habibur-rahman',
    'https://www.ucb.com.bd/know-ucb/about-us/senior-management/mr-md-abdullah-al-mamoon',
    'https://www.ucb.com.bd/know-ucb/about-us/senior-management/mr-md-sohrab-mustafa',
    'https://www.ucb.com.bd/know-ucb/about-us/senior-management/mr-mohammad-mamdudur-rashid',
    'https://www.ucb.com.bd/know-ucb/about-us/senior-management/mr-mohammed-shawkat-jamil',
    'https://www.ucb.com.bd/know-ucb/about-us/senior-management/mr-n-mustafa-tarek',
    'https://www.ucb.com.bd/know-ucb/about-us/senior-management/mr-nabil-mustafizur-rahman',
    'https://www.ucb.com.bd/know-ucb/about-us/senior-management/mr-syed-faridul-islam',
    'https://www.ucb.com.bd/know-ucb/about-us/senior-management/profile',
    'https://www.ucb.com.bd/know-ucb/about-us/shariah-supervisory-committee/dr-mohammad-manjurur-rahman',
    'https://www.ucb.com.bd/know-ucb/about-us/shariah-supervisory-committee/dr-mohammed-nasir-uddin-azhary-',
    'https://www.ucb.com.bd/know-ucb/about-us/shariah-supervisory-committee/hajee-m-a-kalam',
    'https://www.ucb.com.bd/know-ucb/about-us/shariah-supervisory-committee/mr-bazal-ahmed',
    'https://www.ucb.com.bd/know-ucb/about-us/shariah-supervisory-committee/mr-mohammed-shawkat-jamil',
    'https://www.ucb.com.bd/know-ucb/about-us/shariah-supervisory-committee/professor-dr--a-f-m-akbar-hossain',
    'https://www.ucb.com.bd/know-ucb/about-us/shariah-supervisory-committee/professor-dr-k-m-saiful-islam-khan',
    'https://www.ucb.com.bd/know-ucb/about-us/shariah-supervisory-committee/professor-dr-mohammad-abdur-rashid',
    'https://www.ucb.com.bd/know-ucb/about-us/shariah-supervisory-committee/profile',
    'https://www.ucb.com.bd/know-ucb/corporate-social-responsibility/photo-gallery',
    'https://www.ucb.com.bd/know-ucb/disclosures-on-risk-based-capital',
    'https://www.ucb.com.bd/know-ucb/investor-relations/credit-rating-report',
    'https://www.ucb.com.bd/know-ucb/investor-relations/directors-report',
    'https://www.ucb.com.bd/know-ucb/investor-relations/report-on-corporate-governance',
    'https://www.ucb.com.bd/know-ucb/investor-relations/right-share-issue',
    'https://www.ucb.com.bd/know-ucb/reports-and-statements',
    'https://www.ucb.com.bd/know-ucb/reports-and-statements/credit-rating-report',
    'https://www.ucb.com.bd/know-ucb/reports-and-statements/financial-statements',
    'https://www.ucb.com.bd/learn-about-money',
    'https://www.ucb.com.bd/learn-about-money/banking-tips',
    'https://www.ucb.com.bd/learn-about-money/banking-tips/age-limit',
    'https://www.ucb.com.bd/learn-about-money/banking-tips/customer-segment',
    'https://www.ucb.com.bd/learn-about-money/banking-tips/min-income',
    'https://www.ucb.com.bd/learn-about-money/banking-tips/minimum-service',
    'https://www.ucb.com.bd/learn-about-money/banking-tips/purpose',
    'https://www.ucb.com.bd/learn-about-money/investment-tips',
    'https://www.ucb.com.bd/learn-about-money/investment-tips/dps',
    'https://www.ucb.com.bd/learn-about-money/investment-tips/fdr',
    'https://www.ucb.com.bd/learn-about-money/investment-tips/investment-to-meet-financial-goals',
    'https://www.ucb.com.bd/learn-about-money/investment-tips/mix-investments',
    'https://www.ucb.com.bd/learn-about-money/investment-tips/risky-investments',
    'https://www.ucb.com.bd/learn-about-money/investment-tips/shanchay-patra',
    'https://www.ucb.com.bd/learn-about-money/investment-tips/stocks',
    'https://www.ucb.com.bd/learn-about-money/money-management-and-budgeting-tips',
    'https://www.ucb.com.bd/learn-about-money/money-management-and-budgeting-tips/for-families',
    'https://www.ucb.com.bd/learn-about-money/money-management-and-budgeting-tips/for-new-job-entrants',
    'https://www.ucb.com.bd/learn-about-money/money-management-and-budgeting-tips/for-students',
    'https://www.ucb.com.bd/news-and-events',
    'https://www.ucb.com.bd/news-and-events/bank-note-identification',
    'https://www.ucb.com.bd/news-and-events/latest-tvc',
    'https://www.ucb.com.bd/news-and-events/national-mourning-day',
    'https://www.ucb.com.bd/news-and-events/photo-gallery',
    'https://www.ucb.com.bd/news-and-events/press-ad',
    'https://www.ucb.com.bd/news-and-events/press-release/breast-cancer-awareness-campaign-dhaka',
    'https://www.ucb.com.bd/news-and-events/press-release/cheque-handover-to-honourable-prime-minister-for-mujib-borsho',
    'https://www.ucb.com.bd/news-and-events/press-release/dhaka-metropolitan-police-commissioner-and-managing-director-of-united-commercial-bank-limited-recently-sat-on-a-discussion-regarding-development-process-of-e-traffic-prosecution',
    'https://www.ucb.com.bd/news-and-events/press-release/donation-for-prime-minister-s-shikkha-sohayota-trust-and-suchona-foundation-',
    'https://www.ucb.com.bd/news-and-events/press-release/inauguration-of-ucb-taqwa-islamic-banking-window-of-united-commercial-bank-limited',
    'https://www.ucb.com.bd/news-and-events/press-release/ucb-and-incentive-remit-launched-real-time-remittance-service-in-bangladesh',
    'https://www.ucb.com.bd/news-and-events/press-release/ucb-card-holders-get-special-benefits-at-bangkok-hospital',
    'https://www.ucb.com.bd/news-and-events/press-release/ucb-held-strategy-meet-and-budget-session-for-2019',
    'https://www.ucb.com.bd/news-and-events/press-release/ucb-organized-dua-and-milad-mahfil-in-remembrance-of-6th-death-anniversary-of-late-akhtaruzzaman-chowdhury',
    'https://www.ucb.com.bd/news-and-events/press-release/ucb-signs-agreement-with-bangladesh-police-kallyan-trust-security-service',
    'https://www.ucb.com.bd/news-and-events/press-release/united-commercial-bank-limited-and-pragati-life-insurance-limited-signs-an-agreement',
    'https://www.ucb.com.bd/news-and-events/press-release/united-commercial-bank-limited-completed-the-compliance-of-pci-dss-certification',
    'https://www.ucb.com.bd/news-and-events/press-release/united-commercial-bank-limited-donates-blankets-to-honourable-prime-minister-s-relief-fund',
    'https://www.ucb.com.bd/news-and-events/press-release/united-commercial-bank-limited-donates-to-honourable-prime-minister-s-relief-fund',
    'https://www.ucb.com.bd/news-and-events/press-release/united-commercial-bank-limited-holds-414th-meeting-of-the-board-of-directors',
    'https://www.ucb.com.bd/news-and-events/press-release/united-commercial-bank-limited-inaugurates-special-service-counter-for-chinese-nationals-in-bangladesh',
    'https://www.ucb.com.bd/news-and-events/press-release/united-commercial-bank-limited-launches-agent-banking',
    'https://www.ucb.com.bd/news-and-events/press-release/united-commercial-bank-limited-opens-171st-branch-at-balasur-munshiganj',
    'https://www.ucb.com.bd/news-and-events/press-release/united-commercial-bank-limited-opens-172nd-branch-at-katiadi-kishoregonj',
    'https://www.ucb.com.bd/news-and-events/press-release/united-commercial-bank-limited-opens-173rd-branch-at-manikganj',
    'https://www.ucb.com.bd/news-and-events/press-release/united-commercial-bank-limited-opens-174th-branch-at-chandanaish-chittagong',
    'https://www.ucb.com.bd/news-and-events/press-release/united-commercial-bank-limited-opens-175th-branch-at-jamal-khan-chittagong',
    'https://www.ucb.com.bd/news-and-events/press-release/united-commercial-bank-limited-opens-176th-branch-at-anowara-sadar-chittagong',
    'https://www.ucb.com.bd/news-and-events/press-release/united-commercial-bank-limited-opens-177th-branch-at-patiya-chittagong',
    'https://www.ucb.com.bd/news-and-events/press-release/united-commercial-bank-limited-opens-179th-branch-at-kerani-hat-chattogram',
    'https://www.ucb.com.bd/news-and-events/press-release/united-commercial-bank-limited-opens-180th-branch-at-shyamoli-dhaka',
    'https://www.ucb.com.bd/news-and-events/press-release/united-commercial-bank-limited-opens-182nd-nilphamari-and-183rd-boda-panchagarh-branch',
    'https://www.ucb.com.bd/news-and-events/press-release/united-commercial-bank-limited-opens-184th-branch-at-kala-meah-bazar-chattogram',
    'https://www.ucb.com.bd/news-and-events/press-release/united-commercial-bank-limited-opens-185th-branch-at-fakirhat-bagerhat',
    'https://www.ucb.com.bd/news-and-events/press-release/united-commercial-bank-limited-opens-186th-sharaf-bhata-branch-chattogram',
    'https://www.ucb.com.bd/news-and-events/press-release/united-commercial-bank-limited-opens-187th-kashinathpur-branch',
    'https://www.ucb.com.bd/news-and-events/press-release/united-commercial-bank-limited-organizes-a-five-day-advance-level-workshop-on-foreign-trade--foreign-exchange',
    'https://www.ucb.com.bd/news-and-events/press-release/united-commercial-bank-limited-organizes-annual-business-conference-2019',
    'https://www.ucb.com.bd/news-and-events/press-release/united-commercial-bank-limited-organizes-breast-cancer-awareness-campaign-in-chittagong',
    'https://www.ucb.com.bd/news-and-events/press-release/united-commercial-bank-limited-organizes-integrity-related-workshop-at-chattogram',
    'https://www.ucb.com.bd/news-and-events/press-release/united-commercial-bank-limited-presents-high-tea-women-leadership-experience-and-empowerment-program-in-collaboration-with-radission-blu-chittagong-bay-view',
    'https://www.ucb.com.bd/news-and-events/press-release/united-commercial-bank-limited-signs-agreement-with-a2i-access-to-information-project',
    'https://www.ucb.com.bd/news-and-events/press-release/united-commercial-bank-limited-signs-agreement-with-pathao-limited',
    'https://www.ucb.com.bd/news-and-events/press-release/united-commercial-bank-limited-signs-agreement-with-praava-health-bangladesh-limited',
    'https://www.ucb.com.bd/news-and-events/press-release/united-commercial-bank-limited-signs-agreement-with-sonali-bank-limited',
    'https://www.ucb.com.bd/news-and-events/press-release/united-commercial-bank-limited-signs-agreement-with-teletalk-bangladesh-limited',
    'https://www.ucb.com.bd/news-and-events/press-release/united-commercial-bank-limited-ucb-and-khulna-metropolitan-police-kmp-recently-signed-an-agreement-on-traffic-case-fine-payment',
    'https://www.ucb.com.bd/news-and-events/press-release/united-commercial-bank-limited-ucb-distributes-warm-clothes-at-panchagarh',
    'https://www.ucb.com.bd/news-and-events/press-release/united-commercial-bank-limited-ucb-donated-to-prime-minister-s-relief-fund',
    'https://www.ucb.com.bd/news-and-events/press-release/united-commercial-bank-limited-ucb-has-been-awarded-for-best-presented-annual-reports-2016-',
    'https://www.ucb.com.bd/pos-acquiring-service/dual-sim-based-3g-pos-devices',
    'https://www.ucb.com.bd/pos-acquiring-service/e-commerce-online-payment-gateway',
    'https://www.ucb.com.bd/pos-acquiring-service/nfc-based-pos-terminal',
    'https://www.ucb.com.bd/pos-acquiring-service/real-time-online-emi-on-ucb-pos',
    'https://www.ucb.com.bd/rates/interest-rates',
    'https://www.ucb.com.bd/rates/schedule-of-charges',
    'https://www.ucb.com.bd/support/complaint-cell-form',
    'https://www.ucb.com.bd/ucb-ayma/ayma-products/ayma-cards',
    'https://www.ucb.com.bd/ucb-ayma/ayma-products/ayma-deposit/dps',
    'https://www.ucb.com.bd/ucb-ayma/ayma-products/ayma-deposit/fixed-deposit',
    'https://www.ucb.com.bd/ucb-ayma/ayma-products/ayma-entrepreneur/dipti',
    'https://www.ucb.com.bd/ucb-ayma/ayma-products/ayma-entrepreneur/jyoti',
    'https://www.ucb.com.bd/ucb-ayma/ayma-products/ayma-lending',
    'https://www.ucb.com.bd/ucb-taqwa/deposit/mudaraba-monthly-income-term-deposit',
    'https://www.ucb.com.bd/ucb-taqwa/deposit/mudaraba-savings-account',
    'https://www.ucb.com.bd/ucb-taqwa/deposit/mudaraba-term-deposit',
    'https://www.ucb.com.bd/ucb-taqwa/deposit/profit-mechanism-of-mudaraba-deposit-accounts',
    'https://www.ucb.com.bd/ucb-taqwa/imperial-deposit/mudaraba-imperial-savings-account',
    'https://www.ucb.com.bd/ucb-taqwa/imperial-deposit/profit-mechanism-of-mudaraba-deposit-accounts',
    'https://www.ucb.com.bd/ucb-taqwa/investment/corporate',
    'https://www.ucb.com.bd/ucb-taqwa/investment/retail',
    'https://www.ucb.com.bd/ucb-taqwa/investment/retail/auto',
    'https://www.ucb.com.bd/ucb-taqwa/investment/retail/home',
    'https://www.ucb.com.bd/ucb-taqwa/investment/retail/personal',
    'https://www.ucb.com.bd/ucb-taqwa/investment/sme',
    'https://www.ucb.com.bd/unet-connect',
    'https://www.ucb.com.bd/unet-internet-banking',
    'https://www.ucb.com.bd/upay',
    'https://www.ucb.com.bd/upay/customer-offers',
    'https://www.ucb.com.bd/upay/how-to',
    'https://www.ucb.com.bd/upay/limit-and-charges',
    'https://www.ucb.com.bd/upay/merchant-list',
    'https://www.ucb.com.bd/upay/notice',
    'https://www.ucb.com.bd/upay/registration-process',
    'https://www.ucb.com.bd/upay/security-tips',
]

headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}

# ── Step 1: Scrape ──────────────────────────────────────────────────────────
print(f"Scraping {len(missing_urls)} URLs...")
new_pages = []
skipped = 0

for i, url in enumerate(missing_urls):
    try:
        r = requests.get(url, headers=headers, timeout=12)
        soup = BeautifulSoup(r.text, 'html.parser')
        title = soup.find('title').get_text(strip=True) if soup.find('title') else ''
        for tag in soup(['script', 'style', 'nav', 'footer', 'header']):
            tag.decompose()
        section = soup.find('section') or soup.find('main') or soup.find('body')
        text = section.get_text(separator=' ', strip=True) if section else ''
        if len(text) > 50:
            new_pages.append({'url': url, 'title': title, 'text': text, 'language': 'en'})
            print(f"[{i+1}/{len(missing_urls)}] OK   {len(text):5d}c  {url.replace('https://www.ucb.com.bd','')}")
        else:
            skipped += 1
            print(f"[{i+1}/{len(missing_urls)}] SKIP {len(text):5d}c  {url.replace('https://www.ucb.com.bd','')}")
    except Exception as e:
        skipped += 1
        print(f"[{i+1}/{len(missing_urls)}] ERR         {url.replace('https://www.ucb.com.bd','')} - {e}")

# ── Step 2: Save to raw JSON ────────────────────────────────────────────────
with open('data/raw/ucb_raw.json', encoding='utf-8') as f:
    data = json.load(f)
data.extend(new_pages)
with open('data/raw/ucb_raw.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)
print(f"\nSaved. Added {len(new_pages)} pages. Skipped {skipped}. Total raw pages: {len(data)}")

# ── Step 3: Chunk ───────────────────────────────────────────────────────────
all_chunks = []
for page in new_pages:
    text = clean_text(page['text'])
    if not text.strip():
        continue
    for chunk in split_into_chunks(text, CHUNK_SIZE, CHUNK_OVERLAP):
        all_chunks.append({
            'text': chunk,
            'url': page['url'],
            'title': page['title'],
            'language': detect_language(text),
        })
print(f"Total chunks to embed: {len(all_chunks)}")

if not all_chunks:
    print("No chunks — all pages were JS-rendered with no extractable text.")
    sys.exit(0)

# ── Step 4: Embed ───────────────────────────────────────────────────────────
print("Loading embedding model...")
model = SentenceTransformer(EMBEDDING_MODEL, device='cpu')
texts = ['passage: ' + c['text'] for c in all_chunks]
vectors = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True, batch_size=32, show_progress_bar=True)
print("Embedding done.")

# ── Step 5: Upsert to Qdrant ────────────────────────────────────────────────
def make_sparse(text):
    tokens = text.lower().split()
    counts = Counter(tokens)
    total = len(tokens)
    idx_val = {}
    for tok, cnt in counts.items():
        idx = abs(hash(tok)) % SPARSE_VECTOR_SIZE
        tf = float(math.log(1 + cnt / max(total, 1)))
        idx_val[idx] = idx_val.get(idx, 0.0) + tf  # merge hash collisions
    return list(idx_val.keys()), list(idx_val.values())

client = QdrantClient(url=QDRANT_URL, timeout=60)
BATCH = 50
total_upserted = 0
for start in range(0, len(all_chunks), BATCH):
    batch = all_chunks[start:start + BATCH]
    points = []
    for i, c in enumerate(batch):
        idxs, vals = make_sparse(c['text'])
        points.append(PointStruct(
            id=str(uuid.uuid4()),
            vector={'dense': vectors[start + i].tolist(), 'sparse': SparseVector(indices=idxs, values=vals)},
            payload={'text': c['text'], 'url': c['url'], 'title': c['title'], 'language': c['language']},
        ))
    client.upsert(collection_name=QDRANT_COLLECTION, points=points)
    total_upserted += len(points)
    print(f"  Upserted batch {start//BATCH + 1}: {total_upserted}/{len(all_chunks)} points")

print(f"\nDone! Upserted {total_upserted} new vectors into Qdrant.")
