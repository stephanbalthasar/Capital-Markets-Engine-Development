# app.py
# EUCapML Case Tutor — University of Bayreuth
# - Free LLM via Groq (llama-3.1-8b/70b-instant): no credits or payments
# - Web retrieval from EUR-Lex, CURIA, ESMA, BaFin, Gesetze-im-Internet
# - Hidden model answer is authoritative; citations [1], [2] map to sources

import os
import re
import json
import hashlib
import pathlib
import fitz  # PyMuPDF
from typing import List, Dict, Tuple
from urllib.parse import quote_plus, urlparse
import numpy as np
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

import requests
from bs4 import BeautifulSoup

# ---------------- Build fingerprint (to verify latest deployment) ----------------
APP_HASH = hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()[:10]

# ---------------- Embeddings ----------------
@st.cache_resource(show_spinner=False)
def load_embedder():
    """
    Try a small sentence-transformer; if unavailable (e.g., install timeouts), fall back to TF-IDF.
    """
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        return ("sbert", model)
    except Exception:
        return ("tfidf", None)

def embed_texts(texts: List[str], backend):
    kind, model = backend
    if kind == "sbert":
        return model.encode(texts, normalize_embeddings=True)
    vec = TfidfVectorizer(stop_words="english")
    X = vec.fit_transform(texts)
    A = X.toarray()
    norms = np.linalg.norm(A, axis=1, keepdims=True) + 1e-12
    return A / norms

def cos_sim(a, b):
    return float(cosine_similarity(a.reshape(1, -1), b.reshape(1, -1))[0, 0])

def split_into_chunks(text: str, max_words: int = 180):
    words = text.split()
    chunks, cur = [], []
    for w in words:
        cur.append(w)
        if len(cur) >= max_words:
            chunks.append(" ".join(cur)); cur = []
    if cur: chunks.append(" ".join(cur))
    return chunks

# ---------------- Case & Model Answer (YOUR CONTENT) ----------------
CASE = """
Neon AG is a German stock company (Aktiengesellschaft), the shares of which have been admitted to trading on the regulated market of the Frankfurt stock exchange for a number of years. Gerry is Neon’s CEO (Vorstandsvorsitzender) and holds 25% of Neon’s shares. Gerry wants Neon to develop a new business strategy. For this, Neon would have to buy IP licences for 2.5 billion euros but has no means to afford this. Unicorn plc is a competitor of Neon’s based in the UK and owns licences of the type needed for Neon’s plans. After confidential negotiations, Unicorn, Neon, and Gerry in his personal capacity enter into a “Cooperation Framework Agreement” (“CFA”) which names all three as parties and which has the following terms:
1. Unicorn will transfer the licences to Neon by way of a capital contribution in kind (Sacheinlage). In return, Neon will increase its share capital by 30% and issue the new shares to Unicorn. The parties agree that the capital increase should take place within the next 6 months. 
2. Unicorn and Gerry agree that, once the capital increase is complete, they will pre-align major decisions impacting Neon’s business strategy. Where they cannot agree on a specific measure, Gerry agrees to follow Unicorn’s instructions when voting at a shareholder meeting of Neon.
As a result of the capital increase, Gerry will hold approximately 19% in Neon, and Unicorn 23%. Unicorn, Neon and Gerry know that the agreement will come as a surprise to Neon’s shareholders, in particular, because in previous public statements, Gerry had always stressed that he wanted Neon to remain independent. They expect that the new strategy is a “game-changer” for Neon and will change its strategic orientation permanently in a substantial way. 

Questions:
1. Does the conclusion of the CFA trigger capital market disclosure obligations for Neon? What is the timeframe for disclosure? Is there an option for Neon to delay disclosure?
2. Unicorn wants the new shares to be admitted to trading on the regulated market in Frankfurt. Does this require a prospectus under the Prospectus Regulation? What type of information in connection with the CFA would have to be included in such a prospectus?
3. What are the capital market law disclosure obligations that arise for Unicorn once the capital increase and admission to trading are complete and Unicorn acquires the new shares? Can Unicorn participate in Neon’s shareholder meetings if it does not comply with these obligations?

Note:
Your answer will not have to consider the SRD, §§ 111a–111c AktG, or EU capital market law that is not included in your permitted material. You may assume that Gerry and Neon have all corporate authorisations for the conclusion of the CFA and the capital increase.
"""

MODEL_ANSWER = """

1.  Question 1 requires a discussion of whether the conclusion of the CFA triggers an obligation to publish “inside information” pursuant to article 17(1) MAR. 
a)  On the facts of the case (i.e., new shareholder structure of Neon, combined influence of Gerry and Unicorn, substantial change of strategy, etc.), students would have to conclude that the conclusion of the CFA is inside information within the meaning of article 7(1)(a): 
aa) It relates to an issuer (Neon) and has not yet been made public.
bb) Even if the agreement depends on further implementing steps, it creates information of a precise nature within the meaning of article 7(2) MAR in that there is an event that has already occurred – the conclusion of the CFA –, which is sufficient even if one considered it only as an “intermediate step” of a “protracted process”. In addition, subsequent events – the capital increase – can “reasonably be expected to occur” and therefore also qualify as information of a precise nature. A good answer would discuss the “specificity” requirement under article 7(2) MAR and mention that pursuant to the ECJ decision in Lafonta, it is sufficient for the information to be sufficiently specific to constitute a basis on which to assess the effect on the price of the financial instruments, and that the only information excluded by the specificity requirement is information that is “vague or general”. Also, the information is something a reasonable investor would likely use, and therefore likely to have a significant effect on prices within the meaning of article 7(4) MAR.
cc) The information “directly concerns” the issuer in question. As a result, article 17(1) MAR requires Neon to “inform the public as soon as possible”. Students should mention that this allows issuers some time for fact-finding, but otherwise, immediate disclosure is required. Delay is only possible under article 17(4) MAR. However, there is nothing to suggest that Neon has a legitimate interest within the meaning of article 17(4)(a), and at any rate, given previous communication by Neon, a delay would be likely to mislead the public within the meaning of article 17(4)(b). Accordingly, a delay could not be justified under article 17(4) MAR.
Students are not expected to address §§ 33, 38 WpHG. In fact, subscribing to new shares not yet issued (as done in the CFA) does not trigger any disclosure obligations under §§ 38(1), 33(3) WpHG. At any rate, these would only be incumbent on Unicorn, not Neon. 

2.  Question 2 requires an analysis of prospectus requirements under the Prospectus Regulation.
a)  There is no public offer within the meaning of article 2(d) PR that would trigger a prospectus requirement under article 3(1) PR. However, pursuant to article 3(3) PR, admission of securities to trading on a regulated market requires prior publication of a prospectus. Neon shares qualify as securities under article 2(a) PR in conjunction with article 4(1)(44) MiFID II. Students should discuss the fact that there is an exemption for this type of transaction under article 1(5)(a) PR, but that the exemption is limited to a capital increase of 20% or less so does not cover Neon’s case. Accordingly, admission to trading requires publication of a prospectus (under article 21 PR), which in turn makes it necessary to have the prospectus approved under article 20(1) PR). A very complete answer would mention that Neon could benefit from the simplified disclosure regime for secondary issuances under article 14(1)(a) PR.
b)  As regards the content of the prospectus, students are expected to explain that the prospectus would have to include all information in connection with the CFA that is material within the meaning of article 6(1) PR, in particular, as regards the prospects of Neon (article 6(1)(1)(a) PR) and the reasons for the issuance (article 6(1)(1)(c) PR). The prospectus would also have to describe material risks resulting from the CFA and the new strategy (article 16(1) PR). A good answer would mention that the “criterion” for materiality under German case law is whether an investor would “rather than not” use the information for the investment decision.

3.  The question requires candidates to address disclosure obligations under the Transparency Directive and the Takeover Bid Directive and implementing domestic German law. 
a)  As Neon’s shares are listed on a regulated market, Neon is an issuer within the meaning of § 33(4) WpHG, so participations in Neon are subject to disclosure under §§33ff. WpHG. Pursuant to § 33(1) WpHG, Unicorn will have to disclose the acquisition of its stake in Neon. The relevant position to be disclosed includes the 23% stake held by Unicorn directly. In addition, Unicorn will have to take into account Gerry’s 19% stake if the CFA qualifies as “acting in concert” within the meaning of § 34(2) WpHG. In this context, students should differentiate between the two types of acting in concert, namely (i) an agreement to align the exercise of voting rights which qualifies as acting in concert irrespectively of the impact on the issuer’s strategy, and (ii) all other types of alignment which only qualify as acting in concert if it is aimed at modifying substantially the issuer’s strategic orientation. On the facts of the case, both requirements are fulfilled. A good answer should discuss this in the light of the BGH case law, and ideally also consider whether case law on acting in concert under WpÜG can and should be used to assess acting in concert under WpHG. A very complete answer would mention that Unicorn also has to make a statement of intent pursuant to § 43(1) WpHG.
b)  The acquisition of the new shares is also subject to WpÜG requirements pursuant to § 1(1) WpÜG as the shares issued by Neon are securities within the meaning of § 2(2) WpÜG and admitted to trading on a regulated market. Pursuant to § 35(1)(1) WpÜG, Unicorn has to disclose the fact that it acquired “control” in Neon and publish an offer document submit a draft offer to BaFin, §§ 35(2)(1), 14(2)(1) WpÜG. “Control” is defined as the acquisition of 30% or more in an issuer, § 29(2) WpÜG. The 23% stake held by Unicorn directly would not qualify as “control" triggering a mandatory bid requirement. However, § 30(2) WpÜG requires to include in the calculation shares held by other parties with which Unicorn is acting in concert, i.e., Gerry’s 19% stake (students may refer to the discussion of acting in concert under § 34(2) WpHG). The relevant position totals 42% and therefore the disclosure requirements under § 35(1) WpÜG.
c)  Failure to disclose under § 33 WpHG/§ 35 WpÜG will suspend Unicorn’s shareholder rights under § 44 WpHG, § 59 WpÜG. No such sanction exists as regards failure to make a statement of intent under § 43(1) WpHG.
"""

# ---------------- Scoring Rubric ----------------
REQUIRED_ISSUES = [
    {
        "name": "Inside information & timing (Art 7(1),(2),(4) MAR); disclosure & delay (Art 17 MAR; Lafonta)",
        "points": 26,
        "keywords": ["art 7", "inside information", "precise nature", "intermediate step", "protracted process", "lafonta", "art 17", "as soon as possible", "delay", "mislead"],
    },
    {
        "name": "Prospectus requirement on admission (PR 2017/1129: Art 3(3); exemption Art 1(5)(a) ≤20%; approval Art 20; publication Art 21; MiFID II 4(1)(44))",
        "points": 18,
        "keywords": ["prospectus regulation", "art 3(3)", "admission to trading", "art 1(5)(a)", "20%", "article 20", "article 21", "mifid ii", "4(1)(44)"],
    },
    {
        "name": "Prospectus content & risk factors (PR Art 6(1) materiality; reasons 6(1)(c); risk factors Art 16(1))",
        "points": 12,
        "keywords": ["article 6(1)", "material information", "reasons for the issue", "article 16(1)", "risk factors"],
    },
    {
        "name": "Shareholding notifications (WpHG §§ 33, 34(2); acting in concert; §43 statement of intent; §44 sanctions)",
        "points": 18,
        "keywords": ["§ 33 wphg", "§ 34 wphg", "acting in concert", "gemeinschaftliches handeln", "§ 43 wphg", "§ 44 wphg"],
    },
    {
        "name": "Takeover law (WpÜG §§ 29(2), 30(2) control; §35 mandatory offer/disclosure; §59 suspension of rights)",
        "points": 16,
        "keywords": ["§ 29 wpüg", "§ 30 wpüg", "§ 35 wpüg", "mandatory offer", "control 30%", "§ 59 wpüg"],
    },
    {
        "name": "Clarification: subscription doesn’t trigger §38/33(3) WpHG for Neon; only Unicorn",
        "points": 10,
        "keywords": ["§ 38 wphg", "§ 33(3) wphg", "subscription", "neon", "unicorn"],
    },
]
DEFAULT_WEIGHTS = {"similarity": 0.4, "coverage": 0.6}

QUESTION_MAP = {
    "Question 1": [
        "Inside information & timing (Art 7(1),(2),(4) MAR); disclosure & delay (Art 17 MAR; Lafonta)",
        "Clarification: subscription doesn’t trigger §38/33(3) WpHG for Neon; only Unicorn"
    ],
    "Question 2": [
        "Prospectus requirement on admission (PR 2017/1129: Art 3(3); exemption Art 1(5)(a) ≤20%; approval Art 20; publication Art 21; MiFID II 4(1)(44))",
        "Prospectus content & risk factors (PR Art 6(1) materiality; reasons 6(1)(c); risk factors Art 16(1))"
    ],
    "Question 3": [
        "Shareholding notifications (WpHG §§ 33, 34(2); acting in concert; §43 statement of intent; §44 sanctions)",
        "Takeover law (WpÜG §§ 29(2), 30(2) control; §35 mandatory offer/disclosure; §59 suspension of rights)"
    ]
}

def filter_model_answer_and_rubric(selected_question):
    if selected_question == "Question 1":
        model_answer_filtered = MODEL_ANSWER.split("2.")[0].strip()
    elif selected_question == "Question 2":
        model_answer_filtered = MODEL_ANSWER.split("2.")[1].split("3.")[0].strip()
    elif selected_question == "Question 3":
        model_answer_filtered = MODEL_ANSWER.split("3.")[1].strip()
    else:
        model_answer_filtered = MODEL_ANSWER
    selected_issues = [issue for issue in REQUIRED_ISSUES if issue["name"] in QUESTION_MAP[selected_question]]
    return model_answer_filtered, selected_issues

# ---- Manual relevance terms per question ----
MANUAL_KEY_TERMS = {
    "Question 1": [
        "MAR", "Art 7", "Article 7", "Art 17",
        "inside information", "Insiderinformation",
        "precise nature", "intermediate step", "protracted process",
        "delay", "Verzögerung", "mislead", "irreführung"
    ],
    "Question 2": [
        "Prospectus Regulation", "2017/1129",
        "Art 3(3)", "Article 3(3)", "admission to trading",
        "Art 1(5)(a)", "Article 1(5)(a)", "20%", "secondary issuance",
        "MiFID", "4(1)(44)", "transferable securities", "Prospekt"
    ],
    "Question 3": [
        "WpHG", "§ 33", "§ 34", "acting in concert", "gemeinschaftliches Handeln",
        "WpÜG", "§ 29", "§ 30", "§ 35", "§ 59", "Pflichtangebot", "Kontrolle", "30%"
    ]
}

def manual_chunk_relevant(text: str, selected_question: str) -> bool:
    """Return True if the manual chunk contains any key term for the active question."""
    terms = MANUAL_KEY_TERMS.get(selected_question, [])
    low = text.lower()
    return any(term.lower() in low for term in terms)

# ---------------- Robust keyword & citation checks ----------------
def normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

def canonicalize(s: str, strip_paren_numbers: bool = False) -> str:
    s = s.lower()
    s = s.replace("art.", "art").replace("article", "art").replace("–", "-")
    s = s.replace("wpüg", "wpüg")
    s = re.sub(r"\s+", "", s)
    if strip_paren_numbers:
        s = re.sub(r"\(\d+[a-z]?\)", "", s)
    s = re.sub(r"[^a-z0-9§]", "", s)
    return s

def keyword_present(answer: str, kw: str) -> bool:
    ans_can = canonicalize(answer, strip_paren_numbers=True)
    kw_can = canonicalize(kw, strip_paren_numbers=True)
    if kw.strip().lower().startswith(("§", "art")):
        return kw_can in ans_can
    return normalize_ws(kw).lower() in normalize_ws(answer).lower()

def coverage_score(answer: str, issue: Dict) -> Tuple[int, List[str]]:
    hits = [kw for kw in issue["keywords"] if keyword_present(answer, kw)]
    score = int(round(issue["points"] * (len(hits) / max(1, len(issue["keywords"])))))
    return score, hits

def detect_substantive_flags(answer: str) -> List[str]:
    flags = []
    low = answer.lower()
    if "always delay" in low or re.search(r"\b(can|may)\s+always\s+delay\b", low):
        flags.append("Delay under Art 17(4) MAR is conditional: (a) legitimate interest, (b) not misleading, (c) confidentiality ensured.")
    return flags

def summarize_rubric(student_answer: str, model_answer: str, backend, required_issues: List[Dict], weights: Dict):
    embs = embed_texts([student_answer, model_answer], backend)
    sim = cos_sim(embs[0], embs[1])
    sim_pct = max(0.0, min(100.0, 100.0 * (sim + 1) / 2))

    per_issue, tot, got = [], 0, 0
    for issue in required_issues:
        pts = issue.get("points", 10)
        tot += pts
        sc, hits = coverage_score(student_answer, issue)
        got += sc
        per_issue.append({
            "issue": issue["name"], "max_points": pts, "score": sc,
            "keywords_hit": hits, "keywords_total": issue["keywords"],
        })
    cov_pct = 100.0 * got / max(1, tot)
    final = (weights["similarity"] * sim_pct + weights["coverage"] * cov_pct) / (weights["similarity"] + weights["coverage"])

    missing = []
    for row in per_issue:
        missed = [kw for kw in row["keywords_total"] if kw not in row["keywords_hit"]]
        if missed:
            missing.append({"issue": row["issue"], "missed_keywords": missed})

    substantive_flags = detect_substantive_flags(student_answer)

    return {
        "similarity_pct": round(sim_pct, 1),
        "coverage_pct": round(cov_pct, 1),
        "final_score": round(final, 1),
        "per_issue": per_issue,
        "missing": missing,
        "substantive_flags": substantive_flags,   # keep this if you still want it
    }
    
# ---------------- Web Retrieval (RAG) ----------------
ALLOWED_DOMAINS = {
    "eur-lex.europa.eu",        # EU law (MAR, PR, MiFID II, TD)
    "curia.europa.eu",          # CJEU (Lafonta C‑628/13 etc.)
    "www.esma.europa.eu",       # ESMA guidelines/news
    "www.bafin.de",             # BaFin
    "www.gesetze-im-internet.de", "gesetze-im-internet.de",  # WpHG, WpÜG
    "www.bundesgerichtshof.de", # BGH
}

SEED_URLS = [
    "https://eur-lex.europa.eu/eli/reg/2014/596/oj",   # MAR
    "https://eur-lex.europa.eu/eli/reg/2017/1129/oj",  # Prospectus Regulation
    "https://eur-lex.europa.eu/eli/dir/2014/65/oj",    # MiFID II
    "https://eur-lex.europa.eu/eli/dir/2004/109/oj",   # Transparency Directive
    "https://curia.europa.eu/juris/liste.jsf?num=C-628/13",  # Lafonta
    "https://www.gesetze-im-internet.de/wphg/",
    "https://www.gesetze-im-internet.de/wpu_g/",
    "https://www.esma.europa.eu/press-news/esma-news/esma-finalises-guidelines-delayed-disclosure-inside-information-under-mar",
]

UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"}

@st.cache_data(ttl=3600, show_spinner=False)
def duckduckgo_search(query: str, max_results: int = 6) -> List[Dict]:
    url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
    try:
        r = requests.get(url, headers=UA, timeout=20)
        r.raise_for_status()
    except Exception:
        return []
    soup = BeautifulSoup(r.text, "lxml")
    out = []
    for a in soup.select("a.result__a"):
        href = a.get("href")
        title = a.get_text(" ", strip=True)
        if not href:
            continue
        domain = urlparse(href).netloc.lower()
        if any(domain.endswith(d) for d in ALLOWED_DOMAINS):
            out.append({"title": title, "url": href})
        if len(out) >= max_results:
            break
    return out

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_url(url: str) -> Dict:
    try:
        r = requests.get(url, headers=UA, timeout=25)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "lxml")
        for tag in soup(["script", "style", "nav", "header", "footer", "aside"]):
            tag.decompose()
        title = (soup.title.get_text(strip=True) if soup.title else url)
        text = " ".join(soup.stripped_strings)
        text = re.sub(r"\s+", " ", text)
        if len(text) > 120000:
            text = text[:120000]
        return {"url": url, "title": title, "text": text}
    except Exception:
        return {"url": url, "title": url, "text": ""}

def build_queries(student_answer: str, extra_user_q: str = "") -> List[str]:
    base = [
        "Article 17 MAR delay disclosure ESMA guidelines site:eur-lex.europa.eu OR site:esma.europa.eu OR site:bafin.de",
        "Article 7(2) MAR precise information intermediate step Lafonta site:curia.europa.eu OR site:eur-lex.europa.eu",
        "Prospectus Regulation 2017/1129 Article 3(3) admission prospectus requirement site:eur-lex.europa.eu",
        "Prospectus Regulation Article 1(5)(a) 20% exemption site:eur-lex.europa.eu",
        "Prospectus Regulation Article 6(1) information Article 16(1) risk factors site:eur-lex.europa.eu",
        "MiFID II Article 4(1)(44) transferable securities site:eur-lex.europa.eu",
        "WpHG § 33 § 34(2) acting in concert gemeinschaftliches Handeln site:gesetze-im-internet.de OR site:bafin.de",
        "WpHG § 43 Abs 1 Absichtserklärung site:gesetze-im-internet.de OR site:bafin.de",
        "WpHG § 44 Rechte ruhen Sanktion site:gesetze-im-internet.de OR site:bafin.de",
        "WpÜG § 29 § 30 Kontrolle 30 Prozent acting in concert site:gesetze-im-internet.de OR site:bafin.de",
        "WpÜG § 35 Pflichtangebot Veröffentlichung BaFin site:gesetze-im-internet.de OR site:bafin.de",
        "WpÜG § 59 Ruhen von Rechten site:gesetze-im-internet.de OR site:bafin.de",
    ]
    if student_answer:
        base.append(f"({student_answer[:300]}) Neon Unicorn CFA MAR PR WpHG WpÜG site:eur-lex.europa.eu OR site:gesetze-im-internet.de")
    if extra_user_q:
        base.append(extra_user_q + " site:eur-lex.europa.eu OR site:gesetze-im-internet.de OR site:curia.europa.eu OR site:esma.europa.eu OR site:bafin.de")
    return base

def collect_corpus(student_answer: str, extra_user_q: str, max_fetch: int = 20) -> List[Dict]:
    results = [{"title": "", "url": u} for u in SEED_URLS]
    for q in build_queries(student_answer, extra_user_q):
        results.extend(duckduckgo_search(q, max_results=5))
    seen, cleaned = set(), []
    for r in results:
        url = r["url"]
        if url in seen:
            continue
        seen.add(url)
        domain = urlparse(url).netloc.lower()
        if not any(domain.endswith(d) for d in ALLOWED_DOMAINS):
            continue
        cleaned.append(r)
    fetched = []
    for r in cleaned[:max_fetch]:
        pg = fetch_url(r["url"])
        if pg["text"]:
            pg["title"] = pg["title"] or r.get("title") or r["url"]
            fetched.append(pg)
    return fetched

def retrieve_snippets_with_manual(student_answer: str, model_answer: str, pages: List[Dict], backend,
                                  top_k_pages: int = 8, chunk_words: int = 170):
    # ---- Load & chunk Course Booklet with page/para/case metadata
    manual_chunks, manual_metas = [], []
    try:
        manual_chunks, manual_metas = extract_manual_chunks_with_refs(
            "assets/EUCapML - Course Booklet.pdf",
            chunk_words_hint=chunk_words
        )
    except Exception as e:
        st.warning(f"Could not load course manual: {e}")

    # ✅ Filter manual chunks by the active question to avoid irrelevant booklet citations
    selected_q = st.session_state.get("selected_question", "Question 1")
    filtered_chunks, filtered_metas = [], []
    for ch, m in zip(manual_chunks, manual_metas):
        if manual_chunk_relevant(ch, selected_q):
            filtered_chunks.append(ch)
            filtered_metas.append(m)

    # If filtering removes everything (e.g., unusual terms), fall back to the original set
    if filtered_chunks:
        manual_chunks, manual_metas = filtered_chunks, filtered_metas

    # ---- Prepare manual meta tuples with a unique key per *page* so we can group snippets by page
    manual_meta = []
    for m in manual_metas:
        page_key = -(m["page_num"])  
        citation = format_manual_citation(m)  # pre-format a nice line
        # We store citation in 'title' so we can reuse downstream without new structures
        manual_meta.append((page_key, "manual://course-booklet", citation))

    # ---- Prepare web chunks (unchanged)
    web_chunks, web_meta = [], []
    for i, p in enumerate(pages):
        for ch in split_into_chunks(p["text"], max_words=chunk_words):
            web_chunks.append(ch)
            web_meta.append((i + 1, p["url"], p["title"]))

    # ---- Build combined corpus
    all_chunks = manual_chunks + web_chunks
    all_meta   = manual_meta   + web_meta

    # Query vector built from student + model slice
    query = (student_answer or "") + "\n\n" + (model_answer or "")
    embs = embed_texts([query] + all_chunks, backend)
    qv, cvs = embs[0], embs[1:]
    sims = [cos_sim(qv, v) for v in cvs]
    idx = np.argsort(sims)[::-1]

    # ✅ Similarity floor to keep only reasonably relevant snippets
    MIN_SIM = 0.12  # tune 0.10–0.18 if needed

    # ---- Select top snippets grouped by (manual page) or (web page index)
    per_page = {}
    for j in idx:
        if sims[j] < MIN_SIM:
            break
        pi, url, title = all_meta[j]
        snip = all_chunks[j]
        arr = per_page.setdefault(pi, {"url": url, "title": title, "snippets": []})
        if len(arr["snippets"]) < 3:
            arr["snippets"].append(snip)
        if len(per_page) >= top_k_pages:
            break

    # Order by key and build source lines. For manual items we already have 'title' as a full citation line.
    top_pages = [per_page[k] for k in sorted(per_page.keys())][:top_k_pages]

    source_lines = []
    for i, tp in enumerate(top_pages):
        if tp["url"].startswith("manual://"):
            # already a fully formatted citation like: "Course Booklet — p. ii (PDF p. 4), para. 115"
            source_lines.append(f"[{i+1}] {tp['title']}")
        else:
            source_lines.append(f"[{i+1}] {tp['title']} — {tp['url']}")

    return top_pages, source_lines

# ---------------- LLM via Groq (free) ----------------
def call_groq(messages: List[Dict], api_key: str, model_name: str = "llama-3.1-8b-instant",
              temperature: float = 0.2, max_tokens: int = 700) -> str:
    """
    Groq OpenAI-compatible chat endpoint. Models like llama-3.1-8b-instant / 70b-instant are free.
    """
    if not api_key:
        st.error("No GROQ_API_KEY found (add it to Streamlit Secrets).")
        return None
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    data = {
        "model": model_name,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    try:
        r = requests.post(url, headers=headers, json=data, timeout=90)
        if r.status_code != 200:
            # Show exact error so it's easy to fix
            try: body = r.json()
            except Exception: body = r.text
            st.error(f"Groq error {r.status_code}: {body}")
            return None
        return r.json()["choices"][0]["message"]["content"]
    except requests.exceptions.Timeout:
        st.error("Groq request timed out (60s). Try again or reduce max_tokens.")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"Groq request failed: {e}")
        return None

def system_guardrails():
    return (
        "You are a careful EU/German capital markets law tutor.\n"
        "PRIORITY RULES:\n"
        "1. Base all feedback on the authoritative MODEL ANSWER and the numbered SOURCES provided.\n"
        "2. If SOURCES conflict with MODEL ANSWER, follow the MODEL ANSWER and briefly explain why.\n"
        "3. Never reveal or mention that a hidden model answer exists.\n\n"
        "CITATIONS POLICY:\n"
        "- Cite ONLY using [n], where n refers to the SOURCES list below.\n"
        "- Never invent Course Booklet references (pages, paragraphs, cases). "
        "  Only refer to the Course Booklet by citing the corresponding [n] that appears in the SOURCES list; "
        "  do NOT fabricate page/para/case numbers.\n"
        "- Do not cite any material that does not appear in the SOURCES list.\n\n"
        "FEEDBACK PRINCIPLES:\n"
        "- If the student's conclusion is incorrect, explicitly state the correct conclusion first, then explain why with citations [n].\n"
        "- If the student's answer is irrelevant to the selected question, say: 'Are you sure your answer corresponds to the question you selected?'\n"
        "- If central concepts are missing, point this out and explain why they matter.\n"
        "- Correct mis-citations succinctly (e.g., Art 3(1) PR → Art 3(3) PR; §40 WpHG → §43(1) WpHG).\n"
        "- Summarize or paraphrase concepts; do not copy long passages.\n\n"
        "STYLE:\n"
        "- Be concise, didactic, and actionable.\n"
        "- Use ≤400 words, no new sections.\n"
        "- Finish with a single explicit concluding sentence.\n"
        "- Write in the same language as the student's answer when possible (if mixed, default to English)."
    )

def build_feedback_prompt(student_answer: str, rubric: Dict, model_answer: str,
                          sources_block: str, excerpts_block: str) -> str:
    # Derive the coverage checklist from the deterministic rubric you already computed
    issue_names = [row["issue"] for row in rubric.get("per_issue", [])]

    return f"""
GRADE THE STUDENT'S ANSWER USING THE RUBRIC AND THE WEB/BOOKLET SOURCES.

STUDENT ANSWER:
\"\"\"{student_answer}\"\"\"

RUBRIC SUMMARY:
- Similarity to model answer: {rubric.get('similarity_pct', 0)}%
- Issue coverage: {rubric.get('coverage_pct', 0)}%
- Overall score: {rubric.get('final_score', 0)}%

RUBRIC ISSUES TO COVER:
- {", ".join(issue_names) if issue_names else "—"}

MODEL ANSWER (AUTHORITATIVE):
\"\"\"{model_answer}\"\"\"

SOURCES (numbered; you must cite ONLY from this list. Never invent Course Booklet references):
{sources_block}

EXCERPTS (quote sparingly; cite as [n]):
{excerpts_block}

TASK:
Write ≤400 words of actionable feedback.
Start by stating the correct conclusion if the student's conclusion is wrong (e.g., "In fact, the CFA is inside information under Art 7(1),(2) MAR because …") and support it with precise [n] citations.
Then:
- Explain why any incorrect statements are wrong, with citations [n].
- Add missing points for ALL rubric issues and why they matter (with [n]).
- Correct mis-citations succinctly (e.g., Art 3(1) PR → Art 3(3) PR).
- Cover ALL rubric issues for this question, even if the student did not mention them.
IMPORTANT: Cite ONLY the numbered SOURCES above. Do NOT invent any Course Booklet references (pages, paragraphs, cases). If a point is not supported by these sources, say so and avoid making up a citation.
Paraphrase rather than quoting long passages. 
Do not disclose internal materials or say that a hidden model answer exists; rely on the numbered sources and the summary above.
Do not disclose scorings.
Paraphrase rather than quoting long passages; keep the tone clear, didactic, and practical.
Use a clear structure: (1) Correct conclusion, (2) Errors explained, (3) Missing points, (4) Improvement tips.
End with a short concluding sentence.
"""

def build_chat_messages(chat_history: List[Dict], model_answer: str, sources_block: str, excerpts_block: str) -> List[Dict]:
    msgs = [{"role": "system", "content": system_guardrails()}]
    for m in chat_history[-8:]:
        if m["role"] in ("user", "assistant"): msgs.append(m)
    # Pin authoritative context and sources
    msgs.append({"role": "system", "content": "MODEL ANSWER (authoritative):\n" + model_answer})
    msgs.append({"role": "system", "content": "SOURCES:\n" + sources_block})
    msgs.append({"role": "system", "content": "RELEVANT EXCERPTS (quote sparingly):\n" + excerpts_block})
    return msgs

# ------------------------------ Chat Helpers ----------
# ---- Output completeness helpers ----
def is_incomplete_text(text: str) -> bool:
    """Heuristic: returns True if the text likely ends mid-sentence."""
    if not text or not text.strip():
        return True
    tail = text.strip()[-1]
    return tail not in ".!?…’”\")»]"

def truncate_block(s: str, max_chars: int = 3600) -> str:
    """Trim very long prompt sections to reduce truncation risk."""
    s = s or ""
    return s if len(s) <= max_chars else (s[:max_chars] + " …")

def generate_with_continuation(messages, api_key, model_name, temperature=0.2, first_tokens=1200, continue_tokens=350):
    """
    Calls the LLM, and if output ends mid-sentence, asks it to continue once.
    """
    reply = call_groq(messages, api_key, model_name=model_name, temperature=temperature, max_tokens=first_tokens)
    if reply and is_incomplete_text(reply):
        # Ask for a short continuation to finish the sentence + a 1‑sentence conclusion
        cont_msgs = messages + [{
            "role": "user",
            "content": "Continue exactly where you left off. Finish the previous sentence and add a single-sentence conclusion. Do not repeat earlier text."
        }]
        more = call_groq(cont_msgs, api_key, model_name=model_name, temperature=min(temperature, 0.3), max_tokens=continue_tokens)
        if more:
            reply = (reply.rstrip() + "\n" + more.strip())
    return reply

def render_sources_used(source_lines: list[str]) -> None:
    with st.expander("📚 Sources used", expanded=False):
        if not source_lines:
            st.write("— no web sources available —")
            return
        for line in source_lines:
            st.markdown(f"- {line}")

def clear_chat_draft():
    # Clear the persistent composer safely during the button's on_click callback
    st.session_state["chat_draft"] = ""

# ---- Course Booklet parsing helpers ----
# ---- Course Booklet parsing helpers (fixed) ----
def split_into_paragraphs(text: str) -> list[str]:
    """Split page text into paragraphs using blank lines; fall back to grouping lines."""
    paras = [t.strip() for t in re.split(r"\n\s*\n", text) if t.strip()]
    if paras:
        return paras
    # Fallback: group every ~8 lines into a paragraph
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    out, cur = [], []
    for i, ln in enumerate(lines):
        cur.append(ln)
        if (i + 1) % 8 == 0:
            out.append(" ".join(cur)); cur = []
    if cur:
        out.append(" ".join(cur))
    return out

# Paragraph markers actually present in the booklet:
#  - "para. 115"/"paragraph 115"/"Rn. 115"
#  - left-margin bold numbers extracted as **176**, ** 10 **, etc.
_para_patterns = [
    re.compile(r"\bpara(?:graph)?\.?\s*(\d{1,4})\b", re.I),
    re.compile(r"\brn\.?\s*(\d{1,4})\b", re.I),
    re.compile(r"\*\*\s*(\d{1,4})\s*\*\*"),
]

# Case markers: prefer "Case Study N", also accept "Case N"/"Fall N"
_case_patterns = [
    re.compile(r"\bCase\s*Study\s*(\d{1,4})\b", re.I),
    re.compile(r"\b(?:Case|Fall)\s*(\d{1,4})\b", re.I),
]

# Case markers: "Case 14" or "Fall 14"
_case_pattern = re.compile(r"\b(?:Case|Fall)\s*(\d{1,4})\b", re.I)

def detect_para_numbers(text: str) -> list[int]:
    nums = []
    for pat in _para_patterns:
        matches = pat.findall(text)  # e.g., ['115'] or possibly tuples if patterns ever change
        for m in matches:
            # If a pattern produces tuples (multiple groups), pick the first non-empty
            if isinstance(m, tuple):
                m = next((g for g in m if g), "")
            # Keep only pure digits
            if not m or not m.isdigit():
                continue
            nums.append(int(m))

    # unique while preserving order
    out = []
    for n in nums:
        if n not in out:
            out.append(n)
    return out

def detect_case_numbers(text: str) -> list[int]:
    nums = []
    for pat in _case_patterns:
        nums += pat.findall(text)
    out = []
    for x in nums:
        if x not in out:
            out.append(x)
    return [int(x) for x in out]

def extract_manual_chunks_with_refs(pdf_path: str, chunk_words_hint: int = 170) -> tuple[list[str], list[dict]]:
    """
    Returns (chunks, metas) with accurate page *labels* (printed page numbers),
    not just 1-based PDF indices. We split by paragraphs and break very long ones by sentences.
    meta: {pdf_index, page_label, page_num, paras [ints], cases [ints], file}
    """
    chunks, metas = [], []
    try:
        doc = fitz.open(pdf_path)
    except Exception:
        return [], []

    for pno in range(len(doc)):
        page = doc.load_page(pno)
        page_text = page.get_text("text")
        # Remove repeating headers like "Version 11 June 2025" and stray page numbers
        page_text = clean_page_text(page_text)
        page_label = page.get_label() or str(pno + 1)
        paras = split_into_paragraphs(page_text)
        
        for para in paras:
            # Break very long paragraphs roughly to the hint size
            words = para.split()
            if len(words) > chunk_words_hint * 2:
                parts = re.split(r"\(?\<=\[\\.\?\\\!\\]\)\s+", para)  # <-- fixed sentence splitter
                parts = re.split(r"(?<=[\.\?\!…])\s+", para)  # split on whitespace after ., ?, !, …
                cur, cur_words = [], 0
                for s in parts:
                    cur.append(s)
                    cur_words += len(s.split())
                    if cur_words >= chunk_words_hint:
                        para_part = " ".join(cur).strip()
                        chunks.append(para_part)
                        metas.append({
                            "pdf_index": pno,
                            "page_label": page_label,
                            "page_num": pno + 1,  # numeric fallback for anchors
                            "paras": detect_para_numbers(para_part) or detect_para_numbers(para),
                            "cases": detect_case_numbers(para_part) or detect_case_numbers(para),
                            "file": "EUCapML - Course Booklet.pdf",  # <-- unify name with actual path
                        })
                        cur, cur_words = [], 0
                if cur:
                    para_part = " ".join(cur).strip()
                    chunks.append(para_part)
                    metas.append({
                        "pdf_index": pno,
                        "page_label": page_label,
                        "page_num": pno + 1,
                        "paras": detect_para_numbers(para_part) or detect_para_numbers(para),
                        "cases": detect_case_numbers(para_part) or detect_case_numbers(para),
                        "file": "EUCapML - Course Booklet.pdf",
                    })
            else:
                chunks.append(para)
                metas.append({
                    "pdf_index": pno,
                    "page_label": page_label,
                    "page_num": pno + 1,
                    "paras": detect_para_numbers(para),
                    "cases": detect_case_numbers(para),
                    "file": "EUCapML - Course Booklet.pdf",
                })
    doc.close()
    return chunks, metas

def format_manual_citation(meta: dict) -> str:
    """
    Build a human-friendly citation like:
      'Course Booklet — p. ii (PDF p. 4), Case Study 14; para. 115'
      or, if only one is available: '..., Case Study 14'  or  '..., para. 115'
    """
    lbl = meta.get("page_label")
    pdfp = meta.get("page_num")
    paras = meta.get("paras") or []
    cases = meta.get("cases") or []

    parts = [f"Course Booklet — p. {lbl} (PDF p. {pdfp})"]

    # Collect anchors (we show both if both are present)
    anchors = []

    # Prefer to show one explicit Case Study number if available
    if cases:
        anchors.append(f"Case Study {cases[0]}")

    # Add a concise paragraph anchor if available
    if paras:
        # De-duplicate and sort once, then pick the most helpful concise form
        xs = sorted(set(int(p) for p in paras if isinstance(p, (int, str)) and str(p).isdigit()))
        if xs:
            # If we have at least two and they are consecutive, show a short range; else show the first one
            if len(xs) >= 2 and xs[1] == xs[0] + 1:
                anchors.append(f"paras {xs[0]}–{xs[1]}")
            else:
                anchors.append(f"para. {xs[0]}")

    if anchors:
        parts.append("; ".join(anchors))

    return ", ".join(parts)

# ---- Simple page cleaner for booklet parsing ----
def clean_page_text(t: str) -> str:
    """
    Drop repeating page header/footer noise (e.g., 'Version 11 June 2025') and lone page numbers.
    """
    out = []
    for ln in t.splitlines():
        # Header like "Version 11 June 2025"
        if re.match(r"^\s*Version\s+\d{1,2}\s+\w+\s+\d{4}\s*$", ln):
            continue
        # A lone page number line
        if re.match(r"^\s*\d+\s*$", ln):
            continue
        out.append(ln)
    return "\n".join(out)

# --- Chat callbacks ------------------------------------------------------------
def clear_chat_draft():
    # Clear the persistent composer safely
    st.session_state["chat_draft"] = ""
    # optional: st.rerun()

def reset_chat():
    # Wipes the entire conversation (answers + questions + their sources)
    st.session_state["chat_history"] = []
    # optional: also clear the draft:
    # st.session_state["chat_draft"] = ""
    st.rerun()  # ensure immediate re-render

def clear_last_exchange():
    """
    Removes the last assistant message and, if present, the immediately preceding user message.
    Useful if the last answer was off-topic or leaked style.
    """
    hist = list(st.session_state.get("chat_history", []))
    if not hist:
        return
    # Pop trailing whitespace/system noise if any (defensive)
    while hist and hist[-1].get("role") not in ("user", "assistant"):
        hist.pop()

    # Remove last assistant message (if any)
    if hist and hist[-1].get("role") == "assistant":
        hist.pop()

    # Remove the preceding user question (if any)
    if hist and hist[-1].get("role") == "user":
        hist.pop()

    st.session_state["chat_history"] = hist
    st.rerun()

# ---------------- UI ----------------
import streamlit as st
import os
import requests

st.set_page_config(
    page_title="EUCapML Case Tutor", 
    page_icon="⚖️", 
    layout="wide",
    initial_sidebar_state="collapsed",   # ← collapsed by default
)

# Student login
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
    
if not st.session_state.authenticated:
    with st.container():
        logo_col, title_col = st.columns([1, 5])
        with logo_col:
            st.image("assets/logo.png", width=240)
        with title_col:
            st.title("EUCapML Case Tutor")
    
    pin_input = st.text_input("Enter your student PIN", type="password")

    try:
        correct_pin = st.secrets["STUDENT_PIN"]
    except KeyError:
        st.error("STUDENT_PIN not found in secrets. Please configure it in .streamlit/secrets.toml.")
        st.stop()

    if pin_input == correct_pin:
        st.session_state.authenticated = True
        st.success("PIN accepted. Click below to continue. By clicking you accept that this tool is based on artificial intelligence and large language model engines, and that accordingly, answers may not be accurate.")
        if st.button("Continue"):
            st.experimental_rerun()
    elif pin_input:
        st.error("Incorrect PIN. Please try again.")
    st.stop()

# Sidebar (visible to all users after login)
with st.sidebar:
    st.header("⚙️ Settings")
    api_key = (st.secrets.get("GROQ_API_KEY") if hasattr(st, "secrets") else None) or os.getenv("GROQ_API_KEY")
    if api_key:
        st.text_input("GROQ API Key", value="Provided via secrets/env", type="password", disabled=True)
    else:
        api_key = st.text_input("GROQ API Key", type="password", help="Set GROQ_API_KEY in Streamlit Secrets for production.")

    model_name = st.selectbox(
        "Model (free)",
        options=["llama-3.1-8b-instant", "llama-3.1-70b-instant"],
        index=0,
        help="Both are free; 8B is faster, 70B is smarter (and slower)."
    )
    temp = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05)

    st.header("🌐 Web Retrieval")
    enable_web = st.checkbox("Enable web grounding", value=True)
    max_sources = st.slider("Max sources to cite", 3, 10, 6, 1)
    st.caption("DuckDuckGo HTML + filters to EUR‑Lex, CURIA, ESMA, BaFin, Gesetze‑im‑Internet, BGH.")

    st.divider()
    st.subheader("Diagnostics")
    if st.checkbox("Run Groq connectivity test"):
        try:
            r = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key or ''}", "Content-Type": "application/json"},
                json={"model": model_name, "messages": [{"role": "user", "content": "Say: hello from Groq test"}], "max_tokens": 8},
                timeout=20,
            )
            st.write("POST /chat/completions →", r.status_code)
            st.code((r.text or "")[:1000], language="json")
        except Exception as e:
            st.exception(e)
    
    # ---- Course Booklet diagnostics ----
    st.subheader("Course Booklet diagnostics")
    if st.checkbox("Preview parsed booklet (first 6 pages)"):
        try:
            chunks, metas = extract_manual_chunks_with_refs("assets/EUCapML - Course Booklet.pdf", chunk_words_hint=160)
            by_page = {}
            for ch, m in zip(chunks, metas):
                by_page.setdefault(m["page_label"], []).append((ch[:140] + ("…" if len(ch) > 140 else ""), m))
            for i, (lbl, arr) in enumerate(list(by_page.items())[:6], start=1):
                st.markdown(f"**Page label {lbl}** (PDF p. {arr[0][1]['page_num']})")
                for snip, meta in arr[:2]:  # show 2 snippets per page
                    st.write("•", snip)
                    st.caption(f"Cases: {meta['cases'] or '—'} | Paras: {meta['paras'] or '—'}")
        except Exception as e:
            st.warning(f"Preview failed: {e}")
            
# Main UI
st.image("assets/logo.png", width=240)
st.title("EUCapML Case Tutor")

with st.expander("📚 Case (click to read)"):
    st.write(CASE)

selected_question = st.selectbox(
    "Which question are you answering?",
    options=["Question 1", "Question 2", "Question 3"],
    index=0,
    help="This limits feedback to the selected question only."
)
st.session_state["selected_question"] = selected_question

st.subheader("📝 Your Answer")
student_answer = st.text_area("Write your solution here (≥ ~120 words).", height=260)


# ------------- Actions -------------
colA, colB = st.columns([1, 1])

with colA:
    if st.button("🔎 Generate Feedback"):
        if len(student_answer.strip()) < 80:
            st.warning("Please write a bit more so I can evaluate meaningfully (≥ 80 words).")
        else:
            with st.spinner("Scoring and collecting sources..."):
                backend = load_embedder()
                model_answer_filtered, rubric_issues = filter_model_answer_and_rubric(selected_question)
                rubric = summarize_rubric(student_answer, model_answer_filtered, backend, rubric_issues, DEFAULT_WEIGHTS)
                
                top_pages, source_lines = [], []
                if enable_web:
                    pages = collect_corpus(student_answer, "", max_fetch=22)
                    top_pages, source_lines = retrieve_snippets_with_manual(student_answer, model_answer_filtered, pages, backend, top_k_pages=max_sources, chunk_words=170)
                    
            # Breakdown
            with st.expander("🔬 Issue-by-issue breakdown"):
                for row in rubric["per_issue"]:
                    st.markdown(f"**{row['issue']}** — {row['score']} / {row['max_points']}")
                    st.markdown(f"- ✅ Found: {', '.join(row['keywords_hit']) if row['keywords_hit'] else '—'}")
                    miss = [kw for kw in row["keywords_total"] if kw not in row["keywords_hit"]]
                    st.markdown(f"- ⛔ Missing: {', '.join(miss) if miss else '—'}")

            # Deterministic corrections
            if rubric["substantive_flags"]:
                st.markdown("### ⚖️ Detected substantive flags")
                for fl in rubric["substantive_flags"]:
                    st.markdown(f"- ⚖️ {fl}")
            
            # LLM narrative feedback
            sources_block = "\n".join(source_lines) if source_lines else "(no web sources available)"
            excerpts_items = []
            for i, tp in enumerate(top_pages):
                for sn in tp["snippets"]:
                    excerpts_items.append(f"[{i+1}] {sn}")
            excerpts_block = "\n\n".join(excerpts_items[: max_sources * 3]) if excerpts_items else "(no excerpts)"

            st.markdown("### 🧭 Narrative Feedback (with citations)")
            if api_key:
                # Trim large blocks *before* building the prompt
                sources_block = truncate_block(sources_block, 1200)
                excerpts_block = truncate_block(excerpts_block, 3200)
            
                messages = [
                    {"role": "system", "content": system_guardrails()},
                    {"role": "user", "content": build_feedback_prompt(student_answer, rubric, model_answer_filtered, sources_block, excerpts_block)},
                ]
            
                reply = generate_with_continuation(messages, api_key, model_name=model_name, temperature=temp,
                                   first_tokens=1200, continue_tokens=350)
                
                if reply:
                    st.write(reply)
                else:
                    st.info("LLM unavailable. See corrections above and the issue breakdown.")
            else:
                st.info("No GROQ_API_KEY found in secrets/env. Deterministic scoring and corrections shown above.")

            if source_lines:
                with st.expander("📚 Sources used"):
                    for line in source_lines:
                        st.markdown(f"- {line}")

with colB:
    st.markdown("### 💬 Tutor Chat: Ask me anything!")
    st.caption("You can use this chat to ask for help with creating an answer, follow-up questions on feedback given by this app, and discuss cases from the course booklet.")

    # --- state ---
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "chat_draft" not in st.session_state:
        st.session_state.chat_draft = ""

    # --- composer ---
    c1, c2, c3, c4 = st.columns([6, 1, 1, 2])
    with c1:
        st.text_area(
            "Ask a question about your feedback, the law, or how to improve…",
            key="chat_draft",
            height=90
        )
    with c2:
        send = st.button("Send", use_container_width=True, key="send_btn")
    with c3:
        st.button("Clear", use_container_width=True, key="clear_btn", on_click=clear_chat_draft)
    with c4:
        st.button("Reset chat", use_container_width=True, key="reset_chat_btn", on_click=reset_chat)
         
    # --- handle send: UPDATE STATE FIRST, DO NOT RENDER INLINE ---
    if send and st.session_state.chat_draft.strip():
        user_q = st.session_state.chat_draft
        st.session_state.chat_history.append({"role": "user", "content": user_q})

        with st.spinner("Retrieving sources and drafting a grounded reply..."):
            backend = load_embedder()
            top_pages, source_lines = [], []
            if enable_web:
                pages = collect_corpus(student_answer, user_q, max_fetch=20)
                top_pages, source_lines = retrieve_snippets_with_manual(
                    (student_answer or "") + "\n\n" + user_q,
                    model_answer_filtered, pages, backend,
                    top_k_pages=max_sources, chunk_words=170
                )

            
            sources_block = "\n".join(source_lines) if source_lines else "(no web sources available)"
            excerpts_items = []
            for i, tp in enumerate(top_pages):
                for sn in tp["snippets"]:
                    excerpts_items.append(f"[{i+1}] {sn}")
            excerpts_block = "\n\n".join(excerpts_items[: max_sources * 3]) if excerpts_items else "(no excerpts)"
            
            # ✅ Trim large blocks BEFORE building the prompt to free tokens for the answer
            sources_block  = truncate_block(sources_block, 1200)
            excerpts_block = truncate_block(excerpts_block, 3200)

            if api_key:
                msgs = build_chat_messages(
                    st.session_state.chat_history,
                    model_answer_filtered,
                    sources_block,
                    excerpts_block
                )
                reply = generate_with_continuation(
                    msgs, api_key, model_name=model_name, temperature=temp,
                    first_tokens=1200, continue_tokens=350
                )
            else:
                reply = None
            if not reply:
                reply = (
                    "I couldn’t reach the LLM. Here are the most relevant source snippets:\n\n"
                    + (excerpts_block if excerpts_block != "(no excerpts)" else "— no sources available —")
                    + "\n\nIn doubt, follow the model answer."
                )

        # Append the assistant message WITH its per-message sources
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": reply or "",
            "sources": source_lines[:]  # persistent per-message list
        })

    # --- render FULL history AFTER updates so latest sources appear immediately ---
    for msg in st.session_state.chat_history:
        role = msg.get("role", "")
        if role in ("user", "assistant"):
            with st.chat_message(role):
                st.write(msg.get("content", ""))
                if role == "assistant":
                    # Per-message "Sources used"
                    st.markdown("#### 📚 Sources used")
                    srcs = msg.get("sources", [])
                    if not srcs:
                        st.write("— no web sources available —")
                    else:
                        for line in srcs:
                            st.markdown(f"- {line}")

st.divider()
st.markdown(
    "ℹ️ **Notes**: This app is authored by Stephan Balthasar. It provides feedback based on artificial intelligence and large language models, and as a result, answers can be inaccurate." 
    "Students are advised to use caution when using the feedback engine and chat functions."
)
