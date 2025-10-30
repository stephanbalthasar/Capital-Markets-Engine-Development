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
from typing import List, Dict, Tuple
from urllib.parse import quote_plus, urlparse
import numpy as np
import streamlit as st
import statistics as stats
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

import requests
from bs4 import BeautifulSoup

# ---------------- Build fingerprint (to verify latest deployment) ----------------
APP_HASH = hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()[:10]

# ================================
# BEGIN NEW PDF PARSER (do not edit)
# ================================
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Union
import re
import fitz  # PyMuPDF

# ---------- low-level: line reconstruction using layout ----------
@dataclass
class _Line:
    y: float
    items: List[Tuple[float, str]]  # (x0, text)

    def text(self) -> str:
        return "".join(t for _, t in sorted(self.items, key=lambda z: z[0]))

def _iter_lines(page: fitz.Page, y_tol: float = 2.0) -> List[_Line]:
    """
    Build physical lines from blocks, preserving left/right positions.
    """
    blocks = page.get_text("blocks")  # (x0, y0, x1, y1, text, block_no, ...)
    frags: List[Tuple[float, float, str]] = []
    for b in blocks:
        x0, y0, x1, y1, text, *_ = b
        if not text:
            continue
        for raw in text.splitlines():
            if raw.strip():
                frags.append((x0, y0, raw))

    # sort by y then x
    frags.sort(key=lambda t: (round(t[1], 1), t[0]))

    lines: List[_Line] = []
    cur: Optional[_Line] = None
    for x0, y0, t in frags:
        if cur is None or abs(y0 - cur.y) > y_tol:
            if cur:
                lines.append(cur)
            cur = _Line(y=y0, items=[(x0, t)])
        else:
            cur.items.append((x0, t))
    if cur:
        lines.append(cur)
    return lines

# ---------- detectors & text utilities ----------
_NUM_RE_STRICT = re.compile(r"^\s*(\d{1,3})\s*[\.\)]?\s*$")   # e.g., "1", "1.", "(1)"
_CASE_STUDY_RE = re.compile(r"^\s*Case Study\s+(\d+)\.?\s*", re.IGNORECASE)
_WS_RE = re.compile(r"[ \t\u00A0]+")

def _normalize_ws(s: str) -> str:
    return _WS_RE.sub(" ", s or "").strip()

def _line_has_left_number(line: _Line, left_cutoff: float = 70.0) -> Optional[int]:
    """
    Detects gutter paragraph numbers printed near the left margin.
    """
    if not line.items:
        return None
    # Examine the first 1-2 low-x items (sometimes a dash precedes the number)
    for x0, txt in sorted(line.items, key=lambda z: z[0])[:2]:
        if x0 > left_cutoff:
            break
        m = _NUM_RE_STRICT.match(txt.strip())
        if m:
            return int(m.group(1))
    return None

def _join_line_text_excluding_leading_number(line: _Line) -> str:
    parts = [t for _, t in sorted(line.items, key=lambda z: z[0])]
    if parts and _NUM_RE_STRICT.match(parts[0].strip()):
        parts = parts[1:]  # drop the leading number token
    elif len(parts) >= 2 and parts[0].strip() in {"-", "•", "–"} and _NUM_RE_STRICT.match(parts[1].strip()):
        parts = parts[2:]
    return _normalize_ws(" ".join(parts))

def _join_line_text(line: _Line) -> str:
    return _normalize_ws(" ".join(t for _, t in sorted(line.items, key=lambda z: z[0])))

# ---------- core extractors ----------
def _extract_paragraphs(doc: fitz.Document) -> Dict[int, Dict[str, Union[str, int]]]:
    """
    Returns: { para_number: {"text": str, "page": int} }
    Combines left-gutter numbers with body text, spanning pages until next number appears.
    """
    results: Dict[int, Dict[str, Union[str, int]]] = {}
    current_n: Optional[int] = None
    buffer: List[str] = []
    first_seen_page: Dict[int, int] = {}

    def flush():
        nonlocal current_n, buffer
        if current_n is not None and buffer:
            text = _normalize_ws(" ".join(buffer))
            # Keep longest version if the same number is encountered twice
            if current_n not in results or len(text) > len(results[current_n]["text"]):
                results[current_n] = {"text": text, "page": first_seen_page[current_n]}
        buffer = []

    for pidx in range(len(doc)):
        page = doc[pidx]
        for line in _iter_lines(page):
            n = _line_has_left_number(line)
            if n is not None:
                flush()
                current_n = n
                first_seen_page.setdefault(n, pidx + 1)
                rest = _join_line_text_excluding_leading_number(line)
                buffer = [rest] if rest else []
                continue

            if current_n is not None:
                txt = _join_line_text(line)
                if txt:
                    buffer.append(txt)

    flush()
    return results
def _extract_case_studies(doc: fitz.Document) -> Dict[int, Dict[str, Dict[str, Union[str, int]]]]:
    """
    Returns: { N: { "prompt": {"text": str, "page": int}, "note": {"text": str, "page": int} } }
    - 'prompt' : the main Case Study on earlier pages
    - 'note'   : the entry in section "8 CASE NOTES" (if present)
    """
    found: Dict[int, Dict[str, Dict[str, Union[str, int]]]] = {}

    # pass 1: prompts
    for pidx in range(len(doc)):
        page = doc[pidx]
        lines = _iter_lines(page)
        i = 0
        while i < len(lines):
            t = lines[i].text()
            m = _CASE_STUDY_RE.match(t)
            if m:
                n = int(m.group(1))
                tail = t[m.end():].strip()

                # gather subsequent lines until next Case Study or next clean paragraph number headline
                buf = [tail] if tail else []
                j = i + 1
                while j < len(lines):
                    s = lines[j].text().strip()
                    if _CASE_STUDY_RE.match(s) or _NUM_RE_STRICT.match(s):
                        break
                    buf.append(s)
                    j += 1

                text = f"Case Study {n}. " + _normalize_ws(" ".join(buf))
                found.setdefault(n, {})["prompt"] = {"text": text, "page": pidx + 1}
                i = j
                continue
            i += 1

    # pass 2: notes (after heading contains "CASE NOTES")
    in_case_notes = False
    for pidx in range(len(doc)):
        page = doc[pidx]
        if "CASE NOTES" in (page.get_text("text") or "").upper():
            in_case_notes = True
        if not in_case_notes:
            continue

        lines = _iter_lines(page)
        i = 0
        while i < len(lines):
            t = lines[i].text().strip()
            m = _CASE_STUDY_RE.match(t)
            if m:
                n = int(m.group(1))
                buf = []
                j = i + 1
                while j < len(lines):
                    s = lines[j].text().strip()
                    if _CASE_STUDY_RE.match(s):
                        break
                    buf.append(s)
                    j += 1
                text = _normalize_ws(t + " " + " ".join(buf))
                found.setdefault(n, {})["note"] = {"text": text, "page": pidx + 1}
                i = j
                continue
            i += 1

    return found

# ---------- Public helpers you will call from the app ----------
def parse_pdf_all_from_path(pdf_path: str) -> Dict[str, dict]:
    """
    Parse a PDF from a file path. Returns:
    {
      "paragraphs": {1: {"text":..., "page":...}, ...},
      "case_studies": {1: {"prompt": {...}, "note": {...}}, ...}
    }
    """
    doc = fitz.open(pdf_path)
    paragraphs = _extract_paragraphs(doc)
    cases = _extract_case_studies(doc)
    return {"paragraphs": paragraphs, "case_studies": cases}

def parse_pdf_all_from_bytes(pdf_bytes: bytes) -> Dict[str, dict]:
    """
    Parse a PDF from raw bytes (e.g., st.file_uploader).
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    paragraphs = _extract_paragraphs(doc)
    cases = _extract_case_studies(doc)
    return {"paragraphs": paragraphs, "case_studies": cases}

def get_targets_for_app(parsed: Dict[str, dict]) -> Dict[str, str]:
    """
    Convenience for your tutor app: returns the 4 values you showed in your example.
    """
    paras = parsed["paragraphs"]
    cases = parsed["case_studies"]
    out = {
        "para_1": paras.get(1, {}).get("text", ""),
        "para_2": paras.get(2, {}).get("text", ""),
    }
    cs1 = cases.get(1, {})
    out["case_study_1_prompt"] = (cs1.get("prompt") or {}).get("text", "")
    out["case_study_1_note"]   = (cs1.get("note") or {}).get("text", "")
    return out
# ================================
# END NEW PDF PARSER
# ================================

# ========= Parse the Course Booklet once (NEW parser) =========
PDF_PATH = "assets/EUCapML - Course Booklet.pdf"

@st.cache_data(show_spinner=False)
def parse_booklet_new_parser(path: str):
    return parse_pdf_all_from_path(path)

try:
    PARSED_BOOKLET = parse_booklet_new_parser(PDF_PATH)
except Exception as e:
    PARSED_BOOKLET = None
    st.warning(f"NEW parser: could not parse booklet at {PDF_PATH}: {e}")
# ==============================================================

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

DEFAULT_WEIGHTS = {"similarity": 0.4, "coverage": 0.6}

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
import json

# ---------- Helpers for robust JSON extraction ----------
def _first_json_block(s: str):
    """Extract the first JSON object/array from a string (handles ```json ... ```)."""
    if not s:
        return None
    m = re.search(r"```json\s*(\{.*?\}|\[.*?\])\s*```", s, flags=re.S | re.I)
    if m:
        return m.group(1)
    m = re.search(r"(\{.*?\}|\[.*?\])", s, flags=re.S)
    return m.group(1) if m else None

def _coerce_issues(parsed) -> list[dict]:
    """Validate & normalize issues into [{name, keywords, importance}] with sane bounds."""
    if parsed is None:
        return []
    issues = parsed.get("issues") if isinstance(parsed, dict) else parsed
    if not isinstance(issues, list):
        return []
    out = []
    for it in issues:
        if not isinstance(it, dict):
            continue
        name = (it.get("name") or "").strip()
        kws  = it.get("keywords") or []
        try:
            importance = int(it.get("importance", 5))
        except Exception:
            importance = 5
        if not name or not isinstance(kws, list):
            continue
        kws = [k.strip() for k in kws if isinstance(k, str) and k.strip()]
        kws = list(dict.fromkeys(kws))[:8]           # dedupe + cap
        importance = max(1, min(10, importance))     # clamp
        if not kws:
            continue
        out.append({"name": name, "keywords": kws, "importance": importance})
    # keep 4–10 issues if possible
    return out[:10]

def _try_parse_json(raw: str):
    """Parse JSON from raw LLM output, being tolerant to fences."""
    if not raw:
        return None
    block = _first_json_block(raw)
    try:
        if block:
            return json.loads(block)
        return json.loads(raw)
    except Exception:
        return None

# ---------- Purely algorithmic fallback (scales to any case) ----------
def _auto_issues_from_text(text: str, max_issues: int = 8) -> list[dict]:
    """
    Build issues from the MODEL_ANSWER only (no LLM, no hard-coded topics):
      1) Split into sentences.
      2) TF-IDF over uni/bi/tri-grams → top phrases.
      3) Light boost for law-like references (Art/§/C‑number/etc.).
      4) For each phrase, derive 3–6 keywords from its best-matching sentences.
      5) Importance decays with rank (top gets 10).
    """
    clean = (text or "").strip()
    if not clean:
        return []

    # Sentences (simple heuristic)
    sents = [s.strip() for s in re.split(r"(?<=[\.\!\?])\s+", clean) if s.strip()]
    if not sents:
        sents = [clean]

    # TF-IDF over n-grams
    stop = set([
        "the","a","an","and","or","of","to","for","in","on","by","with","without",
        "be","is","are","was","were","as","that","this","those","these","it","its",
        "students","should","would","could","also","however","therefore","pursuant",
        "within","meaning","includes","including"
    ])
    vec = TfidfVectorizer(
        ngram_range=(1,3),
        max_features=3000,
        stop_words="english"  # keep generic; we already added a few extra above
    )
    X = vec.fit_transform(sents)   # shape: [n_sents, n_terms]
    terms = np.array(vec.get_feature_names_out())

    # Aggregate importance per term across sentences (sum TF-IDF)
    tfidf_sum = np.asarray(X.sum(axis=0)).ravel()

    # Light domain-agnostic boosts for law-like patterns (generic & scalable)
    def _boost_for(term: str) -> float:
        t = term.lower()
        boost = 1.0
        if re.search(r"\b(?:art|article)\s*\d", t): boost *= 1.35
        if "§" in term or re.search(r"\b§\s*\d", term): boost *= 1.35
        if re.search(r"\b(?:regulation|directive|mifid|mar|td|wpüg|wphg|pr)\b", t): boost *= 1.2
        if re.search(r"\bC[\u2011\u2010\u202F\u00A0\-–—]?\d+\/\d+\b", term, re.I): boost *= 1.25  # C‑123/45
        if re.search(r"\bprospectus\b|\binside information\b|\bacting in concert\b", t): boost *= 1.1
        return boost

    scores = np.array([tfidf_sum[i] * _boost_for(terms[i]) for i in range(len(terms))])

    # Pick top n distinct phrases, prefer longer n-grams, avoid nested duplicates
    idx = scores.argsort()[::-1]
    chosen = []
    seen = set()
    for i in idx:
        phrase = terms[i]
        # discard trivial tokens
        if len(phrase) < 3 or phrase.lower() in stop:
            continue
        # avoid keeping a phrase fully contained in an already chosen longer phrase
        if any(phrase in c or c in phrase for c in seen):
            continue
        seen.add(phrase)
        chosen.append((phrase, scores[i]))
        if len(chosen) >= max_issues:
            break

    # Map term → sentences it appears in
    term2sent_ix = {t: [] for t, _ in chosen}
    for si, s in enumerate(sents):
        low = s.lower()
        for t, _ in chosen:
            if t.lower() in low:
                term2sent_ix[t].append(si)

    def _keywords_from_sentences(term: str, sent_ix: list[int]) -> list[str]:
        # Collect top co-occurring tokens from the best 2 sentences
        sent_ix = (sent_ix or [])[:2]
        bag = []
        for si in sent_ix:
            bag.extend(re.findall(r"[A-Za-z§][A-Za-z0-9()§.\-\/]*", sents[si]))
        # simple normalisation
        bag = [w.strip(".,;:()").lower() for w in bag]
        bag = [w for w in bag if len(w) >= 3 and w not in stop]
        # keep some legal markers intact
        # Rank by frequency
        freqs = {}
        for w in bag:
            freqs[w] = freqs.get(w, 0) + 1
        # seed with the term itself (split into tokens)
        seeds = [term.lower()]
        # choose top 3–6 keywords
        ordered = sorted(freqs.items(), key=lambda kv: (-kv[1], -len(kv[0])))
        kws = []
        for w, _ in ordered:
            if w not in seeds and w not in kws:
                kws.append(w)
            if len(kws) >= 5:
                break
        # Ensure the term itself (and a title-cased variant) are present
        base = term.strip()
        kws = [base] + kws
        # Unique + cap length
        out = []
        for k in kws:
            if k and k not in out:
                out.append(k)
        return out[:6] if len(out) >= 3 else out  # keep 3–6 if possible

    issues = []
    for rank, (term, sc) in enumerate(chosen, start=1):
        # Name: title-case lightly but keep 'Art'/§ style intact
        name = re.sub(r"\b(article|art)\b", "Art", term, flags=re.I)
        name = name.replace("§", "§ ").replace("  ", " ").strip()
        name = name[:1].upper() + name[1:]
        kws = _keywords_from_sentences(term, term2sent_ix.get(term, []))
        # Importance: simple decay from top (10..max(4,10-(n-1)))
        importance = max(4, 11 - rank)
        issues.append({"name": name, "keywords": kws, "importance": importance})

    return issues
# ---------- Main extractor (no hard-coded topics) ----------
def extract_issues_from_model_answer(model_answer: str, llm_api_key: str) -> list[dict]:
    """
    Try LLM with strict JSON contract (with two repair retries).
    If it still fails, fall back to automatic text mining (no hard-coded issues).
    """
    # Guard
    model_answer = (model_answer or "").strip()
    if not model_answer:
        return []

    # 1) LLM attempt with strict JSON instructions
    sys = "Respond with VALID JSON only: either an array or {\"issues\": [...]}. No prose, no fences."
    user = (
        "Extract the key issues from the MODEL ANSWER.\n"
        "Return JSON ONLY (no code fences). Each item:\n"
        "{ \"name\": \"short issue name\", \"keywords\": [\"3-8 indicative phrases\"], \"importance\": 1-10 }\n\n"
        "MODEL ANSWER:\n" + model_answer
    )
    messages = [
        {"role": "system", "content": sys},
        {"role": "user", "content": user},
    ]
    raw = call_groq(messages, api_key=llm_api_key, model_name="llama-3.1-8b-instant", temperature=0.0, max_tokens=900)
    parsed = _try_parse_json(raw)
    issues = _coerce_issues(parsed)

    # 2) If parsing failed or no items, try a small JSON-repair step once
    if not issues and raw:
        repair_msgs = [
            {"role": "system", "content": "Fix JSON. Output VALID JSON only (no prose, no fences)."},
            {"role": "user", "content": f"Make this into valid JSON array (or {{\"issues\": [...]}}):\n{raw}"},
        ]
        raw2 = call_groq(repair_msgs, api_key=llm_api_key, model_name="llama-3.1-8b-instant", temperature=0.0, max_tokens=900)
        parsed2 = _try_parse_json(raw2)
        issues = _coerce_issues(parsed2)

    # 3) Final fallback: automatic issue mining from text (generic & scalable)
    if not issues:
        issues = _auto_issues_from_text(model_answer, max_issues=8)

    return issues

def generate_rubric_from_model_answer(student_answer: str, model_answer: str, backend, llm_api_key: str, weights: dict) -> dict:
    extracted_issues = extract_issues_from_model_answer(model_answer, llm_api_key)
    if not extracted_issues:
        return {
            "similarity_pct": 0.0,
            "coverage_pct": 0.0,
            "final_score": 0.0,
            "per_issue": [],
            "missing": [],
            "substantive_flags": []
        }

    embs = embed_texts([student_answer, model_answer], backend)
    sim = cos_sim(embs[0], embs[1])
    sim_pct = max(0.0, min(100.0, 100.0 * (sim + 1) / 2))

    per_issue, tot, got = [], 0, 0
    for issue in extracted_issues:
        pts = issue.get("importance", 5) * 2
        tot += pts
        sc, hits = coverage_score(student_answer, {"keywords": issue["keywords"], "points": pts})
        got += sc
        per_issue.append({
            "issue": issue["name"],
            "max_points": pts,
            "score": sc,
            "keywords_hit": hits,
            "keywords_total": issue["keywords"],
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
        "substantive_flags": substantive_flags
    }

def filter_model_answer_and_rubric(selected_question: str, model_answer: str, api_key: str) -> tuple[str, list[dict]]:
    # Find section starts anchored at line-beginnings
    m1 = re.search(r"(?m)^\s*1\.\s", model_answer)
    m2 = re.search(r"(?m)^\s*2\.\s", model_answer)
    m3 = re.search(r"(?m)^\s*3\.\s", model_answer)

    start1 = m1.start() if m1 else 0
    start2 = m2.start() if m2 else None
    start3 = m3.start() if m3 else None
    end = len(model_answer)

    if selected_question == "Question 1":
        lo, hi = start1, (start2 if start2 is not None else end)
    elif selected_question == "Question 2":
        lo, hi = (start2 if start2 is not None else 0), (start3 if start3 is not None else end)
    elif selected_question == "Question 3":
        lo, hi = (start3 if start3 is not None else 0), end
    else:
        lo, hi = 0, end

    model_answer_filtered = model_answer[lo:hi].strip()
    extracted_issues = extract_issues_from_model_answer(model_answer_filtered, api_key)
    return model_answer_filtered, extracted_issues
    
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

# ---- Manual relevance terms per question ----
def manual_chunk_relevant(text: str, extracted_keywords: list[str], user_query: str = "") -> bool:
    q_terms = [w.lower() for w in re.findall(r"[A-Za-zÄÖÜäöüß0-9\-]{3,}", user_query or "")]
    keys = [k.lower() for k in (extracted_keywords or [])]
    tgt = text.lower()
    return any(k in tgt for k in (keys + q_terms))


def retrieve_snippets_with_manual(student_answer, model_answer_filtered, pages, backend,
                                  extracted_keywords, user_query: str = "",
                                  top_k_pages=8, chunk_words=170):
                                      
    # ---- Load & chunk Course Booklet with page/para/case metadata
    manual_chunks, manual_metas = [], []
    try:
        manual_chunks, manual_metas = extract_manual_chunks_with_refs(
            "assets/EUCapML - Course Booklet.pdf",
            chunk_words_hint=chunk_words
        )
    except Exception as e:
        st.warning(f"Could not load course manual: {e}")


    # ✅ Filter manual chunks using keywords + the user's query AND case numbers, if any
    selected_q = st.session_state.get("selected_question", "Question 1")
    uq_cases = detect_case_numbers(user_query or "")
    filtered_chunks, filtered_metas = [], []
    for ch, m in zip(manual_chunks, manual_metas):
        has_kw = manual_chunk_relevant(ch, extracted_keywords, user_query)
        case_match = bool(uq_cases and set(uq_cases).intersection(set(m.get("cases") or [])))
        if has_kw or case_match:
            filtered_chunks.append(ch)
            filtered_metas.append(m)
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
    # ---- Prepare web chunks (fixed: keep meta 1:1 with chunks)
    web_chunks, web_meta = [], []
    selected_q = st.session_state.get("selected_question", "Question 1")
    for i, p in enumerate(pages):
        text = p.get("text", "")
        if not text:
            continue
    # Optional relevance filter (keep if you added MANUAL_KEY_TERMS):
        if 'web_page_relevant' in globals() and not web_page_relevant(text, extracted_keywords):
            continue

        chunks_i = split_into_chunks(text, max_words=chunk_words)
        for ch in chunks_i:
            web_chunks.append(ch)
            web_meta.append((i + 1, p["url"], p["title"]))  # append meta PER CHUNK

    # ---- Build combined corpus
    all_chunks = manual_chunks + web_chunks
    all_meta   = manual_meta   + web_meta
    # Defensive: keep chunks and meta in lockstep
    if len(all_chunks) != len(all_meta):
        m = min(len(all_chunks), len(all_meta))
        all_chunks = all_chunks[:m]
        all_meta   = all_meta[:m]

    # Query vector built from student + model slice
    query = "\n\n".join([s for s in [user_query, student_answer, model_answer_filtered] if s])
    embs = embed_texts([query] + all_chunks, backend)
    qv, cvs = embs[0], embs[1:]
    sims = [cos_sim(qv, v) for v in cvs]
    idx = np.argsort(sims)[::-1]

    # ✅ Similarity floor to keep only reasonably relevant snippets
    MIN_SIM = 0.20  # tune 0.10–0.18 if needed

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
        "- Cite ONLY using numeric brackets that match the SOURCES list (e.g., [1], [2]).\n"
        "- NEVER write the literal placeholder “[n]”.\n"
        "- Never invent Course Booklet references (pages, paragraphs, cases). Only cite the numbered SOURCES.\n\n"
        "- Do NOT fabricate page/para/case numbers.\n"
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

def build_feedback_prompt(student_answer: str,
                          rubric: dict,
                          model_answer: str,
                          sources_block: str,
                          excerpts_block: str) -> str:
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

SOURCES (numbered; cite using [1], [2], … ONLY from this list):
{sources_block}

EXCERPTS (quote sparingly; cite using [1], [2], …):
{excerpts_block}

TASK (you MUST follow these steps):
1) Extract the student's core CLAIMS as short bullets (no more than 3–5 bullets). For EACH claim, give it one of the labels "Correct" / "Incorrect" / "Not supported".
2) Where you label a student core claim as incorrect, explain briefly why.
3) Where important aspects are missing, explain what aspects are missing, and why they are important. 
4) Give concise IMPROVEMENT TIPS (1–3 bullets) tied to the rubric issues, ideally with a numeric citation.
5) End with a single-sentence CONCLUSION.

RULES:
- Use numeric citations matching the SOURCES list (e.g., [1], [2]); never “[n]”.
- Do not invent any Course Booklet page/para/case reference; cite only from SOURCES.
- Be concrete; avoid tautologies (e.g., “refer to Art 7(1) instead of Art 7(1)”).
- ≤400 words total.
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

# ------------------------------ Chat and Feebdack Helpers ----------
def web_page_relevant(text: str, extracted_keywords: list[str]) -> bool:
    return any(kw.lower() in text.lower() for kw in extracted_keywords)

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

# --- Citation post-processing & filtering ---
def parse_cited_indices(text: str) -> list[int]:
    """Return sorted unique [n] indices used in text."""
    try:
        return sorted(set(int(x) for x in re.findall(r"\[(\d+)\]", text or "")))
    except Exception:
        return []

def filter_sources_by_indices(source_lines: list[str], used: list[int]) -> list[str]:
    """Return only those lines whose [n] was actually cited; preserve numbering."""
    if not used:
        return []
    out = []
    for n in used:
        if 1 <= n <= len(source_lines):
            out.append(source_lines[n - 1])
    return out

# Paragraph markers may appear as "para. 115", "paragraph 115", "Rn. 115", "[115]", "¶ 115"

_para_patterns = [
    re.compile(r"\bpara(?:graph)?\.?\s*(\d{1,4})\b", re.I),
    re.compile(r"\brn\.?\s*(\d{1,4})\b", re.I),
    re.compile(r"\[\s*(\d{1,4})\s*\]"),
    re.compile(r"¶\s*(\d{1,4})"),
]

# Case markers: must have the word "Case", "Case Study"
_case_patterns = [
    re.compile(r"\bCase\s*Study\s*(\d{1,4})\b", re.I),
    re.compile(r"\bCase\s*(\d{1,4})\b", re.I),
    ]

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
        nums += pat.findall(text or "")
    out = []
    for x in nums:
        n = int(x)
        if n not in out:
            out.append(n)
    return out

@st.cache_resource(show_spinner=False)

def _median_or_default(xs, default=12.0):
    xs = [x for x in xs if isinstance(x, (int, float))]
    return stats.median(xs) if xs else default

def _dehyphenate_join(prev: str, curr: str) -> str:
    """
    Join two line fragments, removing soft hyphenation like: "disclo-" + "sure" -> "disclosure".
    Only if prev ends with '-' and curr starts with lowercase letter.
    """
    if prev.endswith("-") and curr and curr[:1].islower():
        return prev[:-1] + curr
    # otherwise join with space (avoid double spaces)
    if prev and curr:
        if prev.endswith((" ", "—", "–")) or curr.startswith((" ", "—", "–")):
            return prev + curr
        return prev + " " + curr
    return prev or curr

# ============ Deterministic booklet parsing helpers ============
# Accepts: "12", "12.", "12)", "12 –", "12 —" etc. at the **very start** of a line.
LEAD_NUM_RE   = re.compile(r"^\s*(\d{1,4})(?:[.)]|\s*[-–—])?\s+")
CASE_LINE_RE  = re.compile(r"^\s*Case\s*Study\s*(\d{1,4})\b", re.I)

from typing import Optional

def _page_lines_with_spans(page) -> list[dict]:
    """
    Return ordered line dicts with geometry + spans:
      [{'x0','y0','x1','y1','text','spans':[{'text','size','font','bbox'...}, ...]}, ...]
    """
    d = page.get_text("dict")
    out = []
    for blk in d.get("blocks", []):
        if blk.get("type") != 0:
            continue
        for ln in blk.get("lines", []):
            spans = ln.get("spans", [])
            txt = "".join(s.get("text", "") for s in spans)
            if not txt.strip():
                continue
            x0, y0, x1, y1 = ln.get("bbox", [0, 0, 0, 0])
            out.append({"x0": x0, "y0": y0, "x1": x1, "y1": y1, "text": txt, "spans": spans})
    out.sort(key=lambda L: (L["y0"], L["x0"]))
    return out

def _median(xs, default=12.0):
    xs = [x for x in xs if isinstance(x, (int, float))]
    return stats.median(xs) if xs else default

def _body_left_threshold(lines: list[dict]) -> float:
    """
    Left edge of main body column = median x0 of lines (robust even if margin numbers exist).
    """
    xs = [L["x0"] for L in lines]
    if not xs:
        return 60.0
    # Use 40th percentile as a robust body-left estimate (ignores a few very-left gutter lines)
    xs_sorted = sorted(xs)
    idx = max(0, min(len(xs_sorted)-1, int(0.40 * len(xs_sorted))))
    return xs_sorted[idx]

def _dehyphen_join(prev: str, curr: str) -> str:
    if not prev:
        return curr
    if prev.endswith("-") and curr and curr[:1].islower():
        return prev[:-1] + curr
    # normal join with single space
    return (prev + " " + curr).strip()

def _match_leading_number(line_text: str) -> Optional[int]:
    m = LEAD_NUM_RE.match(line_text)
    if not m:
        return None
    try:
        n = int(m.group(1))
        return n
    except Exception:
        return None

def _strip_leading_number(line_text: str) -> str:
    """
    Remove the leading '12', '12.', '12)', '12 –', etc., and return the remaining text.
    Ensures we do **not** lose the first line of the paragraph.
    """
    return LEAD_NUM_RE.sub("", line_text, count=1).strip()

def _find_case_starts(lines: list[dict], body_left: float) -> list[dict]:
    """
    Return [{'case':N,'y0':<line top>}, ...] for lines starting with 'Case Study N ...'
    that appear **in the body column** (not the left number gutter).
    """
    hits = []
    for L in lines:
        # keep only lines that begin near/at the body column
        if L["x0"] < body_left - 3.0:
            continue
        m = CASE_LINE_RE.match(L["text"])
        if m:
            hits.append({"case": int(m.group(1)), "y0": L["y0"]})
    hits.sort(key=lambda d: d["y0"])
    return hits

def _extract_page_paragraphs(page) -> tuple[list[dict], list[dict]]:
    """
    Parse a page into anchored paragraphs and case-start markers.

    Returns:
      para_items: [{'para':N, 'y0':float, 'text':str}]
      case_starts: [{'case':K, 'y0':float}]
    """
    lines = _page_lines_with_spans(page)
    if not lines:
        return [], []

    body_left = _body_left_threshold(lines)

    para_items: list[dict] = []
    case_starts = _find_case_starts(lines, body_left)

    cur_para_num = None
    cur_para_y0  = None
    cur_text     = ""

    for L in lines:
        txt = L["text"].strip()

        # New paragraph if this line starts with a leading number
        n = _match_leading_number(txt)
        if n is not None:
            # flush previous paragraph
            if cur_para_num is not None and cur_text.strip():
                para_items.append({"para": cur_para_num, "y0": cur_para_y0, "text": cur_text.strip()})
            # start new paragraph; keep the **rest of this line** (first-line text!)
            cur_para_num = n
            cur_para_y0  = L["y0"]
            cur_text     = _strip_leading_number(txt)
        else:
            # continuation line for current paragraph
            if cur_para_num is not None:
                cur_text = _dehyphen_join(cur_text, txt)
            else:
                # lines before the first numbered paragraph on the page: ignore for numbered parsing
                pass

    # flush trailing paragraph
    if cur_para_num is not None and cur_text.strip():
        para_items.append({"para": cur_para_num, "y0": cur_para_y0, "text": cur_text.strip()})

    return para_items, case_starts
# ============ /Deterministic booklet parsing helpers ============
def extract_manual_chunks_with_refs(pdf_path: str, chunk_words_hint: int = 170) -> tuple[list[str], list[dict]]:
    """
    Deterministic extraction:
      • Each chunk corresponds to one numbered paragraph (paras=[N]).
      • 'case_section' stores the enclosing "Case Study K" based on the nearest
        preceding 'Case Study K ...' line (persists across pages).
      • Very long paragraphs are split by sentences near 'chunk_words_hint' while
        keeping the same anchors.
    """
    chunks, metas = [], []
    try:
        doc = fitz.open(pdf_path)
    except Exception:
        return [], []

    current_case_section: Optional[int] = None  # carried across pages

    for pno in range(len(doc)):
        page = doc.load_page(pno)
        page_label = page.get_label() or str(pno + 1)

        para_items, case_starts = _extract_page_paragraphs(page)
        # walk down the page; whenever a case start appears above the paragraph, update section
        case_idx = 0
        case_starts = sorted(case_starts, key=lambda d: d["y0"])

        for it in sorted(para_items, key=lambda d: d["y0"]):
            # advance case section
            while case_idx < len(case_starts) and case_starts[case_idx]["y0"] <= it["y0"] + 0.5:
                current_case_section = case_starts[case_idx]["case"]
                case_idx += 1

            para_no = it["para"]
            text    = it["text"]

            # split long paragraphs (keep anchors)
            parts = [text]
            words = text.split()
            if len(words) > chunk_words_hint * 2:
                bits = re.split(r"(?<=[\.\?\!…])\s+", text)
                cur, acc, parts = [], 0, []
                for s in bits:
                    cur.append(s); acc += len(s.split())
                    if acc >= chunk_words_hint:
                        parts.append(" ".join(cur).strip()); cur, acc = [], 0
                if cur:
                    parts.append(" ".join(cur).strip())

            for part in parts:
                if not part:
                    continue
                chunks.append(part)
                metas.append({
                    "pdf_index": pno,
                    "page_label": page_label,
                    "page_num": pno + 1,
                    "paras": [para_no],          # <- primary anchor
                    "cases": [],                 # <- not used for paragraph chunks
                    "case_section": current_case_section,  # <- enclosing Case Study (may be None)
                    "file": "EUCapML - Course Booklet.pdf",
                })

    doc.close()
    return chunks, metas


def format_manual_citation(meta: dict) -> str:
    paras = meta.get("paras") or []
    cases = meta.get("cases") or []
    page_label = meta.get("page_label") or ""
    pdf_p = meta.get("page_num")

    para_anchor = f"para. {paras[0]}" if paras else ""
    case_anchor = f"Case Study {cases[0]}" if cases else ""

    anchors = []
    if case_anchor:
        anchors.append(case_anchor)
    if page_label or pdf_p:
        if page_label and pdf_p:
            anchors.append(f"p. {page_label} (PDF {pdf_p})")
        elif page_label:
            anchors.append(f"p. {page_label}")
        else:
            anchors.append(f"PDF {pdf_p}")
    if para_anchor:
        anchors.append(para_anchor)

    return "Course Booklet — " + (", ".join(anchors) if anchors else "(no page anchor found)")

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
        st.success("PIN accepted. By clicking CONTINUE below you accept that this tool uses artificial intelligence and large language models, and that accordingly, answers may not be accurate. No liability is accepted for use of this tool.")
        if st.button("Continue"):
            st.rerun()
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
    # ---- Tiny diagnostic to confirm parsing ----
    with st.sidebar:
        st.subheader("🔎 NEW parser sanity check")
        if PARSED_BOOKLET:
            paras = PARSED_BOOKLET["paragraphs"]      # {n: {"text","page"}}
            cases = PARSED_BOOKLET["case_studies"]     # {n: {"prompt":{...},"note":{...}}}
    
            st.caption(f"Found {len(paras)} numbered paragraphs and {len(cases)} case studies.")
            # ---- Paragraph preview ----
            if paras:
                n = st.number_input("Paragraph number", min_value=min(paras.keys()),
                                     max_value=max(paras.keys()), value=min(paras.keys()), step=1)
                p = paras.get(int(n))
                if p:
                    st.write(f"**Paragraph {int(n)} (page {p['page']})**")
                    st.text(p["text"][:2500])
            # ---- Case Study preview ----
            if cases:
                k = st.number_input("Case Study #", min_value=min(cases.keys()),
                                     max_value=max(cases.keys()), value=min(cases.keys()), step=1,
                                     key="cs_preview_new")
                cs = cases.get(int(k), {})
                if "prompt" in cs:
                    st.write(f"**Case Study {int(k)} — prompt (page {cs['prompt']['page']})**")
                    st.text(cs["prompt"]["text"][:2500])
                if "note" in cs:
                    st.write(f"**Case Study {int(k)} — note (page {cs['note']['page']})**")
                    st.text(cs["note"]["text"][:2500])
        else:
            st.info("NEW parser is not available (file missing or parse error).")
        
        st.subheader("🔎 Parser check (dev)")
        test_page = st.number_input("PDF page (1-based)", min_value=1, value=7, step=1)
        if st.checkbox("Show paragraphs + case-section on this page"):
            try:
                doc = fitz.open("assets/EUCapML - Course Booklet.pdf")
                p = doc.load_page(int(test_page) - 1)
                para_items, case_starts = _extract_page_paragraphs(p)
                st.write(f"Case starts on this page: {[c['case'] for c in case_starts] or '—'}")
                for it in para_items[:15]:
                    snip = it["text"][:180] + ("…" if len(it["text"]) > 180 else "")
                    st.write(f"• para {it['para']} (y0={it['y0']:.1f}): {snip}")
                doc.close()
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
                model_answer_filtered, extracted_issues = filter_model_answer_and_rubric(selected_question, MODEL_ANSWER, api_key)
                extracted_keywords = [kw for issue in extracted_issues for kw in issue.get("keywords", [])]
                rubric = generate_rubric_from_model_answer(
                    student_answer,
                    model_answer_filtered,
                    backend,
                    api_key,
                    DEFAULT_WEIGHTS
                )
                                
                top_pages, source_lines = [], []
                if enable_web:
                    pages = collect_corpus(student_answer, "", max_fetch=22)
                    top_pages, source_lines = retrieve_snippets_with_manual(
                        student_answer, model_answer_filtered, pages, backend, extracted_keywords,
                        user_query="", top_k_pages=max_sources, chunk_words=170
                    )
                    
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

            st.markdown("### 🧭 Narrative Feedback")
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
                    # Safety net: strip any stray “[n]” placeholders
                    reply = re.sub(r"\[(?:n|N)\]", "", reply or "")
                    st.write(reply)

                    # Show only the sources actually cited in the narrative:
                    used_idxs = parse_cited_indices(reply)
                    display_source_lines = filter_sources_by_indices(source_lines, used_idxs) or source_lines
                else:
                    st.info("LLM unavailable. See corrections above and the issue breakdown.")
            else:
                st.info("No GROQ_API_KEY found in secrets/env. Deterministic scoring and corrections shown above.")

            if source_lines:
                with st.expander("📚 Sources used"):
                    for line in display_source_lines:
                        st.markdown(f"- {line}")

with colB:
    st.markdown("### 💬 Tutor Chat: Ask me anything!")
    
    # --- state ---
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "chat_draft" not in st.session_state:
        st.session_state.chat_draft = ""

    # --- composer ---
    c1, c2, c3, c4 = st.columns([6, 1, 1, 2])
    with c1:
        st.text_area(
            "You can use this chat to ask for help with creating an answer, follow-up questions on feedback given by this app, etc.",
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
                model_answer_filtered, extracted_issues = filter_model_answer_and_rubric(
                    st.session_state.get("selected_question", "Question 1"),
                    MODEL_ANSWER,
                    api_key
                )
                extracted_keywords = [kw for issue in extracted_issues for kw in issue.get("keywords", [])]

                pages = collect_corpus(student_answer, user_q, max_fetch=20)
                top_pages, source_lines = retrieve_snippets_with_manual(
                    student_answer, model_answer_filtered, pages, backend, extracted_keywords,
                    user_query=user_q, top_k_pages=max_sources, chunk_words=170
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
        # ---- SAFETY NET (CHAT): normalize citations + show only cited sources ----
        # Safety net: remove literal “[n]”
        reply = re.sub(r"\[(?:n|N)\]", "", reply or "")
        
        # Keep only sources actually cited in this reply
        used_idxs = parse_cited_indices(reply)
        msg_sources = filter_sources_by_indices(source_lines, used_idxs) or source_lines[:]

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": reply,
            "sources": msg_sources
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
    "ℹ️ **Notes**: This app is authored by Stephan Balthasar. It provides feedback based on artificial intelligence and large language models, and as a result, answers can be inaccurate. " 
    "Students are advised to use caution when using the feedback engine and chat functions."
)
