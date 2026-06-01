"""
Resume Ranking Engine  —  AI RecruitX
======================================
Core Algorithms:
  1. TF-IDF Vectorisation + Cosine Similarity  (sklearn)
  2. Deep AI/ML Ensemble Classifier            (Naive Bayes + SVM + Logistic Regression)
  3. Hidden Text Detection                     (pdfplumber char-level color/size scan)
  4. Overlapping Date Detection                (experience timeline conflict check)
  5. Keyword Stuffing / Anomaly Detection      (existing)

Scoring Weights
---------------
  TF-IDF Cosine Similarity   : 40%
  AI/ML Ensemble Classifier  : 20%
  Skill Keyword Match        : 25%
  Experience Relevance       : 10%
  Structural (Edu+Proj+Cert) :  5%
"""

import re
import hashlib
import numpy as np
from itertools import combinations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler

from utils.parser import SKILLS_VOCABULARY


# ═══════════════════════════════════════════════════════════════
# SECTION 0 – Utilities
# ═══════════════════════════════════════════════════════════════

def get_text_hash(text: str) -> str:
    """MD5 hash of cleaned text — used to identify duplicate resumes."""
    cleaned = re.sub(r'\s+', '', text).lower()
    return hashlib.md5(cleaned.encode('utf-8')).hexdigest()


_DISPLAY_OVERRIDES = {
    'c++': 'C++', 'c#': 'C#', 'node.js': 'Node.js',
    'ci/cd': 'CI/CD', 'rest api': 'REST API',
}

def _fmt_skill(s: str) -> str:
    return _DISPLAY_OVERRIDES.get(s, s.title())


def _clean_text(text: str) -> str:
    """Lowercase → strip non-alpha → collapse whitespace."""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


# ═══════════════════════════════════════════════════════════════
# SECTION 1 – TF-IDF + Cosine Similarity
# ═══════════════════════════════════════════════════════════════

def compute_tfidf_cosine_similarity(resume_text: str, job_description: str) -> float:
    """
    Step-by-step:
      corpus       = [resume_text, job_description]
      tfidf_matrix = TfidfVectorizer(...).fit_transform(corpus)
      similarity   = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]
    Returns float in [0, 1].
    """
    resume_clean = _clean_text(resume_text)
    job_clean    = _clean_text(job_description)
    if not resume_clean or not job_clean:
        return 0.0

    corpus = [resume_clean, job_clean]

    vectorizer = TfidfVectorizer(
        stop_words='english',
        ngram_range=(1, 2),
        max_features=5000,
        sublinear_tf=True,
    )
    tfidf_matrix = vectorizer.fit_transform(corpus)

    # cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])  →  shape (1, 1)
    similarity = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]
    return float(np.clip(similarity, 0.0, 1.0))


# ═══════════════════════════════════════════════════════════════
# SECTION 2 – Deep AI/ML Ensemble Classifier
# ═══════════════════════════════════════════════════════════════
# Three sklearn models trained on a labeled corpus of resume
# sentences labelled as "relevant" (1) or "irrelevant" (0)
# relative to a software/tech job description.
#
# Models used:
#   • MultinomialNB      – fast probabilistic baseline (Naive Bayes)
#   • LinearSVC          – maximum-margin linear classifier (SVM)
#   • LogisticRegression – probabilistic linear model (gives calibrated probs)
#
# Final score = mean of the three models' predicted probabilities
# that the resume belongs to class "relevant".
# ═══════════════════════════════════════════════════════════════

# ── Training corpus ────────────────────────────────────────────
# 1 = relevant to a software/tech job
# 0 = not relevant
_TRAIN_CORPUS = [
    # Relevant (1)
    ("python developer flask rest api django sql postgresql docker kubernetes git aws", 1),
    ("software engineer machine learning tensorflow pytorch deep learning nlp model training", 1),
    ("full stack developer react angular node express javascript typescript html css", 1),
    ("data scientist pandas numpy scikit learn data analysis jupyter notebook tableau", 1),
    ("devops engineer ci cd jenkins docker kubernetes linux bash cloud aws azure gcp", 1),
    ("backend developer java spring boot microservices rest api sql oracle redis", 1),
    ("android developer kotlin java android studio firebase rest api git", 1),
    ("frontend developer html css javascript react redux typescript bootstrap responsive", 1),
    ("cloud architect aws azure gcp terraform ansible kubernetes docker infrastructure", 1),
    ("database administrator postgresql mysql oracle mongodb redis performance tuning", 1),
    ("cybersecurity analyst penetration testing ethical hacking linux python network", 1),
    ("ai engineer pytorch tensorflow nlp computer vision model deployment api python", 1),
    ("web developer php laravel html css javascript mysql git linux ubuntu", 1),
    ("software tester selenium junit pytest automation testing qa api testing", 1),
    ("data engineer spark hadoop kafka airflow python sql etl pipeline aws", 1),
    ("python flask sql git docker rest api internship bachelor computer science", 1),
    ("react node.js postgresql aws git agile scrum software development b.tech", 1),
    ("machine learning scikit learn nlp text classification python pandas feature engineering", 1),
    ("java microservices spring boot kubernetes jenkins ci cd devops git maven", 1),
    ("angular typescript rxjs rest api html css git agile sprint planning", 1),
    # Not relevant (0)
    ("marketing manager social media campaigns brand strategy excel powerpoint", 0),
    ("accountant financial reporting tax compliance ledger balance sheet excel", 0),
    ("chef cooking food preparation recipes kitchen management", 0),
    ("teacher classroom management curriculum lesson planning students", 0),
    ("sales executive client relationship revenue target cold calling crm", 0),
    ("graphic designer adobe photoshop illustrator creativity branding logo", 0),
    ("hr recruiter talent acquisition onboarding payroll compliance", 0),
    ("doctor medicine surgery clinical trials patient care hospital", 0),
    ("lawyer legal advice contracts litigation corporate law", 0),
    ("mechanical engineer autocad solidworks manufacturing cnc lathe milling", 0),
    ("civil engineer structural design autocad construction project management", 0),
    ("journalist writing reporting media news editorial publishing", 0),
    ("logistics coordinator supply chain warehouse inventory freight", 0),
    ("customer service call centre complaints resolution communication", 0),
    ("fashion designer clothing apparel sketching textile trends", 0),
]

_TRAIN_TEXTS  = [t for t, _ in _TRAIN_CORPUS]
_TRAIN_LABELS = [l for _, l in _TRAIN_CORPUS]

# ── Build shared TF-IDF features for ML models ─────────────────
_ML_VECTORIZER = TfidfVectorizer(
    stop_words='english',
    ngram_range=(1, 2),
    max_features=3000,
    sublinear_tf=True,
)
_X_TRAIN = _ML_VECTORIZER.fit_transform(_TRAIN_TEXTS)

# Model 1: Multinomial Naive Bayes
#   • Assumes feature independence; fast, good baseline for text
_nb_model = MultinomialNB(alpha=0.5)
_nb_model.fit(_X_TRAIN, _TRAIN_LABELS)

# Model 2: LinearSVC (Support Vector Machine)
#   • Finds the maximum-margin hyperplane in feature space
#   • Wrapped to produce probability estimates via cross-validation calibration
#   • Note: LinearSVC has no predict_proba; we use decision_function + sigmoid
_svm_model = LinearSVC(C=1.0, max_iter=2000)
_svm_model.fit(_X_TRAIN, _TRAIN_LABELS)

# Model 3: Logistic Regression
#   • Probabilistic linear model; naturally calibrated probabilities
_lr_model = LogisticRegression(C=1.0, max_iter=1000, solver='lbfgs')
_lr_model.fit(_X_TRAIN, _TRAIN_LABELS)


def _sigmoid(x: float) -> float:
    """Maps LinearSVC decision_function output to [0, 1] probability."""
    return 1.0 / (1.0 + np.exp(-x))


def compute_ml_ensemble_score(resume_text: str, job_description: str) -> float:
    """
    Runs the resume + job description through three ML classifiers
    and returns an ensemble relevance score in [0, 1].

    Pipeline:
        1. Combine resume and job description into a single document.
        2. Transform with the pre-trained TF-IDF vectorizer.
        3. Get relevance probability from each model.
        4. Return the average (soft voting ensemble).

    Returns float in [0, 1] where 1 = highly relevant to the job.
    """
    combined = _clean_text(resume_text + " " + job_description)
    if not combined.strip():
        return 0.0

    X = _ML_VECTORIZER.transform([combined])

    # NB probability of class 1 (relevant)
    nb_prob = _nb_model.predict_proba(X)[0][1]

    # SVM decision score → sigmoid to get pseudo-probability
    svm_decision = _svm_model.decision_function(X)[0]
    svm_prob = _sigmoid(svm_decision)

    # LR probability of class 1 (relevant)
    lr_prob = _lr_model.predict_proba(X)[0][1]

    # Soft voting: equal weight to all three models
    ensemble = (nb_prob + svm_prob + lr_prob) / 3.0
    return float(np.clip(ensemble, 0.0, 1.0))


# ═══════════════════════════════════════════════════════════════
# SECTION 3 – Hidden Text Detection
# ═══════════════════════════════════════════════════════════════
# Detects "invisible ink" tricks used to fool ATS systems:
#   a) Near-white text (color components all > 0.85 — hard to see)
#   b) Tiny text (font size < 2pt — invisible to human eye)
#   c) White-on-white / background-matching text
#
# Uses pdfplumber's low-level char objects which expose:
#   char['non_stroking_color'] — the fill (text) color as RGB tuple
#   char['size']               — font size in points
# ═══════════════════════════════════════════════════════════════

def detect_hidden_text(pdf_path: str) -> list:
    """
    Scans every character in the PDF using pdfplumber's char-level API.

    Hidden text tactics detected:
      1. Tiny text  : font size < 2pt (invisible to human readers)
      2. White text : non_stroking_color close to white (all RGB > 0.85)
      3. Invisible  : render_mode == 3 (text set to invisible in PDF spec)

    Returns a list of fraud flag strings. Empty list = no hidden text found.
    """
    try:
        import pdfplumber
    except ImportError:
        return []

    flags = []
    tiny_count  = 0
    white_count = 0
    invis_count = 0

    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                chars = page.chars  # list of char dicts from pdfplumber

                for char in chars:
                    text_char = char.get('text', '').strip()
                    if not text_char:
                        continue

                    # ── Check 1: Font size (tiny text) ─────────────
                    size = char.get('size', 12)
                    if isinstance(size, (int, float)) and size < 2:
                        tiny_count += 1

                    # ── Check 2: Near-white / invisible color ───────
                    color = char.get('non_stroking_color')
                    if color is not None:
                        # pdfplumber returns color as tuple (R, G, B) in [0,1]
                        # or as a single float (grayscale)
                        if isinstance(color, (int, float)):
                            # Grayscale: 1.0 = white
                            if float(color) > 0.90:
                                white_count += 1
                        elif isinstance(color, (list, tuple)) and len(color) >= 3:
                            r, g, b = float(color[0]), float(color[1]), float(color[2])
                            if r > 0.88 and g > 0.88 and b > 0.88:
                                white_count += 1

                    # ── Check 3: PDF render mode 3 = invisible ──────
                    render = char.get('render_mode', 0)
                    if render == 3:
                        invis_count += 1

    except Exception:
        # If PDF is corrupt or locked, skip detection gracefully
        return []

    if tiny_count >= 5:
        flags.append(
            f"Hidden Text: {tiny_count} characters with font size < 2pt detected (invisible text)"
        )
    if white_count >= 5:
        flags.append(
            f"Hidden Text: {white_count} near-white colored characters detected (white-on-white trick)"
        )
    if invis_count >= 3:
        flags.append(
            f"Hidden Text: {invis_count} characters with invisible render mode (PDF render_mode=3)"
        )

    return flags


# ═══════════════════════════════════════════════════════════════
# SECTION 4 – Overlapping Date Detection
# ═══════════════════════════════════════════════════════════════
# Detects timeline inconsistencies such as:
#   • Two jobs running at exact same time (full overlap)
#   • Partial date range overlaps > 2 months
#
# Parses patterns like:
#   "Jan 2021 – Mar 2022", "2020 - 2022", "June 2019 to Present"
#   "01/2020 – 03/2021", "2021–2023"
# ═══════════════════════════════════════════════════════════════

_MONTH_MAP = {
    'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4,  'may': 5, 'jun': 6,
    'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12,
    'january': 1, 'february': 2, 'march': 3,   'april': 4,
    'june': 6, 'july': 7, 'august': 8, 'september': 9,
    'october': 10, 'november': 11, 'december': 12,
}

_CURRENT_YEAR = 2025


def _parse_month_year(token: str):
    """
    Parses a date token into (year, month).
    Accepts:
        'Jan 2021', '2021', '01/2021', 'Present', 'Current', 'Now'
    Returns (year: int, month: int) or None if unparseable.
    """
    token = token.strip().lower()

    # "Present" / "Current" / "Now" → current date
    if token in ('present', 'current', 'now', 'till date', 'today'):
        return (_CURRENT_YEAR, 12)

    # MM/YYYY or MM-YYYY
    m = re.match(r'^(\d{1,2})[/-](\d{4})$', token)
    if m:
        month, year = int(m.group(1)), int(m.group(2))
        if 1 <= month <= 12 and 1900 <= year <= 2100:
            return (year, month)

    # Month name + year: "Jan 2021" or "January 2021"
    m = re.match(r'^([a-z]+)\s+(\d{4})$', token)
    if m:
        mo_str, yr_str = m.group(1), m.group(2)
        mo_num = _MONTH_MAP.get(mo_str)
        if mo_num:
            return (int(yr_str), mo_num)

    # Year only: "2021"
    m = re.match(r'^(\d{4})$', token)
    if m:
        yr = int(m.group(1))
        if 1900 <= yr <= 2100:
            return (yr, 1)   # assume January if no month given

    return None


def _to_month_index(year: int, month: int) -> int:
    """Converts (year, month) to a monotonic integer for easy comparison."""
    return year * 12 + month


def extract_date_ranges(text: str) -> list:
    """
    Extracts all date ranges from text.
    Pattern: <start_date> (–/-/to) <end_date>
    Returns list of (start_idx, end_idx) tuples in month-index units.
    """
    # Normalise separators so '–', '—', 'to', '-' all become '|'
    normalised = re.sub(r'\s*(–|—|to|-)\s*', ' | ', text, flags=re.IGNORECASE)

    # Find candidate date-range tokens: "Jan 2020 | Mar 2022"
    pattern = re.compile(
        r'(?:(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|'
        r'jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)'
        r'\s+\d{4}|\d{1,2}[/-]\d{4}|\d{4}|present|current|now|till\s+date|today)'
        r'\s*\|\s*'
        r'(?:(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|'
        r'jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)'
        r'\s+\d{4}|\d{1,2}[/-]\d{4}|\d{4}|present|current|now|till\s+date|today)',
        re.IGNORECASE
    )

    ranges = []
    for match in pattern.finditer(normalised):
        parts = match.group(0).split('|')
        if len(parts) == 2:
            start = _parse_month_year(parts[0].strip())
            end   = _parse_month_year(parts[1].strip())
            if start and end:
                s_idx = _to_month_index(*start)
                e_idx = _to_month_index(*end)
                if s_idx <= e_idx:
                    ranges.append((s_idx, e_idx, match.group(0).strip()))
    return ranges


def detect_overlapping_dates(resume_text: str) -> list:
    """
    Detects overlapping experience date ranges in the resume.

    Algorithm:
      1. Extract all date ranges from the text.
      2. For every pair of ranges, check if they overlap.
         Overlap condition: max(start1, start2) <= min(end1, end2)
      3. If overlap > 2 months, flag it.

    Returns a list of fraud flag strings.
    """
    flags = []
    ranges = extract_date_ranges(resume_text)

    if len(ranges) < 2:
        return flags

    for (s1, e1, label1), (s2, e2, label2) in combinations(ranges, 2):
        overlap_start = max(s1, s2)
        overlap_end   = min(e1, e2)
        overlap_months = overlap_end - overlap_start

        # Only flag significant overlaps (> 2 months) to avoid false positives
        if overlap_months > 2:
            flags.append(
                f"Overlapping Dates: '{label1}' overlaps '{label2}' "
                f"by ~{overlap_months} months"
            )
            # Report at most 2 overlaps to keep the flag list concise
            if len(flags) >= 2:
                break

    return flags


# ═══════════════════════════════════════════════════════════════
# SECTION 5 – Existing Anomaly Detection (enhanced)
# ═══════════════════════════════════════════════════════════════

def detect_anomalies(text: str, extracted_skills: list) -> list:
    """
    Detects keyword stuffing, short resumes, future years, skill density.
    Hidden text and overlapping dates are added separately in the main pipeline.
    """
    flags = []
    text_lower = text.lower()
    words = text_lower.split()

    # 1. Suspiciously short
    if len(words) < 15:
        flags.append("Suspiciously short content (< 15 words)")

    # 2. Keyword stuffing
    for skill in SKILLS_VOCABULARY:
        pattern = re.escape(skill) if ('+' in skill or '.' in skill) \
                  else r'\b' + re.escape(skill) + r'\b'
        matches = re.findall(pattern, text_lower)
        if len(matches) > 6:
            flags.append(
                f"Keyword Stuffing: '{skill.upper()}' repeated {len(matches)} times"
            )

    # 3. Future years
    future_years = re.findall(r'\b(202[7-9]|20[3-9]\d)\b', text)
    if future_years:
        flags.append(
            f"Inconsistent Dates: Future year '{future_years[0]}' detected"
        )

    # 4. Excessive skill density
    if words:
        term_count = sum(
            len(re.findall(
                re.escape(sk) if ('+' in sk or '.' in sk)
                else r'\b' + re.escape(sk) + r'\b',
                text_lower
            ))
            for sk in SKILLS_VOCABULARY
        )
        density = (term_count / len(words)) * 100
        if density > 35:
            flags.append(
                f"Excessive Skill Density: {round(density, 1)}% of text is skill keywords"
            )

    return flags


# ═══════════════════════════════════════════════════════════════
# SECTION 6 – Sub-score Helpers (Skill, Exp, Edu, Proj, Cert)
# ═══════════════════════════════════════════════════════════════

def _skill_match_score(resume_lower: str, job_lower: str):
    def extract_skills(text):
        found = []
        for skill in SKILLS_VOCABULARY:
            pattern = re.escape(skill) if ('+' in skill or '.' in skill) \
                      else r'\b' + re.escape(skill) + r'\b'
            if re.search(pattern, text):
                found.append(skill)
        return found

    resume_skills = extract_skills(resume_lower)
    job_skills    = extract_skills(job_lower)
    matched = list(set(resume_skills) & set(job_skills))
    missing = list(set(job_skills) - set(resume_skills))
    score = (len(matched) / len(job_skills) * 100.0) if job_skills else 100.0
    return score, matched, missing, resume_skills


def _experience_score(resume_lower: str, job_lower: str) -> float:
    score = 40.0
    exp_matches = re.findall(
        r'\b(\d{1,2})\s*(?:\+|-|\bto\b|\band\b)?\s*\d{0,2}\s*(?:years?|yrs?)\b',
        resume_lower
    )
    if exp_matches:
        years = max(int(y) for y in exp_matches if int(y) < 30)
        if   years >= 5: score = 100.0
        elif years >= 3: score =  90.0
        elif years >= 2: score =  80.0
        elif years >= 1: score =  70.0
        else:            score =  50.0
    else:
        exp_kws = ["experience", "worked", "internship", "job",
                   "position", "employment", "history", "role"]
        if any(kw in resume_lower for kw in exp_kws):
            score = 65.0
            jd_words  = set(re.findall(r'\b\w{4,}\b', job_lower))
            res_words = set(re.findall(r'\b\w{4,}\b', resume_lower))
            score += min(35.0, len(jd_words & res_words) * 2.5)
    return score


def _education_score(resume_lower: str) -> float:
    high = ["computer science", "computer engineering", "information technology",
            "btech", "b.tech", "b.e.", "bachelor of engineering", "mca", "software engineering"]
    med  = ["bachelor", "degree", "master", "mtech", "m.tech",
            "bsc", "b.sc", "graduate", "diploma", "university", "college"]
    if any(kw in resume_lower for kw in high): return 100.0
    if any(kw in resume_lower for kw in med):  return  75.0
    return 45.0


def _project_score(resume_lower: str) -> float:
    kws = ["project", "portfolio", "github.com", "git", "built",
           "developed", "designed", "implemented", "application", "system"]
    n = sum(1 for kw in kws if kw in resume_lower)
    if n >= 5 or "github.com" in resume_lower or "projects" in resume_lower: return 100.0
    if n >= 3: return 80.0
    if n >= 1: return 60.0
    return 30.0


def _cert_score(resume_lower: str) -> float:
    cert_kws = ["certification", "certified", "cert", "aws certified",
                "microsoft certified", "google certified", "udemy", "coursera",
                "certificate", "nptel"]
    cloud_kws = ["aws", "azure", "gcp", "oracle", "java", "python", "scrum", "agile"]
    if any(kw in resume_lower for kw in cert_kws):
        return 100.0 if any(kw in resume_lower for kw in cloud_kws) else 80.0
    if "course" in resume_lower or "training" in resume_lower:
        return 50.0
    return 30.0


# ═══════════════════════════════════════════════════════════════
# SECTION 7 – Main Public API
# ═══════════════════════════════════════════════════════════════

def calculate_match_score(resume_text: str, job_description: str,
                          pdf_path: str = None) -> dict:
    """
    Full resume-to-job scoring pipeline.

    Scoring Weights
    ---------------
    | Component                    | Weight |
    |------------------------------|--------|
    | TF-IDF Cosine Similarity     |  40%   |
    | AI/ML Ensemble Classifier    |  20%   |
    | Skill Keyword Match          |  25%   |
    | Experience Relevance         |  10%   |
    | Structural (Edu+Proj+Cert)   |   5%   |

    Fraud Detection
    ---------------
    | Check                        | Method                  |
    |------------------------------|-------------------------|
    | Keyword Stuffing             | Regex frequency count   |
    | Short Resume                 | Word count < 15         |
    | Future Dates                 | Year regex > 2026       |
    | Skill Density                | Term/word ratio > 35%   |
    | Hidden Text                  | pdfplumber char scan    |
    | Overlapping Dates            | Timeline interval check |
    | Duplicate Resume             | MD5 hash + email match  |

    Parameters
    ----------
    resume_text   : str  — raw extracted text from the PDF
    job_description: str — job posting text
    pdf_path      : str  — path to PDF file (for hidden text scan; optional)

    Returns dict with: score, recommendation, matched_skills, missing_skills,
                       resume_skills, fraud_flags, text_hash, cosine_score, ml_score
    """
    resume_lower = resume_text.lower()
    job_lower    = job_description.lower()

    # ── Guard ───────────────────────────────────────────────────
    if not resume_lower.strip() or not job_lower.strip():
        return {
            "score":          0.0,
            "recommendation": "Weak Fit",
            "resume_skills":  [],
            "matched_skills": [],
            "missing_skills": [],
            "fraud_flags":    ["Missing resume content or job description"],
            "text_hash":      get_text_hash(resume_text),
            "cosine_score":   0.0,
            "ml_score":       0.0,
        }

    # ── (1) TF-IDF Cosine Similarity  [40%] ────────────────────
    cosine_raw = compute_tfidf_cosine_similarity(resume_text, job_description)
    cosine_pct = cosine_raw * 100.0

    # ── (2) AI/ML Ensemble Classifier  [20%] ───────────────────
    ml_raw = compute_ml_ensemble_score(resume_text, job_description)
    ml_pct = ml_raw * 100.0

    # ── (3) Skill Keyword Match  [25%] ─────────────────────────
    skill_pct, matched, missing, resume_skills = _skill_match_score(resume_lower, job_lower)

    # ── (4) Experience Relevance  [10%] ────────────────────────
    exp_pct = _experience_score(resume_lower, job_lower)

    # ── (5) Structural signals  [5%] ───────────────────────────
    struct_pct = (_education_score(resume_lower) +
                  _project_score(resume_lower) +
                  _cert_score(resume_lower)) / 3.0

    # ── Weighted Composite ─────────────────────────────────────
    score = (
        0.40 * cosine_pct +
        0.20 * ml_pct     +
        0.25 * skill_pct  +
        0.10 * exp_pct    +
        0.05 * struct_pct
    )
    score = round(min(score, 100.0), 2)

    # ── Recommendation Band ────────────────────────────────────
    if   score >= 70: recommendation = "Strong Fit"
    elif score >= 40: recommendation = "Moderate Fit"
    else:             recommendation = "Weak Fit"

    # ── Fraud Detection ────────────────────────────────────────
    fraud_flags = detect_anomalies(resume_text, resume_skills)

    # Hidden Text (requires pdf_path)
    if pdf_path:
        fraud_flags.extend(detect_hidden_text(pdf_path))

    # Overlapping Dates (text-level, no pdf_path needed)
    fraud_flags.extend(detect_overlapping_dates(resume_text))

    return {
        "score":          score,
        "recommendation": recommendation,
        "resume_skills":  [_fmt_skill(s) for s in resume_skills],
        "matched_skills": [_fmt_skill(s) for s in matched],
        "missing_skills": [_fmt_skill(s) for s in missing],
        "fraud_flags":    fraud_flags,
        "text_hash":      get_text_hash(resume_text),
        "cosine_score":   round(cosine_raw, 4),
        "ml_score":       round(ml_raw, 4),
    }