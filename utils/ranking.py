from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import hashlib
from utils.parser import SKILLS_VOCABULARY

def get_text_hash(text):
    """
    Generates an MD5 hash of the parsed text to identify duplicates.
    """
    cleaned = re.sub(r'\s+', '', text).lower()
    return hashlib.md5(cleaned.encode('utf-8')).hexdigest()

def detect_anomalies(text, extracted_skills):
    """
    Performs keyword stuffing, text length, and formatting checks on resume text.
    """
    flags = []
    text_lower = text.lower()
    words = text_lower.split()
    
    # 1. Suspiciously Short Resume
    if len(words) < 15:
        flags.append("Suspiciously short content (< 15 words)")
        
    # 2. Keyword Stuffing Check
    for skill in SKILLS_VOCABULARY:
        if '+' in skill or '.' in skill:
            pattern = re.escape(skill)
        else:
            pattern = r'\b' + re.escape(skill) + r'\b'
            
        matches = re.findall(pattern, text_lower)
        # If a technical keyword is repeated more than 6 times, flag it
        if len(matches) > 6:
            flags.append(f"Keyword stuffing: '{skill.upper()}' repeated {len(matches)} times")
            
    # 3. Future Year Check
    # Looks for year boundaries like 2027 to 2099
    future_years = re.findall(r'\b(202[7-9]|20[3-9]\d)\b', text)
    if future_years:
        flags.append(f"Inconsistent dates: Future reference year '{future_years[0]}' detected")
        
    # 4. Dense Technical Terms (Resume Flags check)
    if len(words) > 0:
        term_count = 0
        for skill in SKILLS_VOCABULARY:
            if '+' in skill or '.' in skill:
                pattern = re.escape(skill)
            else:
                pattern = r'\b' + re.escape(skill) + r'\b'
            term_count += len(re.findall(pattern, text_lower))
            
        density = (term_count / len(words)) * 100
        if density > 35:
            flags.append(f"Excessive skill density: {round(density, 1)}% of text contains skills")
            
    return flags

def calculate_match_score(resume_text, job_description):
    """
    Main matching engine calculating similarity scores, skill gap analysis, and fraud checks.
    """
    resume_text_clean = resume_text.lower()
    job_desc_clean = job_description.lower()
    
    # Ensure documents have content before processing
    if not resume_text_clean.strip() or not job_desc_clean.strip():
        return {
            "score": 0.0,
            "recommendation": "Weak Fit",
            "resume_skills": [],
            "matched_skills": [],
            "missing_skills": [],
            "fraud_flags": ["Missing resume content or job description"],
            "text_hash": get_text_hash(resume_text)
        }
        
    # 1. TF-IDF and Cosine Similarity (Preserved core engine logic)
    documents = [resume_text_clean, job_desc_clean]
    tfidf = TfidfVectorizer(stop_words='english')
    
    try:
        tfidf_matrix = tfidf.fit_transform(documents)
        similarity = cosine_similarity(
            tfidf_matrix[0:1],
            tfidf_matrix[1:2]
        )
        base_score = similarity[0][0] * 100
    except Exception:
        base_score = 0.0
        
    # Smart boosting score is calculated below after skill gaps are determined
        
    # 2. Skill Gap Extraction
    resume_skills = []
    for skill in SKILLS_VOCABULARY:
        if '+' in skill or '.' in skill:
            pattern = re.escape(skill)
        else:
            pattern = r'\b' + re.escape(skill) + r'\b'
            
        if re.search(pattern, resume_text_clean):
            resume_skills.append(skill)
            
    job_skills = []
    for skill in SKILLS_VOCABULARY:
        if '+' in skill or '.' in skill:
            pattern = re.escape(skill)
        else:
            pattern = r'\b' + re.escape(skill) + r'\b'
            
        if re.search(pattern, job_desc_clean):
            job_skills.append(skill)
            
    # Calculate overlaps and differences
    matched_skills = list(set(resume_skills) & set(job_skills))
    missing_skills = list(set(job_skills) - set(resume_skills))
    
    # 3. Blended Score & Recommendation Calculation
    skills_ratio = len(matched_skills) / len(job_skills) if len(job_skills) > 0 else 0.0
    boosted_score = (skills_ratio * 90) + (base_score * 1.5) + 25
    if boosted_score > 100:
        boosted_score = 100.0
    if base_score == 0:
        boosted_score = 0.0
        
    score = round(boosted_score, 2)
    
    if score >= 70:
        recommendation = "Strong Fit"
    elif score >= 40:
        recommendation = "Moderate Fit"
    else:
        recommendation = "Weak Fit"
    # Format skills for visual representation in UI
    def clean_display_name(s):
        if s == 'c++': return 'C++'
        if s == 'c#': return 'C#'
        if s == 'node.js': return 'Node.js'
        if s == 'ci/cd': return 'CI/CD'
        if s == 'rest api': return 'REST API'
        return s.title()
        
    matched_formatted = [clean_display_name(x) for x in matched_skills]
    missing_formatted = [clean_display_name(x) for x in missing_skills]
    resume_formatted = [clean_display_name(x) for x in resume_skills]
    
    # 3. Run Fraud & Anomaly Audit
    fraud_flags = detect_anomalies(resume_text, resume_skills)
    text_hash = get_text_hash(resume_text)
    
    return {
        "score": score,
        "recommendation": recommendation,
        "resume_skills": resume_formatted,
        "matched_skills": matched_formatted,
        "missing_skills": missing_formatted,
        "fraud_flags": fraud_flags,
        "text_hash": text_hash
    }