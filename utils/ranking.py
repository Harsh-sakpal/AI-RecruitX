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
    Main matching engine calculating similarity scores using the Weighted Candidate Scoring Algorithm:
    - Skill Match: 40%
    - Experience Relevance: 25%
    - Education Relevance: 15%
    - Project Relevance: 10%
    - Certifications: 10%
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
        
    # 1. Skill Match (40% Weight)
    resume_skills = []
    for skill in SKILLS_VOCABULARY:
        pattern = re.escape(skill) if ('+' in skill or '.' in skill) else r'\b' + re.escape(skill) + r'\b'
        if re.search(pattern, resume_text_clean):
            resume_skills.append(skill)
            
    job_skills = []
    for skill in SKILLS_VOCABULARY:
        pattern = re.escape(skill) if ('+' in skill or '.' in skill) else r'\b' + re.escape(skill) + r'\b'
        if re.search(pattern, job_desc_clean):
            job_skills.append(skill)
            
    matched_skills = list(set(resume_skills) & set(job_skills))
    missing_skills = list(set(job_skills) - set(resume_skills))
    
    skill_score = (len(matched_skills) / len(job_skills) * 100.0) if len(job_skills) > 0 else 100.0

    # 2. Experience Relevance (25% Weight)
    exp_score = 40.0  # Base score for having resume text
    exp_matches = re.findall(r'\b(\d{1,2})\s*(?:\+|-|\bto\b|\band\b)?\s*\d{0,2}\s*(?:years?|yrs?)\b', resume_text_clean)
    if exp_matches:
        years = max([int(y) for y in exp_matches if int(y) < 30])
        if years >= 5: exp_score = 100.0
        elif years >= 3: exp_score = 90.0
        elif years >= 2: exp_score = 80.0
        elif years >= 1: exp_score = 70.0
        else: exp_score = 50.0
    else:
        # Check for experience keywords and matching roles
        exp_kws = ["experience", "worked", "internship", "job", "position", "employment", "history", "role"]
        has_exp = any(kw in resume_text_clean for kw in exp_kws)
        if has_exp:
            exp_score = 65.0
            # Boost based on overlap with job description words
            jd_words = set(re.findall(r'\b\w{4,}\b', job_desc_clean))
            res_words = set(re.findall(r'\b\w{4,}\b', resume_text_clean))
            overlap = len(jd_words & res_words)
            exp_score += min(35.0, overlap * 2.5)
            
    # 3. Education Relevance (15% Weight)
    edu_score = 30.0
    high_align_kws = ["computer science", "computer engineering", "information technology", "btech", "b.tech", "b.e.", "bachelor of engineering", "mca", "software engineering"]
    med_align_kws = ["bachelor", "degree", "master", "mtech", "m.tech", "bsc", "b.sc", "graduate", "diploma", "university", "college"]
    
    if any(kw in resume_text_clean for kw in high_align_kws):
        edu_score = 100.0
    elif any(kw in resume_text_clean for kw in med_align_kws):
        edu_score = 75.0
    else:
        edu_score = 45.0
        
    # 4. Project Relevance (10% Weight)
    proj_score = 30.0
    project_kws = ["project", "portfolio", "github.com", "git", "built", "developed", "designed", "implemented", "application", "system"]
    proj_matches = sum(1 for kw in project_kws if kw in resume_text_clean)
    if proj_matches >= 5 or "github.com" in resume_text_clean or "projects" in resume_text_clean:
        proj_score = 100.0
    elif proj_matches >= 3:
        proj_score = 80.0
    elif proj_matches >= 1:
        proj_score = 60.0
        
    # 5. Certifications (10% Weight)
    cert_score = 30.0
    cert_kws = ["certification", "certified", "cert", "aws certified", "microsoft certified", "google certified", "oracle certified", "udemy", "coursera", "certificate", "nptel"]
    if any(kw in resume_text_clean for kw in cert_kws):
        if any(cloud_cert in resume_text_clean for cloud_cert in ["aws", "azure", "gcp", "oracle", "java", "python", "scrum", "agile"]):
            cert_score = 100.0
        else:
            cert_score = 80.0
    else:
        if "course" in resume_text_clean or "training" in resume_text_clean:
            cert_score = 50.0

    # Weighted calculation
    score = (0.40 * skill_score) + (0.25 * exp_score) + (0.15 * edu_score) + (0.10 * proj_score) + (0.10 * cert_score)
    score = round(score, 2)
    
    if score > 100.0:
        score = 100.0
        
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