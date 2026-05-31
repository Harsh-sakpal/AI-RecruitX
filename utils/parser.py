import pdfplumber
import re
import os

def extract_text_from_pdf(pdf_path):
    """
    Extracts raw text from a PDF resume.
    """
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
    
    # Clean up horizontal spaces per line but preserve newlines for parser split
    lines = []
    for line in text.split('\n'):
        cleaned_line = re.sub(r'[ \t\r\f\v]+', ' ', line).strip()
        if cleaned_line:
            lines.append(cleaned_line)
    return "\n".join(lines)

# Predefined vocabulary of standard technical skills for parsing
SKILLS_VOCABULARY = [
    'python', 'java', 'c++', 'c#', 'php', 'ruby', 'go', 'rust', 'swift', 'kotlin',
    'javascript', 'typescript', 'html', 'css', 'react', 'angular', 'vue', 'jquery',
    'node.js', 'node', 'express', 'flask', 'django', 'fastapi', 'spring boot',
    'sql', 'mysql', 'postgresql', 'mongodb', 'sqlite', 'redis', 'cassandra', 'oracle',
    'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'jenkins', 'git', 'github', 'gitlab',
    'machine learning', 'deep learning', 'data science', 'pandas', 'numpy', 'scikit-learn',
    'tensorflow', 'pytorch', 'nlp', 'computer vision', 'tableau', 'powerbi', 'excel',
    'agile', 'scrum', 'linux', 'unix', 'devops', 'ci/cd', 'rest api', 'graphql'
]

def parse_resume_data(text, filename):
    """
    Extracts name, email, phone, skills, experience, and education from resume text.
    """
    text_lower = text.lower()
    
    # 1. Extract Email
    email_match = re.search(r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+', text)
    email = email_match.group(0) if email_match else "N/A"
    
    # 2. Extract Phone
    # Common formats: +91 9876543210, (123) 456-7890, 123-456-7890, etc.
    phone_pattern = r'(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'
    phone_match = re.search(phone_pattern, text)
    phone = phone_match.group(0) if phone_match else "N/A"
    
    # 3. Extract Name
    # Priority 1: Extract from the first 3 non-empty lines of the text (most accurate for PDFs)
    name = ""
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    for line in lines[:3]:
        clean_line = re.sub(r'[^a-zA-Z\s]', '', line).strip()
        words_in_line = clean_line.split()
        # A name should contain only alphabetical characters, have 2 to 3 words, and not look like section headers
        if 2 <= len(words_in_line) <= 3 and not any(h in clean_line.lower() for h in ['resume', 'cv', 'experience', 'education', 'skills', 'summary', 'contact', 'developer', 'engineer']):
            name = " ".join([w.capitalize() for w in words_in_line])
            break

    # Priority 2: Fall back to clean file name if text parsing did not yield a valid name
    if not name:
        base_name = os.path.basename(filename)
        name_part = os.path.splitext(base_name)[0]
        # Remove common words like "resume", "cv", "pdf", and special characters
        clean_name = re.sub(r'(?i)(resume|cv|pdf|_|\-|\bfor\b|\bof\b)', ' ', name_part).strip()
        clean_name = re.sub(r'\s+', ' ', clean_name)
        
        # Don't accept purely numeric names or filenames that are too generic
        if clean_name and not clean_name.lower() in ["resume", "cv", "my resume", "mycv", "job"] and not clean_name.isdigit() and len(clean_name) > 3:
            name = " ".join([word.capitalize() for word in clean_name.split()])
        else:
            name = "Candidate"

    # 4. Extract Skills (case-insensitive keyword matching)
    extracted_skills = []
    for skill in SKILLS_VOCABULARY:
        # Match word boundaries unless special characters like ++ or . are present
        if '+' in skill or '.' in skill:
            pattern = re.escape(skill)
        else:
            pattern = r'\b' + re.escape(skill) + r'\b'
            
        if re.search(pattern, text_lower):
            display_name = skill
            # Format display names nicely
            if display_name == 'c++':
                display_name = 'C++'
            elif display_name == 'c#':
                display_name = 'C#'
            elif display_name == 'node.js':
                display_name = 'Node.js'
            elif display_name == 'ci/cd':
                display_name = 'CI/CD'
            elif display_name == 'rest api':
                display_name = 'REST API'
            else:
                display_name = display_name.title()
                
            if display_name not in extracted_skills:
                extracted_skills.append(display_name)
                
    # 5. Extract Education and Experience Sections (Quick heuristic-based extraction)
    sections = {"experience": "N/A", "education": "N/A"}
    
    exp_headers = [r'work experience', r'professional experience', r'experience', r'employment history', r'history']
    edu_headers = [r'education', r'academic background', r'qualifications', r'academic history']
    
    # Locate sections and extract text following the header
    for header in exp_headers:
        match = re.search(r'(?i)\b' + header + r'\b', text)
        if match:
            start_idx = match.end()
            # Extract next 350 chars
            snippet = text[start_idx:start_idx+350].strip()
            sections["experience"] = snippet
            break
            
    for header in edu_headers:
        match = re.search(r'(?i)\b' + header + r'\b', text)
        if match:
            start_idx = match.end()
            snippet = text[start_idx:start_idx+350].strip()
            sections["education"] = snippet
            break

    # Truncate strings to prevent UI layout breaks
    exp_text = sections["experience"]
    if exp_text != "N/A":
        exp_text = exp_text[:280] + ("..." if len(exp_text) > 280 else "")
        
    edu_text = sections["education"]
    if edu_text != "N/A":
        edu_text = edu_text[:280] + ("..." if len(edu_text) > 280 else "")

    return {
        "name": name,
        "email": email,
        "phone": phone,
        "skills": extracted_skills,
        "experience": exp_text,
        "education": edu_text
    }