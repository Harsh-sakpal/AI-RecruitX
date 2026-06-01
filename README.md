# AI RecruitX - AI Resume Screening & Ranking System

AI RecruitX is a simple AI-based Resume Screening and Ranking System developed for internship learning and academic project purposes. The main goal of this project is to help recruiters analyze resumes faster by automatically comparing resumes with job descriptions using basic NLP techniques like TF-IDF and Cosine Similarity.

The project is built using Python, Flask, SQLite, HTML, CSS, JavaScript, and Bootstrap. It can extract text from PDF resumes, calculate resume matching scores, identify missing skills, and rank candidates based on relevance to the job role.

The system also provides features like recruiter login, single and bulk resume upload, compare candidates, recruiter dashboard, and basic resume flags.

---

## Features

- Resume PDF parsing using pdfplumber
- Resume matching using TF-IDF and Cosine Similarity
- Candidate ranking system
- Skill gap analysis
- Basic resume flags and keyword stuffing detection
- Recruiter dashboard
- Single and bulk resume upload
- Compare candidates
- SQLite database integration
- Login-based recruiter access

---

## Technologies Used

### Backend

- Python
- Flask

### Database

- SQLite

### AI / NLP

- Scikit-Learn
- TF-IDF Vectorizer
- Cosine Similarity

### Frontend

- HTML5
- CSS3
- JavaScript
- Bootstrap 5

### PDF Processing

- pdfplumber

---

## Project Structure

```text
AI-Resume-Screening-System/

├── app.py
├── database.db
├── requirements.txt

├── uploads/
│   └── Uploaded resume PDF files

├── utils/
│   ├── parser.py
│   └── ranking.py

├── templates/
│   ├── login.html
│   ├── upload.html
│   └── dashboard.html

├── static/
│   └── style.css

└── README.md
```

---

## Installation and Setup

### 1. Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_LINK
```

### 2. Open the Project Folder

```bash
cd AI-Resume-Screening-System
```

### 3. Create Virtual Environment

```bash
python -m venv venv
```

### 4. Activate Virtual Environment

#### Windows

```bash
venv\Scripts\activate
```

#### Linux / Mac

```bash
source venv/bin/activate
```

### 5. Install Required Packages

```bash
pip install -r requirements.txt
```

### 6. Run the Flask Application

```bash
python app.py
```

The application will run on:

```text
http://127.0.0.1:5000
```

---

## How the System Works

### Step 1: Resume Upload

Recruiters upload resumes in PDF format.

### Step 2: Resume Parsing

The system extracts text from resumes using pdfplumber.

### Step 3: Text Processing

Resume text and job descriptions are cleaned and processed.

### Step 4: TF-IDF Vectorization

The text is converted into numerical vectors using TF-IDF.

### Step 5: Cosine Similarity

Cosine similarity compares resume vectors with job description vectors to generate a matching score.

### Step 6: Candidate Ranking

Candidates are ranked based on their matching scores.

---

## TF-IDF and Cosine Similarity

### TF-IDF

TF-IDF (Term Frequency - Inverse Document Frequency) helps identify important words in resumes and job descriptions.

### Cosine Similarity

Cosine Similarity is used to measure how closely a resume matches a job description.

Higher similarity means the resume is more relevant to the given job role.

---

## Basic Resume Flags Features

The system includes simple resume flag checks such as:

- Keyword stuffing detection
- Duplicate resume checking
- Invalid future year detection
- High keyword density warnings

---

## Deployment Support

The project structure is compatible with:

## Deployment

- AWS EC2 Deployment
- Render HTTPS Hosting
- Gunicorn Production Server

Future improvements may include AWS S3 storage and PostgreSQL integration.

---

## Project Developed By
### Team AI RecruitX

- Harsh Sakpal
- Harshada Pawar
- Sakshi Raul
