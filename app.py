import os
import re
import json
import sqlite3
from functools import wraps
from flask import (
    Flask, render_template, request, jsonify,
    redirect, url_for, make_response
)
from werkzeug.utils import secure_filename

# Core parsing and ranking engines
from utils.parser import extract_text_from_pdf, parse_resume_data, SKILLS_VOCABULARY
from utils.ranking import calculate_match_score
from utils.jwt_auth import generate_jwt, verify_jwt

app = Flask(__name__)

# ---------------------------------------------------------------
# Configuration
# SECRET_KEY: loaded from environment variable for AWS security.
# Falls back to a hardcoded dev key so local testing still works.
# On AWS — set this as an environment variable:
#   export SECRET_KEY="some-long-random-string"
# ---------------------------------------------------------------
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "ai_recruitx_secret_2026_dev")

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
DATABASE      = os.path.join(BASE_DIR, "database.db")

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ---------------------------------------------------------------
# Shared skill-name formatter (used in multiple routes — defined
# once here to avoid copy-paste repetition across the file).
# ---------------------------------------------------------------
def format_skill_name(skill: str) -> str:
    """Returns a nicely formatted display name for a raw skill string."""
    special = {"c++": "C++", "c#": "C#", "node.js": "Node.js",
                "ci/cd": "CI/CD", "rest api": "REST API"}
    return special.get(skill, skill.title())


# ---------------------------------------------------------------
# SQLite Database Helpers
# ---------------------------------------------------------------
def get_db_connection():
    """Opens a SQLite connection with Row factory so columns are accessible by name."""
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Creates tables on first run and seeds two sample job profiles."""
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS jobs (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        title       TEXT NOT NULL,
        description TEXT NOT NULL,
        created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )""")

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS candidates (
        id             INTEGER PRIMARY KEY AUTOINCREMENT,
        job_id         INTEGER NOT NULL,
        filename       TEXT NOT NULL,
        name           TEXT,
        email          TEXT,
        phone          TEXT,
        skills         TEXT,
        experience     TEXT,
        education      TEXT,
        score          REAL,
        recommendation TEXT,
        fraud_flags    TEXT,
        text_hash      TEXT,
        upload_date    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(job_id) REFERENCES jobs(id)
    )""")

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS uploads (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        filename    TEXT NOT NULL,
        upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )""")

    # Seed sample jobs only once
    if cursor.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 0:
        cursor.execute(
            "INSERT INTO jobs (title, description) VALUES (?, ?)",
            ("Python Developer Intern",
             "We are looking for a Python Developer Intern. Required skills: "
             "Python, Flask, SQL, Git, REST API, Linux, and Docker.")
        )
        cursor.execute(
            "INSERT INTO jobs (title, description) VALUES (?, ?)",
            ("Frontend Web Developer Intern",
             "Seeking an intern with expertise in HTML, CSS, JavaScript, "
             "React, Bootstrap, Git, and Web Design.")
        )

    conn.commit()
    conn.close()


init_db()


# ---------------------------------------------------------------
# JWT-based Auth Middleware
# Reads the 'auth_token' cookie, verifies the JWT, and blocks
# the request if the token is missing, expired, or tampered with.
# ---------------------------------------------------------------
def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.cookies.get("auth_token")
        if not token:
            return redirect(url_for("login"))

        payload = verify_jwt(token, app.config["SECRET_KEY"])
        if payload is None:
            # Token invalid or expired — send back to login
            response = make_response(redirect(url_for("login")))
            response.delete_cookie("auth_token")
            return response

        return f(*args, **kwargs)
    return decorated


# ---------------------------------------------------------------
# Shared helper: compute missing skills for a candidate vs a job
# ---------------------------------------------------------------
def compute_missing_skills(matched_skills: list, job_description: str) -> list:
    """
    Returns a list of skills required by the job but absent in the candidate's resume.
    """
    job_desc_lower = job_description.lower()
    job_skills = []
    for skill in SKILLS_VOCABULARY:
        pattern = r'\b' + re.escape(skill) + r'\b'
        if re.search(pattern, job_desc_lower):
            job_skills.append(format_skill_name(skill))
    return list(set(job_skills) - set(matched_skills))


# ---------------------------------------------------------------
# Authentication Routes
# ---------------------------------------------------------------
@app.route("/login", methods=["GET", "POST"])
def login():
    # Already logged in — redirect straight to dashboard
    if request.cookies.get("auth_token"):
        payload = verify_jwt(request.cookies["auth_token"], app.config["SECRET_KEY"])
        if payload:
            return redirect(url_for("dashboard"))

    error = None
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()

        if username == "admin" and password == "admin":
            # Generate a JWT token valid for 2 hours
            token = generate_jwt(username, app.config["SECRET_KEY"], expiry_hours=2)

            response = make_response(redirect(url_for("dashboard")))
            # httponly=True prevents JavaScript from reading the cookie (XSS protection)
            # samesite="Lax" prevents CSRF attacks
            response.set_cookie(
                "auth_token", token,
                httponly=True,
                samesite="Lax",
                max_age=7200  # 2 hours in seconds
            )
            return response
        else:
            error = "Incorrect username or password. Please try again."

    return render_template("login.html", error=error)


@app.route("/logout")
def logout():
    response = make_response(redirect(url_for("login")))
    response.delete_cookie("auth_token")
    return response


@app.route("/")
def home():
    if request.cookies.get("auth_token"):
        payload = verify_jwt(request.cookies["auth_token"], app.config["SECRET_KEY"])
        if payload:
            return redirect(url_for("dashboard"))
    return redirect(url_for("login"))


# ---------------------------------------------------------------
# Page Routes (render HTML templates)
# ---------------------------------------------------------------
@app.route("/dashboard")
@login_required
def dashboard():
    selected_job_id = request.args.get("job_id", type=int)
    conn = get_db_connection()
    jobs = conn.execute("SELECT * FROM jobs ORDER BY id DESC").fetchall()
    if not selected_job_id and jobs:
        selected_job_id = jobs[0]["id"]
    conn.close()
    return render_template("dashboard.html", jobs=jobs, selected_job_id=selected_job_id)


@app.route("/upload")
@login_required
def upload():
    conn = get_db_connection()
    jobs = conn.execute("SELECT * FROM jobs ORDER BY id DESC").fetchall()
    conn.close()
    return render_template("upload.html", jobs=jobs)


# ---------------------------------------------------------------
# API: Create Job Profile
# ---------------------------------------------------------------
@app.route("/create-job", methods=["POST"])
@login_required
def create_job():
    data = request.get_json()
    if not data or not data.get("title") or not data.get("description"):
        return jsonify({"success": False, "error": "Missing job title or description"}), 400

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO jobs (title, description) VALUES (?, ?)",
        (data["title"].strip(), data["description"].strip())
    )
    job_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return jsonify({"success": True, "job_id": job_id})


# ---------------------------------------------------------------
# Shared resume processing logic (avoids duplication between
# single-upload and bulk-upload routes)
# ---------------------------------------------------------------
def _process_single_resume(file_path, filename, job_id, job_description, conn):
    """
    Parses and scores one resume. Returns a dict with results.
    Raises an exception on failure — caller handles it.
    """
    cursor = conn.cursor()

    # Log the upload action
    cursor.execute("INSERT INTO uploads (filename) VALUES (?)", (filename,))
    conn.commit()

    # 1. Extract text from PDF
    resume_text = extract_text_from_pdf(file_path)

    # 2. Parse candidate details
    parsed = parse_resume_data(resume_text, filename)

    # 3. Calculate match score + skill gaps + resume flags
    results = calculate_match_score(resume_text, job_description)

    # 4. Duplicate check (by email OR content hash)
    if parsed["email"] != "N/A":
        dup = conn.execute(
            "SELECT COUNT(*) FROM candidates WHERE job_id=? AND (text_hash=? OR email=?)",
            (job_id, results["text_hash"], parsed["email"])
        ).fetchone()[0]
    else:
        dup = conn.execute(
            "SELECT COUNT(*) FROM candidates WHERE job_id=? AND text_hash=?",
            (job_id, results["text_hash"])
        ).fetchone()[0]

    fraud_flags = results["fraud_flags"]
    if dup > 0:
        fraud_flags.append("Duplicate Resume: Candidate already exists for this job")

    # 5. Save candidate to database
    cursor.execute("""
        INSERT INTO candidates
            (job_id, filename, name, email, phone, skills, experience,
             education, score, recommendation, fraud_flags, text_hash)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        job_id, filename, parsed["name"], parsed["email"], parsed["phone"],
        json.dumps(results["matched_skills"]), parsed["experience"], parsed["education"],
        results["score"], results["recommendation"],
        json.dumps(fraud_flags), results["text_hash"]
    ))
    candidate_id = cursor.lastrowid
    conn.commit()

    return {
        "candidate_id": candidate_id,
        "name":         parsed["name"],
        "score":        results["score"],
        "recommendation": results["recommendation"],
        "fraud_flags":  fraud_flags
    }


# ---------------------------------------------------------------
# API: Upload Single Resume
# ---------------------------------------------------------------
@app.route("/upload-resume", methods=["POST"])
@login_required
def upload_resume():
    if "resume" not in request.files:
        return jsonify({"success": False, "error": "No file uploaded"}), 400

    file   = request.files["resume"]
    job_id = request.form.get("job_id")

    if not file.filename:
        return jsonify({"success": False, "error": "Empty file name"}), 400
    if not job_id:
        return jsonify({"success": False, "error": "Job ID is required"}), 400

    conn = get_db_connection()
    job  = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    if not job:
        conn.close()
        return jsonify({"success": False, "error": "Job profile not found"}), 404

    filename  = secure_filename(file.filename)
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(file_path)

    try:
        result = _process_single_resume(file_path, filename, job_id, job["description"], conn)
    except Exception as e:
        conn.close()
        return jsonify({"success": False, "error": f"Processing failed: {str(e)}"}), 500

    conn.close()

    response_data = {"success": True, **result}

    # Return JSON for AJAX callers, redirect for plain form submits
    is_ajax = (
        "application/json" in request.headers.get("Accept", "")
        or request.headers.get("X-Requested-With") == "XMLHttpRequest"
    )
    if is_ajax:
        return jsonify(response_data)
    return redirect(url_for("dashboard", job_id=job_id))


# ---------------------------------------------------------------
# API: Bulk Upload (sequential, one file at a time)
# ---------------------------------------------------------------
@app.route("/bulk-upload", methods=["POST"])
@login_required
def bulk_upload():
    if "resumes" not in request.files:
        return jsonify({"success": False, "error": "No files uploaded"}), 400

    files  = request.files.getlist("resumes")
    job_id = request.form.get("job_id")

    if not job_id:
        return jsonify({"success": False, "error": "Job ID is required"}), 400

    conn = get_db_connection()
    job  = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    if not job:
        conn.close()
        return jsonify({"success": False, "error": "Job profile not found"}), 404
    conn.close()

    results = []
    for file in files:
        if not file.filename:
            continue

        filename  = secure_filename(file.filename)
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(file_path)

        conn = get_db_connection()
        try:
            result = _process_single_resume(file_path, filename, job_id, job["description"], conn)
            results.append({
                "filename":       filename,
                "score":          result["score"],
                "recommendation": result["recommendation"],
                "status":         "Success"
            })
        except Exception as e:
            results.append({"filename": filename, "status": "Error", "error": str(e)})
        finally:
            conn.close()

    return jsonify({"success": True, "processed_count": len(results), "files": results})


# ---------------------------------------------------------------
# API: Get Ranked Candidates for a Job
# ---------------------------------------------------------------
@app.route("/ranked-candidates")
@login_required
def ranked_candidates():
    job_id = request.args.get("job_id", type=int)
    if not job_id:
        return jsonify({"error": "Missing job_id parameter"}), 400

    conn = get_db_connection()
    job  = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    if not job:
        conn.close()
        return jsonify({"error": "Job profile not found"}), 404

    candidates = conn.execute(
        "SELECT * FROM candidates WHERE job_id = ? ORDER BY score DESC",
        (job_id,)
    ).fetchall()
    conn.close()

    result = []
    for cand in candidates:
        cand_dict      = dict(cand)
        matched_skills = json.loads(cand_dict["skills"] or "[]")
        missing_skills = compute_missing_skills(matched_skills, job["description"])
        cand_dict["missing_skills"] = json.dumps(missing_skills)
        result.append(cand_dict)

    return jsonify(result)


# ---------------------------------------------------------------
# API: Get Single Candidate Details
# ---------------------------------------------------------------
@app.route("/candidate/<int:candidate_id>")
@login_required
def get_candidate(candidate_id):
    conn      = get_db_connection()
    candidate = conn.execute("SELECT * FROM candidates WHERE id = ?", (candidate_id,)).fetchone()
    if not candidate:
        conn.close()
        return jsonify({"error": "Candidate not found"}), 404

    job = conn.execute("SELECT * FROM jobs WHERE id = ?", (candidate["job_id"],)).fetchone()
    conn.close()

    cand_dict      = dict(candidate)
    matched_skills = json.loads(cand_dict["skills"] or "[]")
    missing_skills = compute_missing_skills(matched_skills, job["description"])

    cand_dict["matched_skills"] = json.dumps(matched_skills)
    cand_dict["missing_skills"] = json.dumps(missing_skills)
    return jsonify(cand_dict)


# ---------------------------------------------------------------
# API: Analytics / KPI Stats
# ---------------------------------------------------------------
@app.route("/analytics")
@login_required
def analytics():
    job_id = request.args.get("job_id", type=int)
    conn   = get_db_connection()

    if job_id:
        total     = conn.execute("SELECT COUNT(*) FROM candidates WHERE job_id=?", (job_id,)).fetchone()[0]
        strong    = conn.execute("SELECT COUNT(*) FROM candidates WHERE job_id=? AND recommendation='Strong Fit'", (job_id,)).fetchone()[0]
        avg_row   = conn.execute("SELECT AVG(score) FROM candidates WHERE job_id=?", (job_id,)).fetchone()
        flag_rows = conn.execute("SELECT fraud_flags FROM candidates WHERE job_id=?", (job_id,)).fetchall()
    else:
        total     = conn.execute("SELECT COUNT(*) FROM candidates").fetchone()[0]
        strong    = conn.execute("SELECT COUNT(*) FROM candidates WHERE recommendation='Strong Fit'").fetchone()[0]
        avg_row   = conn.execute("SELECT AVG(score) FROM candidates").fetchone()
        flag_rows = conn.execute("SELECT fraud_flags FROM candidates").fetchall()

    conn.close()

    avg_score   = round(avg_row[0], 2) if avg_row[0] else 0.0
    fraud_count = sum(1 for row in flag_rows if json.loads(row["fraud_flags"] or "[]"))

    return jsonify({
        "total_resumes":  total,
        "strong_fits":    strong,
        "average_score":  avg_score,
        "flagged_fraud":  fraud_count
    })


# ---------------------------------------------------------------
# API: Delete a Single Candidate
# ---------------------------------------------------------------
@app.route("/delete-candidate/<int:candidate_id>", methods=["POST"])
@login_required
def delete_candidate(candidate_id):
    conn      = get_db_connection()
    cursor    = conn.cursor()
    candidate = conn.execute("SELECT filename FROM candidates WHERE id=?", (candidate_id,)).fetchone()

    if not candidate:
        conn.close()
        return jsonify({"success": False, "error": "Candidate not found"}), 404

    # Remove the uploaded PDF from disk
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], candidate["filename"])
    if os.path.exists(file_path):
        try:
            os.remove(file_path)
        except OSError:
            pass  # File might already be deleted — ignore

    cursor.execute("DELETE FROM candidates WHERE id=?", (candidate_id,))
    conn.commit()
    conn.close()
    return jsonify({"success": True})


# ---------------------------------------------------------------
# API: Clear All Candidates for a Job
# ---------------------------------------------------------------
@app.route("/clear-job-data/<int:job_id>", methods=["POST"])
@login_required
def clear_job_data(job_id):
    conn       = get_db_connection()
    cursor     = conn.cursor()
    candidates = conn.execute("SELECT filename FROM candidates WHERE job_id=?", (job_id,)).fetchall()

    for cand in candidates:
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], cand["filename"])
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except OSError:
                pass

    cursor.execute("DELETE FROM candidates WHERE job_id=?", (job_id,))
    conn.commit()
    conn.close()
    return jsonify({"success": True})


# ---------------------------------------------------------------
# Run the app
# debug=False is important for AWS production — never expose the
# debug reloader or interactive debugger to the internet.
# ---------------------------------------------------------------
if __name__ == "__main__":
    debug_mode = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    app.run(debug=debug_mode, host="0.0.0.0", port=5000)