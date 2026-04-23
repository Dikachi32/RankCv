from flask import Flask, render_template, request, redirect, flash, jsonify, session, url_for
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime, date, timezone
from functools import wraps
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
import hashlib
import os
import json
import re
import io
import logging

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# =============================================================================
# PDF EXTRACTION: pdfplumber primary, PyPDF2 fallback
# =============================================================================
try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False
    import PyPDF2

# =============================================================================
# SEMANTIC EMBEDDINGS (OPTIONAL BUT HIGHLY RECOMMENDED)
# =============================================================================
try:
    # Suppress Hugging Face "unauthenticated" warning
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

    from sentence_transformers import SentenceTransformer, util
    SENTENCE_TRANSFORMERS_AVAILABLE = True
    print("Loading sentence transformer model (one-time ~80MB download)...")
    EMBEDDING_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
    print("Sentence transformer loaded successfully!")
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    EMBEDDING_MODEL = None
    print("WARNING: sentence-transformers not installed. Using improved TF-IDF fallback.")
    print("For much better results, run: pip install sentence-transformers")

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-this-in-production'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:root@localhost/rankcv_db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_IMAGE_EXTENSIONS

# =============================================================================
# TIMEZONE-AWARE UTC HELPER (fixes Python 3.12 deprecation)
# =============================================================================
def utc_now():
    return datetime.now(timezone.utc).replace(tzinfo=None)

@app.context_processor
def inject_now():
    return {'now': utc_now()}

# =============================================================================
# CUSTOM STOPWORDS — only fluff. NEVER tech terms like python, java, scala, etc.
# =============================================================================
CUSTOM_STOPWORDS = {
    'a', 'an', 'and', 'are', 'as', 'at', 'be', 'been', 'being', 'by', 'for', 'from',
    'has', 'have', 'had', 'he', 'her', 'him', 'his', 'how', 'i', 'in', 'is', 'it',
    'its', 'me', 'my', 'of', 'on', 'or', 'our', 'she', 'so', 'than', 'that', 'the',
    'their', 'them', 'then', 'there', 'these', 'they', 'this', 'those', 'to', 'too',
    'was', 'we', 'were', 'what', 'when', 'where', 'which', 'who', 'whom', 'whose',
    'why', 'will', 'with', 'would', 'you', 'your', 'yours', 'yourself', 'yourselves',
    'am', 'shall', 'may', 'might', 'must', 'can', 'could', 'need', 'dare', 'ought', 'used',
    'about', 'above', 'across', 'after', 'against', 'along', 'among', 'around', 'before',
    'behind', 'below', 'beneath', 'beside', 'between', 'beyond', 'but', 'despite', 'down',
    'during', 'except', 'inside', 'into', 'like', 'near', 'off', 'onto', 'opposite', 'out',
    'outside', 'over', 'past', 'regarding', 'round', 'since', 'through', 'throughout',
    'till', 'toward', 'towards', 'under', 'underneath', 'until', 'up', 'upon', 'via',
    'within', 'without', 'here', 'now', 'once', 'again', 'further', 'also', 'only', 'just',
    'even', 'both', 'all', 'any', 'some', 'many', 'much', 'few', 'little', 'more', 'most',
    'other', 'another', 'such', 'no', 'nor', 'not', 'same', 'own', 'very', 'quite', 'rather',
    'still', 'yet', 'already', 'almost', 'enough', 'indeed', 'instead', 'else', 'otherwise',
    'perhaps', 'probably', 'possibly', 'certainly', 'surely', 'actually', 'really', 'truly',
    'clearly', 'obviously', 'apparently', 'generally', 'usually', 'normally', 'typically',
    'especially', 'particularly', 'specifically', 'mainly', 'mostly', 'largely', 'partly',
    'fully', 'highly', 'greatly', 'deeply', 'strongly', 'widely', 'closely', 'directly',
    'indirectly', 'easily', 'hardly', 'nearly', 'badly', 'well', 'better', 'best', 'worse',
    'worst', 'less', 'least', 'ago', 'ahead', 'away', 'back', 'forward', 'backward',
    'downward', 'upward', 'inward', 'outward', 'somewhere', 'anywhere', 'everywhere',
    'nowhere', 'elsewhere', 'sometime', 'anytime', 'always', 'never', 'ever', 'forever',
    'temporarily', 'permanently', 'constantly', 'continuously', 'repeatedly', 'frequently',
    'often', 'sometimes', 'occasionally', 'rarely', 'seldom', 'daily', 'weekly', 'monthly',
    'yearly', 'hourly', 'nightly', 'today', 'tomorrow', 'yesterday', 'tonight', 'everyday'
}

# =============================================================================
# TEXT PREPROCESSING
# =============================================================================
def preprocess_text(text):
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r'[\/\|\-\\–\—\•\·\⋅\>\<\[\]\{\}\(\)\:\;\,\"\']', ' ', text)
    text = re.sub(r'http\S+|www\.\S+', ' ', text)
    text = re.sub(r'\S+@\S+\.\S+', ' ', text)
    text = re.sub(r'[^\w\s\.\+\#]', ' ', text)
    text = re.sub(r'\.{2,}', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def split_into_sentences(text):
    if not text:
        return []
    raw = re.split(r'(?<=[.!?])\s+', text)
    sentences = []
    for s in raw:
        s = s.strip()
        if len(s) >= 15 and len(s.split()) >= 3:
            sentences.append(s)
    return sentences if sentences else [text]

def extract_keywords(text):
    words = preprocess_text(text).split()
    keywords = [w for w in words if w not in CUSTOM_STOPWORDS and len(w) >= 3]
    return set(keywords)

def keyword_overlap_score(job_desc, cv_text):
    jd_kw = extract_keywords(job_desc)
    cv_kw = extract_keywords(cv_text)
    if not jd_kw:
        return 0.0
    matches = len(jd_kw & cv_kw)
    return (matches / len(jd_kw)) * 100

# =============================================================================
# SEMANTIC SCORING (sentence-transformers)
# =============================================================================
def semantic_scores(job_desc, cv_texts):
    if not SENTENCE_TRANSFORMERS_AVAILABLE or EMBEDDING_MODEL is None:
        return None

    job_embedding = EMBEDDING_MODEL.encode(job_desc, convert_to_tensor=True)
    scores = []

    for cv_text in cv_texts:
        sentences = split_into_sentences(cv_text)
        if not sentences:
            scores.append(0.0)
            continue

        chunks = []
        current_chunk = []
        current_len = 0

        for sent in sentences:
            sent_len = len(sent.split())
            if current_len + sent_len > 150 and current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = [sent]
                current_len = sent_len
            else:
                current_chunk.append(sent)
                current_len += sent_len

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        chunks = chunks[:15]

        if not chunks:
            scores.append(0.0)
            continue

        chunk_embeddings = EMBEDDING_MODEL.encode(chunks, convert_to_tensor=True)
        similarities = util.cos_sim(job_embedding, chunk_embeddings)
        max_sim = float(similarities.max())

        kw_score = keyword_overlap_score(job_desc, cv_text)
        blended = (max_sim * 80) + (min(kw_score, 100) * 0.2)
        scores.append(round(min(blended, 99.9), 1))

    return scores

# =============================================================================
# FALLBACK SCORING: Best-sentence TF-IDF + keyword boost
# =============================================================================
def fallback_scores(job_desc, cv_texts):
    jd_sentences = split_into_sentences(job_desc)
    query_doc = " ".join(jd_sentences) if jd_sentences else job_desc

    all_sentences = []
    cv_ranges = []

    for cv_text in cv_texts:
        sentences = split_into_sentences(cv_text)
        if not sentences:
            sentences = [cv_text if cv_text else "empty"]
        start = len(all_sentences)
        all_sentences.extend(sentences)
        cv_ranges.append((start, len(all_sentences)))

    docs = [query_doc] + all_sentences

    vectorizer = TfidfVectorizer(
        stop_words=list(CUSTOM_STOPWORDS),
        ngram_range=(1, 2),
        max_features=5000,
        min_df=1,
        max_df=0.9,
        sublinear_tf=True
    )

    try:
        tfidf_matrix = vectorizer.fit_transform(docs)
    except ValueError:
        return [0.0] * len(cv_texts)

    query_vec = tfidf_matrix[0]
    sentence_vecs = tfidf_matrix[1:]

    scores = []
    for i, (start, end) in enumerate(cv_ranges):
        if start >= end:
            scores.append(0.0)
            continue

        sims = cosine_similarity(query_vec, sentence_vecs[start:end]).flatten()
        top_sims = sorted(sims, reverse=True)[:3]
        avg_top = sum(top_sims) / len(top_sims) if top_sims else 0
        tfidf_score = float(avg_top) * 100

        kw_score = keyword_overlap_score(job_desc, cv_texts[i])
        blended = (tfidf_score * 0.7) + (kw_score * 0.3)
        scores.append(round(min(blended, 99.9), 1))

    return scores

# =============================================================================
# MASTER SCORING FUNCTION
# =============================================================================
def compute_similarity_scores(job_desc, cv_texts):
    sem_scores = semantic_scores(job_desc, cv_texts)
    if sem_scores is not None:
        return sem_scores
    return fallback_scores(job_desc, cv_texts)

# =============================================================================
# PDF EXTRACTION
# =============================================================================
def extract_text_from_pdf(file):
    try:
        file.seek(0)
        file_content = file.read()
        if not file_content:
            return None, "File is empty"

        pdf_file = io.BytesIO(file_content)
        text = ""

        if HAS_PDFPLUMBER:
            with pdfplumber.open(pdf_file) as pdf:
                if len(pdf.pages) == 0:
                    return None, "PDF has no pages"
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
        else:
            reader = PyPDF2.PdfReader(pdf_file)
            if reader.is_encrypted:
                try:
                    reader.decrypt('')
                except:
                    return None, "PDF is password protected"
            for page in reader.pages:
                try:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                except:
                    continue

        clean_text = text.strip()
        if not clean_text:
            return None, "PDF appears to be a scanned image (no text found)"

        return clean_text, None

    except Exception as e:
        return None, f"Could not read PDF: {str(e)}"


# =============================================================================
# DATABASE MODELS
# =============================================================================
class User(db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(255))
    profile_pic = db.Column(db.String(255), nullable=True)
    subscription_plan = db.Column(db.String(20), default='free')
    subscription_expires = db.Column(db.DateTime, nullable=True)
    stripe_customer_id = db.Column(db.String(255), nullable=True)
    created_at = db.Column(db.TIMESTAMP, default=utc_now)
    updated_at = db.Column(db.TIMESTAMP, default=utc_now, onupdate=utc_now)
    usage_logs = db.relationship('UsageLog', backref='user', lazy=True, cascade='all, delete-orphan')
    settings = db.relationship('UserSettings', backref='user', uselist=False, cascade='all, delete-orphan')
    history = db.relationship('RankingHistory', backref='user', lazy=True, cascade='all, delete-orphan')

class UsageLog(db.Model):
    __tablename__ = 'usage_logs'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    session_date = db.Column(db.Date, nullable=False)
    sessions_used = db.Column(db.Integer, default=0)
    cvs_processed = db.Column(db.Integer, default=0)
    created_at = db.Column(db.TIMESTAMP, default=utc_now)
    __table_args__ = (db.UniqueConstraint('user_id', 'session_date', name='unique_user_date'),)

class UserSettings(db.Model):
    __tablename__ = 'user_settings'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False, unique=True)
    experience_enabled = db.Column(db.Boolean, default=True)
    min_experience = db.Column(db.Integer, default=0)
    min_experience_unit = db.Column(db.String(10), default='years')
    max_experience = db.Column(db.Integer, default=10)
    max_experience_unit = db.Column(db.String(10), default='years')
    show_score_bar = db.Column(db.Boolean, default=True)
    save_cvs = db.Column(db.Boolean, default=False)
    cv_limit = db.Column(db.Integer, default=20)

class RankingHistory(db.Model):
    __tablename__ = 'ranking_history'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    job_title = db.Column(db.String(255), nullable=False, default='Job Ranking')
    date_created = db.Column(db.TIMESTAMP, default=utc_now)
    cv_count = db.Column(db.Integer, default=0)
    top_score = db.Column(db.DECIMAL(5,2), default=0)
    top_candidate = db.Column(db.String(255))
    results_json = db.Column(db.Text)

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please log in to access this page.', 'warning')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

PLAN_LIMITS = {
    'free': {'cvs_per_session': 10, 'max_sessions': 1, 'price': 0},
    'basic': {'cvs_per_session': 25, 'max_sessions': 2, 'price': 10},
    'pro': {'cvs_per_session': 50, 'max_sessions': 3, 'price': 20},
    'premium': {'cvs_per_session': float('inf'), 'max_sessions': float('inf'), 'price': 50}
}

# =============================================================================
# FIXED: SQLAlchemy 2.0 style — db.session.get() instead of Model.query.get()
# =============================================================================
def get_current_user():
    if 'user_id' not in session:
        return None
    return db.session.get(User, session['user_id'])

def check_subscription_status(user):
    if not user:
        return False
    if user.subscription_plan == 'free':
        return True
    if user.subscription_expires and user.subscription_expires < utc_now():
        user.subscription_plan = 'free'
        db.session.commit()
        return False
    return True

def check_usage_limit(user_id, cvs_count):
    user = db.session.get(User, user_id)
    if not user:
        return False, "User not found", {}
    if not check_subscription_status(user):
        return False, "Your subscription has expired. Please renew your plan.", {
            'limit_type': 'subscription', 'plan': user.subscription_plan
        }
    today = date.today()
    plan = user.subscription_plan
    limits = PLAN_LIMITS[plan]
    usage = UsageLog.query.filter_by(user_id=user_id, session_date=today).first()
    if not usage:
        usage = UsageLog(user_id=user_id, session_date=today, sessions_used=0, cvs_processed=0)
        db.session.add(usage)
        db.session.commit()
    if usage.sessions_used >= limits['max_sessions']:
        return False, f"Daily session limit reached! You've used {usage.sessions_used}/{limits['max_sessions']} sessions today. Upgrade your plan or try again tomorrow.", {
            'limit_type': 'sessions', 'current': usage.sessions_used, 'max': limits['max_sessions'], 'plan': plan
        }
    if cvs_count > limits['cvs_per_session']:
        return False, f"CV limit exceeded! {plan.capitalize()} plan allows max {int(limits['cvs_per_session'])} CVs per session. You tried to upload {cvs_count} CV(s).", {
            'limit_type': 'cvs', 'allowed': int(limits['cvs_per_session']), 'requested': cvs_count, 'plan': plan
        }
    return True, "Within limits", {
        'sessions_used': usage.sessions_used,
        'sessions_remaining': limits['max_sessions'] - usage.sessions_used,
        'plan': plan
    }

def record_usage(user_id, cvs_count):
    today = date.today()
    usage = UsageLog.query.filter_by(user_id=user_id, session_date=today).first()
    if usage:
        usage.sessions_used += 1
        usage.cvs_processed += cvs_count
        db.session.commit()
    return usage

def save_to_history(user_id, job_title, cv_count, results):
    if not results:
        return
    top_candidate = results[0] if results else None
    history_entry = RankingHistory(
        user_id=user_id,
        job_title=job_title[:50] if job_title else 'Untitled Ranking',
        cv_count=cv_count,
        top_score=top_candidate['score'] if top_candidate else 0,
        top_candidate=top_candidate['name'][:50] if top_candidate else 'Unknown',
        results_json=json.dumps(results)
    )
    db.session.add(history_entry)
    db.session.commit()

# =============================================================================
# ROUTES
# =============================================================================
@app.route('/')
@login_required
def home():
    return render_template('home.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        user = User.query.filter_by(email=email).first()
        password_valid = False
        if user and user.password_hash:
            if check_password_hash(user.password_hash, password):
                password_valid = True
            elif user.password_hash == hashlib.sha256(password.encode()).hexdigest():
                user.password_hash = generate_password_hash(password)
                db.session.commit()
                password_valid = True
        if user and password_valid:
            session['user_id'] = user.id
            session['user_email'] = user.email
            session['user_profile_pic'] = user.profile_pic or ''
            flash(f'Welcome back, {email}!', 'success')
            return redirect(url_for('home'))
        else:
            flash('Invalid email or password.', 'error')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        confirm_password = request.form.get('confirm_password', '')
        if not re.match(r'^[\w\.-]+@[\w\.-]+\.\w+$', email):
            flash('Please enter a valid email address.', 'error')
            return redirect(url_for('register'))
        if len(password) < 6:
            flash('Password must be at least 6 characters long.', 'error')
            return redirect(url_for('register'))
        if password != confirm_password:
            flash('Passwords do not match.', 'error')
            return redirect(url_for('register'))
        if User.query.filter_by(email=email).first():
            flash('Email already registered.', 'error')
            return redirect(url_for('register'))
        user = User(email=email, password_hash=generate_password_hash(password), subscription_plan='free')
        db.session.add(user)
        db.session.commit()
        settings = UserSettings(user_id=user.id)
        db.session.add(settings)
        db.session.commit()
        flash('Registration successful! Please log in.', 'success')
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))

@app.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():
    user = get_current_user()
    if not user:
        return redirect(url_for('login'))
    if request.method == 'POST':
        if 'profile_pic' in request.files:
            file = request.files['profile_pic']
            if file and file.filename != '':
                filename = secure_filename(f"user_{user.id}_{file.filename}")
                if not allowed_file(filename):
                    flash('Invalid file type. Please upload a PNG, JPG, JPEG, or GIF image.', 'error')
                    return redirect(url_for('profile'))
                upload_dir = os.path.join(app.static_folder, 'uploads', 'profile_pics')
                os.makedirs(upload_dir, exist_ok=True)
                file_path = os.path.join(upload_dir, filename)
                file.save(file_path)
                if user.profile_pic:
                    old_path = os.path.join(app.static_folder, user.profile_pic)
                    if os.path.exists(old_path):
                        os.remove(old_path)
                user.profile_pic = f"uploads/profile_pics/{filename}"
                db.session.commit()
                session['user_profile_pic'] = user.profile_pic
                flash('Profile picture updated successfully!', 'success')
                return redirect(url_for('profile'))
    today = date.today()
    usage = UsageLog.query.filter_by(user_id=user.id, session_date=today).first()
    sessions_used = usage.sessions_used if usage else 0
    plan_limits = PLAN_LIMITS[user.subscription_plan]
    return render_template('profile.html', user=user, plan_limits=plan_limits, sessions_used=sessions_used)

@app.route('/rank', methods=['POST'])
@login_required
def rank():
    user = get_current_user()
    if not user:
        flash('Session expired. Please log in again.', 'error')
        return redirect(url_for('login'))

    job_description = request.form.get('job_description', '').strip()
    files = request.files.getlist('cvs')
    valid_files = [f for f in files if f.filename != '']

    allowed, message, details = check_usage_limit(user.id, len(valid_files))
    if not allowed:
        flash(message, 'error')
        return redirect('/')

    if not job_description:
        flash('Please enter a job description', 'warning')
        return redirect('/')
    if not valid_files:
        flash('Please upload at least one CV', 'warning')
        return redirect('/')

    cv_texts = []
    cv_names = []
    failed_files = []

    for f in valid_files:
        text, error = extract_text_from_pdf(f)
        if text:
            cv_texts.append(text)
            cv_names.append(f.filename)
        else:
            failed_files.append(f"{f.filename}: {error}")

    if failed_files:
        error_msg = "Some files could not be processed:<br>" + "<br>".join(failed_files)
        flash(error_msg, 'warning')

    if not cv_texts:
        flash('No readable CVs found. Please upload text-based PDFs (not scanned images).', 'error')
        return redirect('/')

    print(f"\n{'='*50}")
    print(f"RANKING DEBUG")
    print(f"{'='*50}")
    print(f"Job description: {len(job_description)} chars")
    for name, text in zip(cv_names, cv_texts):
        print(f"  {name}: {len(text)} chars | {len(text.split())} words")
    print(f"{'='*50}\n")

    flash(f'Successfully processed {len(cv_texts)} CV(s). Analyzing...', 'success')

    scores = compute_similarity_scores(job_description, cv_texts)

    print(f"Raw scores: {list(zip(cv_names, scores))}\n")

    results = sorted(
        [{"name": cv_names[i], "score": scores[i]} for i in range(len(cv_names))],
        key=lambda x: x["score"],
        reverse=True
    )

    record_usage(user.id, len(valid_files))
    save_to_history(user.id, job_description[:100], len(valid_files), results)

    return render_template('new_ranking.html', job=job_description, results=results, view_mode=False)

@app.route('/history')
@login_required
def history():
    user = get_current_user()
    if not user:
        return redirect(url_for('login'))
    history_data = RankingHistory.query.filter_by(user_id=user.id).order_by(RankingHistory.date_created.desc()).all()
    formatted_history = []
    for h in history_data:
        formatted_history.append({
            'id': h.id,
            'job_title': h.job_title,
            'date': h.date_created.strftime('%b %d, %Y'),
            'date_raw': h.date_created.isoformat(),
            'cv_count': h.cv_count,
            'top_score': float(h.top_score),
            'top_candidate': h.top_candidate
        })
    return render_template('history.html', history_data=formatted_history)

@app.route('/history/view/<int:history_id>')
@login_required
def view_history_detail(history_id):
    user = get_current_user()
    if not user:
        return redirect(url_for('login'))
    history_entry = RankingHistory.query.filter_by(id=history_id, user_id=user.id).first_or_404()
    try:
        results = json.loads(history_entry.results_json) if history_entry.results_json else []
    except:
        results = []
    return render_template('new_ranking.html', job=history_entry.job_title, results=results, view_mode=True, history_id=history_id)

@app.route('/settings')
@login_required
def settings():
    user = get_current_user()
    if not user:
        return redirect(url_for('login'))
    today = date.today()
    usage = UsageLog.query.filter_by(user_id=user.id, session_date=today).first()
    sessions_used = usage.sessions_used if usage else 0
    user_settings = UserSettings.query.filter_by(user_id=user.id).first()
    plan_limits = PLAN_LIMITS[user.subscription_plan]
    return render_template('settings.html', user=user, user_settings=user_settings, plan_limits=plan_limits, sessions_used=sessions_used, PLAN_LIMITS=PLAN_LIMITS)

@app.route('/api/settings', methods=['POST'])
@login_required
def save_settings():
    user = get_current_user()
    if not user:
        return jsonify({'error': 'Not authenticated'}), 401
    data = request.json
    settings = UserSettings.query.filter_by(user_id=user.id).first()
    if not settings:
        settings = UserSettings(user_id=user.id)
        db.session.add(settings)
    settings.experience_enabled = data.get('experienceEnabled', True)
    settings.min_experience = data.get('minExperience', 0)
    settings.min_experience_unit = data.get('minExperienceUnit', 'years')
    settings.max_experience = data.get('maxExperience', 10)
    settings.max_experience_unit = data.get('maxExperienceUnit', 'years')
    settings.show_score_bar = data.get('showScoreBar', True)
    settings.save_cvs = data.get('saveCvs', False)
    settings.cv_limit = data.get('cvLimit', 20)
    db.session.commit()
    return jsonify({'success': True})

@app.route('/api/usage-status')
@login_required
def usage_status():
    user = get_current_user()
    if not user:
        return jsonify({'error': 'Not authenticated'}), 401
    today = date.today()
    usage = UsageLog.query.filter_by(user_id=user.id, session_date=today).first()
    plan = PLAN_LIMITS[user.subscription_plan]
    return jsonify({
        'plan': user.subscription_plan,
        'sessions_used': usage.sessions_used if usage else 0,
        'sessions_limit': plan['max_sessions'] if plan['max_sessions'] != float('inf') else 'Unlimited',
        'cvs_per_session': plan['cvs_per_session'] if plan['cvs_per_session'] != float('inf') else 'Unlimited',
        'subscription_expires': user.subscription_expires.isoformat() if user.subscription_expires else None
    })

@app.route('/subscribe/<plan>')
@login_required
def subscribe(plan):
    if plan not in PLAN_LIMITS:
        return "Invalid plan", 400
    token = request.args.get('token')
    if token != 'demo-confirm':
        flash('Invalid subscription request. Please use the pricing page to upgrade.', 'error')
        return redirect(url_for('pricing'))
    user = get_current_user()
    if not user:
        return redirect(url_for('login'))
    user.subscription_plan = plan
    user.subscription_expires = datetime(2026, 12, 31)
    db.session.commit()
    flash(f'Upgraded to {plan.capitalize()} plan! You can now process {PLAN_LIMITS[plan]["cvs_per_session"]} CVs per session.', 'success')
    return redirect(url_for('settings'))

@app.route('/api/delete-history', methods=['POST'])
@login_required
def delete_history():
    user = get_current_user()
    if not user:
        return jsonify({'error': 'Not authenticated'}), 401
    RankingHistory.query.filter_by(user_id=user.id).delete()
    db.session.commit()
    return jsonify({'success': True})

@app.route('/pricing')
@login_required
def pricing():
    user = get_current_user()
    if not user:
        return redirect(url_for('login'))
    return render_template('pricing.html', user=user, PLAN_LIMITS=PLAN_LIMITS, current_plan=user.subscription_plan)

@app.route('/how-it-works')
def how_it_works():
    return render_template('how_it_works.html')

@app.route('/faq')
def faq():
    return render_template('faq.html')

@app.route('/changelog')
def changelog():
    return render_template('changelog.html')

@app.route('/presentation')
def presentation():
    return app.send_static_file('rankcv_present.html')

@app.errorhandler(404)
def not_found(e):
    return "<h1 style='text-align:center;margin-top:50px;'>404 — Page Not Found</h1><p style='text-align:center;'><a href='/'>Go Home</a></p>", 404

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        print("MySQL tables created/verified")
        print("Ready to run! Visit http://localhost:5000")
    app.run(debug=True, host='0.0.0.0')