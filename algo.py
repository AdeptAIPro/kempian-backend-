import os
import threading
import json
import re
import math
from datetime import datetime, timedelta
from functools import lru_cache
from collections import Counter

import boto3
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from sentence_transformers import SentenceTransformer
import requests
import openai
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import logging
from dotenv import load_dotenv

# Setup
dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path)
nltk.download('stopwords')
openai.api_key = os.getenv("OPENAI_API_KEY")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": ["http://localhost:5055", "http://127.0.0.1:5055"]}})

REGION = 'ap-south-1'
dynamodb = boto3.resource('dynamodb', region_name=REGION,
                          aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                          aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"))
# table = dynamodb.Table('resume_metadata')
table = dynamodb.Table('user-resume-metadata')
feedback_table = dynamodb.Table('resume_feedback')
model = SentenceTransformer('all-MiniLM-L6-v2')

FEEDBACK_FILE = 'feedback.json'


def load_feedback():
    if os.path.exists(FEEDBACK_FILE):
        with open(FEEDBACK_FILE, 'r') as f:
            return json.load(f)
    return {}


def save_feedback(data):
    with open(FEEDBACK_FILE, 'w') as f:
        json.dump(data, f)


def register_feedback(candidate_id, positive=True):
    data = load_feedback()
    if candidate_id not in data:
        data[candidate_id] = {"positive": 0, "negative": 0}
    if positive:
        data[candidate_id]["positive"] += 1
    else:
        data[candidate_id]["negative"] += 1
    save_feedback(data)


DOMAIN_KEYWORDS = {
    "software": {
        'developer', 'engineer', 'backend', 'frontend', 'fullstack', 'programmer', 'software',
        'ai', 'ml', 'machine learning', 'data', 'api', 'apis', 'rest', 'graphql',
        'javascript', 'typescript', 'react', 'node', 'python', 'java', 'flask', 'django',
        'spring', 'springboot', 'hibernate', 'express', 'fastapi', 'nextjs', 'angular',
        'aws', 'gcp', 'azure', 'cloud', 'devops', 'microservices', 'docker', 'kubernetes',
        'lambda', 'serverless', 'terraform', 'ansible', 'jenkins', 'ci', 'cd',
        'linux', 'bash', 'shell', 'sql', 'nosql', 'mongodb', 'postgresql', 'mysql',
        'security', 'cybersecurity', 'firewall', 'penetration', 'siem', 'compliance',
        'iso', 'soc2', 'ceh', 'comptia', 'agile', 'scrum', 'jira', 'git', 'github', 'bitbucket',
        'unit testing', 'integration testing', 'automation', 'selenium', 'pytest', 'cypress',
        'nlp', 'cv', 'transformer', 'bert', 'gpt', 'llm', 'huggingface', 'pytorch', 'tensorflow',
        'pandas', 'numpy', 'matplotlib', 'scikit-learn', 'kafka', 'redis', 'elasticsearch',
        'firebase', 'sentry', 'newrelic', 'logstash', 'prometheus', 'grafana',
    },
    "healthcare": {
    'nurse', 'nursing', 'rn', 'bsc nursing', 'm.sc nursing', 'icu', 'surgery',
    'healthcare', 'patient', 'hospital', 'medical', 'clinical', 'ward', 'gynae',
    'cardiology', 'therapeutic', 'anesthesia', 'nursys', 'registered nurse',
    'cna', 'care', 'charting', 'vitals', 'mobility', 'therapy', 'rehab',
    'phlebotomy', 'pediatrics', 'geriatrics', 'ophthalmology', 'dermatology',
    'radiology', 'oncology', 'pharmacy', 'diagnosis', 'prescription', 'labs',
    'first aid', 'emergency', 'triage', 'bcls', 'acls', 'infection control',
    'patient care', 'clinical documentation', 'medication', 'wound care',
    'telemedicine', 'public health', 'mental health', 'physician', 'assistant',
    'doctor', 'dentist', 'midwife', 'vaccination', 'epidemiology', 'biomedical',
    'health record', 'ehr', 'emr', 'insurance', 'hipaa', 'claims', 'billing',

    
    'lab technician', 'radiographer', 'ultrasound', 'x-ray', 'immunization',
    'hematology', 'pathology', 'microbiology', 'clinical trials', 'vaccine',
    'occupational therapy', 'speech therapy', 'physical therapy', 'audiology',
    'home health', 'ambulatory care', 'long-term care', 'geriatrics nurse',
    'palliative care', 'end of life care', 'hospice', 'dementia', 'alzheimers',
    'behavioral health', 'psychology', 'psychiatry', 'mental illness', 'counseling',
    'blood pressure', 'temperature monitoring', 'surgical tech', 'scrub nurse',

    
    'health informatics', 'clinical informatics', 'medical coding', 'icd-10',
    'cpt coding', 'hl7', 'fhir', 'pacs', 'ris', 'health it', 'medical records',
    'case manager', 'insurance claims', 'utilization review', 'care coordinator',
    'revenue cycle', 'medical scribe', 'compliance', 'regulatory', 'audit',
    'cms', 'medicare', 'medicaid', 'prior authorization', 'medical transcription',
    'ehr implementation', 'healthcare analytics', 'population health', 'care quality',
    'patient satisfaction', 'value-based care', 'telehealth', 'virtual care',
    'remote monitoring', 'patient portal', 'healthcare provider'
}
}


class CeipalAuth:
    def __init__(self):
        self.auth_url = "https://api.ceipal.com/v1/createAuthtoken/"
        self.email = os.getenv("CEIPAL_EMAIL")
        self.password = os.getenv("CEIPAL_PASSWORD")
        self.api_key = os.getenv("CEIPAL_API_KEY")
        self.token = None
        self.token_expiry = None

    def authenticate(self):
        payload = {"email": self.email, "password": self.password, "api_key": self.api_key, "json": "1"}
        try:
            response = requests.post(self.auth_url, json=payload)
            response.raise_for_status()
            data = response.json()
            if "access_token" in data:
                self.token = data["access_token"]
                self.token_expiry = datetime.now() + timedelta(hours=1)
                return True
        except Exception as e:
            logger.error(f"CEIPAL auth error: {str(e)}")
        return False

    def get_token(self):
        if not self.token or datetime.now() >= self.token_expiry:
            if not self.authenticate():
                return None
        return self.token


class CeipalJobPostingsAPI:
    def __init__(self, auth):
        self.auth = auth
        self.base_url = "https://api.ceipal.com"
        self.job_postings_endpoint = "/getCustomJobPostingDetails/Z3RkUkt2OXZJVld2MjFpOVRSTXoxZz09/e6e04af381e7f42eeb7f942c8bf5ab6d"
        self.job_details_endpoint = "/v1/getJobPostingDetails/"

    def get_job_postings(self, paging_length=30):
        token = self.auth.get_token()
        headers = {"Authorization": f"Bearer {token}"}
        response = requests.get(f"{self.base_url}{self.job_postings_endpoint}", headers=headers,
                                params={"paging_length": paging_length})
        return response.json().get("results", [])

    def get_job_details(self, job_code):
        token = self.auth.get_token()
        headers = {"Authorization": f"Bearer {token}"}
        response = requests.get(f"{self.base_url}{self.job_details_endpoint}", headers=headers,
                                params={"job_id": job_code})
        return response.json()


def extract_keywords(text):
    tokenizer = RegexpTokenizer(r'\w+')
    words = tokenizer.tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    exclusions = {"year", "years", "experience", "skills", "development", "team", "platform", "plan"}
    words = [w for w in words if w not in stop_words and w not in exclusions and len(w) > 2]
    keyword_freq = Counter(words)
    return [word for word, freq in keyword_freq.items() if freq >= 2 or word in DOMAIN_KEYWORDS['software'] or word in DOMAIN_KEYWORDS['healthcare']]


def detect_domain(keywords):
    sw = sum(1 for w in keywords if w in DOMAIN_KEYWORDS['software'])
    hw = sum(1 for w in keywords if w in DOMAIN_KEYWORDS['healthcare'])
    return 'software' if sw >= hw else 'healthcare'


def get_grade(score):
    if score >= 85:
        return 'A'
    elif score >= 70:
        return 'B'
    elif score >= 50:
        return 'C'
    else:
        return 'D'


def nlrga_score(matched_keywords, total_keywords, candidate_id):
    base_ratio = len(matched_keywords) / max(total_keywords, 1)
    feedback = load_feedback().get(candidate_id, {"positive": 0, "negative": 0})
    feedback_factor = 1 / (1 + math.exp(-(feedback['positive'] - feedback['negative'])))
    combined = base_ratio * 0.7 + feedback_factor * 0.3
    return round(combined * 100, 2)


def keyword_search(job_description, top_k=10):
    query_keywords = extract_keywords(job_description)
    domain = detect_domain(query_keywords)
    domain_keywords = DOMAIN_KEYWORDS[domain]

    response = table.scan()
    results = []

    for item in response.get('Items', []):
        if not item.get('resume_text') or 'skills' not in item or not item['skills']:
            continue

        resume_text = item['resume_text'].lower() + ' ' + ' '.join(item['skills']).lower()
        resume_words = extract_keywords(resume_text)

        domain_overlap = set(resume_words).intersection(domain_keywords)
        keyword_overlap = set(resume_words).intersection(query_keywords)
        match_percent = len(keyword_overlap) / max(len(query_keywords), 1)

        if match_percent >= 0.4 and domain_overlap:
            score = nlrga_score(keyword_overlap, len(query_keywords), item.get('CandidateID', item['email']))
            results.append({
                'FullName': item.get('full_name'),
                'email': item.get('email'),
                'phone': item.get('phone'),
                'Skills': item.get('skills', []),
                'Certifications': item.get('certifications', []),
                'Education': item.get('education'),
                'Experience': f"{item.get('total_experience_years', 0)} years",
                'ResumeFile': item.get('filename'),
                'sourceURL': item.get('sourceURL'),
                'MatchPercent': round(match_percent * 100, 2),
                'Score': score,
                'Grade': get_grade(score)
            })

            
            if 'sourceURL' in item:
                results[-1]['sourceURL'] = item['sourceURL']
    results.sort(key=lambda x: x['Score'], reverse=True)
    return results[:top_k], f"Top {len(results[:top_k])} candidates found."


def rerank_with_gpt4(job_description, candidates):
    prompt = f"""You are an expert recruiter. Given the job description below, rerank the following candidates based on their skills, certifications, education, and experience.\n\nJob Description:\n{job_description}\n\nCandidates:\n"""
    for i, c in enumerate(candidates, 1):
        prompt += f"\n{i}. {c['FullName']}\n- Skills: {', '.join(c.get('Skills', []))}\n- Certifications: {', '.join(c.get('Certifications', []))}\n- Education: {c.get('Education')}\n- Experience: {c.get('Experience')}\n- sourceURL: {c.get('sourceURL')}"

    prompt += "\n\nReturn the top candidates in order with reasoning for each. Format:\nRank. Name - Reason"

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        explanation_text = response['choices'][0]['message']['content']
        explanation_lines = explanation_text.strip().split("\n")

        ranked = []
        for line in explanation_lines:
            match = re.match(r"\d+\.\s*(.+?)\s*-\s*(.+)", line)
            if match:
                name = match.group(1).strip()
                reason = match.group(2).strip()
                candidate = next((c for c in candidates if c['FullName'] == name), None)
                if candidate:
                    candidate["LLM_Reasoning"] = reason
                    ranked.append(candidate)
        return ranked, "Top candidates reranked by GPT-4 with reasoning."
    except Exception as e:
        logger.error(f"GPT-4 reranking error: {e}")
        return candidates, "Fallback to semantic ranking due to GPT-4 error."


# Plan quotas
PLAN_QUOTAS = {
    "starter": 50,
    "growth": 500,
    "professional": 2500,
    "enterprise": 10000
}

# DynamoDB setup for user and search logs
users_table = dynamodb.Table('users')  # Table with user info (id, email, plan, etc.)
jd_search_logs_table = dynamodb.Table('jd_search_logs')  # Table to log searches

def get_user_info(email):
    resp = users_table.scan(FilterExpression=boto3.dynamodb.conditions.Attr('email').eq(email))
    items = resp.get('Items', [])
    if not items:
        return None
    return items[0]

def get_search_count_this_month(user_id):
    now = datetime.utcnow()
    month_start = datetime(now.year, now.month, 1)
    # Query for logs this month
    resp = jd_search_logs_table.scan(
        FilterExpression=boto3.dynamodb.conditions.Attr('user_id').eq(user_id) &
                         boto3.dynamodb.conditions.Attr('searched_at').gte(month_start.isoformat())
    )
    return len(resp.get('Items', []))

def log_search(user_id):
    now = datetime.utcnow().isoformat()
    jd_search_logs_table.put_item(Item={
        'id': f'{user_id}-{now}',
        'user_id': user_id,
        'searched_at': now
    })

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/search', methods=['POST'])
def search():
    data = request.get_json()
    query = data.get('query')
    algo = data.get('algorithm', 'keyword')
    email = data.get('email')  # Must be sent from frontend

    if not query:
        return jsonify({"error": "Empty query"}), 400
    if not email:
        return jsonify({"error": "Missing user email"}), 400

    user = get_user_info(email)
    if not user:
        return jsonify({"error": "User not found"}), 404

    plan = user.get('plan', 'starter').lower()
    quota = PLAN_QUOTAS.get(plan, 50)
    user_id = user['id']

    # Count searches this month
    search_count = get_search_count_this_month(user_id)
    if search_count >= quota:
        return jsonify({
            "error": f"Search quota exceeded for your plan ({quota} per month).",
            "quota": quota,
            "used": search_count,
            "remaining": 0
        }), 403

    # Log this search
    log_search(user_id)

    if algo == 'keyword':
        results, summary = keyword_search(query)
    elif algo == 'llm':
        top_k_results, _ = keyword_search(query, top_k=15)
        results, summary = rerank_with_gpt4(query, top_k_results)
    else:
        results, summary = [], 'Unsupported algorithm.'

    return jsonify({
        "results": results,
        "summary": summary,
        "quota": quota,
        "used": search_count + 1,
        "remaining": max(0, quota - (search_count + 1))
    })


@app.route('/feedback', methods=['POST'])
def feedback():
    data = request.get_json()
    candidate_id = data.get('email')
    is_positive = data.get('feedback') == 'good'
    register_feedback(candidate_id, positive=is_positive)
    return jsonify({"status": "success"})


@app.route('/api/v1/ceipal/jobs', methods=['GET'])
def get_ceipal_jobs():
    try:
        count = int(request.args.get('count', 50))
        auth = CeipalAuth()
        if not auth.authenticate():
            logger.error("Failed to authenticate with CEIPAL")
            return jsonify({"error": "Authentication failed"}), 401
        jobs = CeipalJobPostingsAPI(auth).get_job_postings(paging_length=count)
        return jsonify({"jobs": jobs})
    except Exception as e:
        logger.error(f"Error fetching CEIPAL jobs: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/v1/ceipal/getJobDetails', methods=['GET'])
def get_job_details():
    job_code = request.args.get('job_code')
    if not job_code:
        return jsonify({"error": "Missing job_code parameter"}), 400
    try:
        job = CeipalJobPostingsAPI(CeipalAuth()).get_job_details(job_code)
        return jsonify({"job_details": job})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    if not all([os.getenv(v) for v in ["CEIPAL_EMAIL", "CEIPAL_PASSWORD", "CEIPAL_API_KEY", "OPENAI_API_KEY"]]):
        logger.warning("CEIPAL or OpenAI credentials are not fully set.")
    app.run(host='127.0.0.1', port=8000, debug=True)