import os
import boto3
import math
import json
import re
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from sentence_transformers import SentenceTransformer
import openai
import nltk
nltk.download('stopwords')

# Setup (assume environment already loaded)
REGION = 'ap-south-1'
dynamodb = boto3.resource('dynamodb', region_name=REGION,
                          aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                          aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"))
table = dynamodb.Table('user-resume-metadata')
model = SentenceTransformer('all-MiniLM-L6-v2')



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
        'patient satisfaction', 'value-based care', 'telehealth',
        'remote monitoring', 'patient portal', 'healthcare provider'
    }
}


def extract_keywords(text):
    if not text:
        return []
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
    # Feedback logic omitted for now
    base_ratio = len(matched_keywords) / max(total_keywords, 1)
    feedback_factor = 0.5  # Placeholder, can be extended
    combined = base_ratio * 0.7 + feedback_factor * 0.3
    return round(combined * 100, 2)

def keyword_search(job_description, top_k=10):
    query_keywords = extract_keywords(job_description)
    print("[DEBUG] Query keywords:", query_keywords)
    domain = detect_domain(query_keywords)
    domain_keywords = DOMAIN_KEYWORDS[domain]
    response = table.scan()
    results = []
    for item in response.get('Items', []):
        if not item.get('resume_text') or 'skills' not in item or not item['skills']:
            print(f"[DEBUG] Skipping {item.get('full_name', item.get('email', 'unknown'))} due to missing resume_text or skills")
            continue
        resume_text = item['resume_text'].lower() + ' ' + ' '.join(item['skills']).lower()
        resume_words = extract_keywords(resume_text)
        print(f"[DEBUG] Resume words for {item.get('full_name', item.get('email', 'unknown'))}: {resume_words}")
        domain_overlap = set(resume_words).intersection(domain_keywords)
        keyword_overlap = set(resume_words).intersection(query_keywords)
        match_percent = len(keyword_overlap) / max(len(query_keywords), 1)
        print(f"[DEBUG] Candidate: {item.get('full_name', item.get('email', 'unknown'))}, domain_overlap: {domain_overlap}, keyword_overlap: {keyword_overlap}, match_percent: {match_percent}")
        # Lower threshold for debugging
        if match_percent >=0.4 and domain_overlap:
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
                'Location': item.get('location'),
                'Title': item.get('title'),
                'LinkedIn': item.get('linkedin'),
                'MatchPercent': round(match_percent * 100, 2),
                'Score': score,
                'Grade': get_grade(score),
                # Add any other available fields
                'summary': item.get('summary'),
                'profile_pic': item.get('profile_pic'),
                'contactInfo': item.get('contactInfo'),
                'projects': item.get('projects'),
                'awards': item.get('awards'),
                'publications': item.get('publications'),
                'languages': item.get('languages'),
                'interests': item.get('interests'),
                'github': item.get('github'),
                'website': item.get('website'),
                'twitter': item.get('twitter'),
                'facebook': item.get('facebook'),
                'instagram': item.get('instagram'),
                'dob': item.get('dob'),
                'gender': item.get('gender'),
                'address': item.get('address'),
                'city': item.get('city'),
                'state': item.get('state'),
                'country': item.get('country'),
                'zipcode': item.get('zipcode'),
                'alternate_email': item.get('alternate_email'),
                'alternate_phone': item.get('alternate_phone'),
                'marital_status': item.get('marital_status'),
                'nationality': item.get('nationality'),
                'work_authorization': item.get('work_authorization'),
                'relocation': item.get('relocation'),
                'notice_period': item.get('notice_period'),
                'current_employer': item.get('current_employer'),
                'current_salary': item.get('current_salary'),
                'expected_salary': item.get('expected_salary'),
                'availability': item.get('availability'),
                'portfolio': item.get('portfolio'),
                'references': item.get('references'),
                'custom_fields': item.get('custom_fields'),
            })
            if 'sourceURL' in item:
                results[-1]['sourceURL'] = item['sourceURL']
    results.sort(key=lambda x: x['Score'], reverse=True)
    return results[:top_k], f"Top {len(results[:top_k])} candidates found."

def semantic_match(job_description):
    # For now, use keyword search only
    results, summary = keyword_search(job_description)
    return {
        'results': results,
        'summary': summary
    } 