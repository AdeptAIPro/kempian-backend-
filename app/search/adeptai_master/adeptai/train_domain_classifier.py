"""
Training script for ML Domain Classifier

This script demonstrates how to train the ML domain classifier using labeled data.
"""

import os
import sys
from typing import List, Tuple

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_domain_classifier import MLDomainClassifier, get_ml_domain_classifier


def prepare_training_data() -> Tuple[List[str], List[str]]:
    """
    Prepare training data for domain classification.
    
    In a real scenario, you would:
    1. Load labeled data from your database/logs
    2. Extract candidate text (skills, resume_text, etc.)
    3. Assign domain labels based on ground truth
    4. Clean and preprocess the data
    
    Returns:
        Tuple of (texts, labels)
    """
    # Example training data
    # In production, load this from your database
    
    texts = []
    labels = []
    
    # Technology examples
    tech_texts = [
        "Senior Python developer with 8 years of experience in AWS cloud services, Docker, and Kubernetes. Expert in backend development with Django and FastAPI.",
        "Full-stack JavaScript developer proficient in React, Node.js, and MongoDB. Experience with microservices architecture and CI/CD pipelines.",
        "Data scientist with expertise in machine learning, Python, TensorFlow, and data analysis. Strong background in statistical modeling.",
        "DevOps engineer with experience in AWS, Azure, Kubernetes, Docker, and infrastructure automation. Proficient in Terraform and Ansible.",
        "Software engineer specialized in Java, Spring Boot, and microservices. Experience with cloud platforms and distributed systems.",
        "Frontend developer with React, Vue.js, and TypeScript expertise. Strong UI/UX design skills and responsive web development.",
        "Backend engineer with Python, Django, PostgreSQL, and REST API development. Experience with scalable system architecture.",
        "Cloud architect with AWS, Azure, and GCP certifications. Expert in cloud infrastructure design and optimization.",
        "AI/ML engineer with deep learning, neural networks, and natural language processing. Proficient in PyTorch and TensorFlow.",
        "Full-stack developer with MEAN stack (MongoDB, Express, Angular, Node.js) and modern web technologies."
    ]
    texts.extend(tech_texts)
    labels.extend(['technology'] * len(tech_texts))
    
    # Healthcare examples
    healthcare_texts = [
        "Registered Nurse with 7 years of experience in ICU and emergency medicine. ACLS and BLS certified with expertise in patient care.",
        "Board-certified Internal Medicine physician with 8 years of experience in hospital and clinic settings. Expert in patient care, diagnosis, and treatment planning.",
        "Pediatric Registered Nurse with 5 years experience in children's hospital. Expert in pediatric patient care and family education.",
        "Emergency Medicine physician with 10 years experience in trauma centers. Expert in critical care and emergency medical procedures.",
        "Cardiologist with 12 years of experience in hospital settings. Specialized in echocardiography and cardiac patient care.",
        "Surgical Registered Nurse with 6 years experience in operating room settings. Expert in surgical procedures and patient recovery care.",
        "Clinical pharmacist with expertise in medication management and patient counseling. Experience in hospital and retail pharmacy settings.",
        "Physical therapist specializing in rehabilitation and injury recovery. Experience with orthopedic and neurological conditions.",
        "Medical laboratory technician with expertise in clinical testing and diagnostic procedures. Proficient in laboratory equipment and protocols.",
        "Nurse practitioner with primary care experience. Skilled in patient assessment, diagnosis, and treatment planning."
    ]
    texts.extend(healthcare_texts)
    labels.extend(['healthcare'] * len(healthcare_texts))
    
    # Finance examples
    finance_texts = [
        "Financial analyst with 5 years of experience in investment analysis and portfolio management. CFA certified with expertise in financial modeling.",
        "Senior accountant with CPA certification and 8 years of experience in financial reporting, auditing, and tax preparation.",
        "Investment banker with expertise in mergers and acquisitions, capital markets, and corporate finance. Strong analytical and negotiation skills.",
        "Risk management analyst specialized in credit risk, market risk, and operational risk. Experience with regulatory compliance and risk assessment.",
        "Financial advisor with expertise in retirement planning, investment advisory, and wealth management. Series 7 and Series 66 licenses.",
        "Corporate finance manager with experience in budgeting, forecasting, and financial planning. Strong background in financial analysis and reporting.",
        "Tax accountant with expertise in individual and corporate tax preparation, tax planning, and compliance. CPA certified.",
        "Treasury analyst with experience in cash management, liquidity planning, and financial risk management. Proficient in financial modeling.",
        "Credit analyst specialized in credit risk assessment, loan underwriting, and portfolio management. Strong analytical skills.",
        "Compliance officer with expertise in regulatory compliance, risk management, and financial regulations. Experience with audit and monitoring."
    ]
    texts.extend(finance_texts)
    labels.extend(['finance'] * len(finance_texts))
    
    # Education examples
    education_texts = [
        "High school mathematics teacher with 10 years of experience in curriculum development and student assessment. Expert in algebra, geometry, and calculus instruction.",
        "University professor of computer science with research expertise in machine learning and artificial intelligence. Published author and conference speaker.",
        "Elementary school teacher specialized in early childhood education and literacy development. Experience with diverse learning needs and inclusive education.",
        "Academic advisor with expertise in student counseling, course planning, and career guidance. Strong background in higher education administration.",
        "Educational technology specialist with experience in online learning platforms, instructional design, and educational software integration.",
        "Curriculum developer with expertise in K-12 education standards, assessment design, and instructional materials development.",
        "Research scientist in education with focus on learning analytics and educational data mining. PhD in Education with strong research background.",
        "Special education teacher with expertise in working with students with disabilities. Skilled in individualized education plans and adaptive instruction.",
        "School administrator with experience in school leadership, budget management, and educational policy. Strong background in educational administration.",
        "Adult education instructor with expertise in workforce development and continuing education programs. Experience with adult learners and professional development."
    ]
    texts.extend(education_texts)
    labels.extend(['education'] * len(education_texts))
    
    # Marketing examples
    marketing_texts = [
        "Digital marketing manager with 7 years of experience in SEO, SEM, social media marketing, and content strategy. Strong analytical and creative skills.",
        "Brand manager with expertise in brand strategy, brand positioning, and brand management. Experience with consumer products and brand development.",
        "Social media manager specialized in content creation, social media strategy, and community management. Proficient in social media analytics and engagement.",
        "Marketing analyst with expertise in market research, consumer behavior analysis, and marketing analytics. Strong data analysis and reporting skills.",
        "Content marketing specialist with experience in content creation, content strategy, and content distribution. Skilled in copywriting and content optimization.",
        "Marketing communications manager with expertise in public relations, media relations, and corporate communications. Strong writing and communication skills.",
        "Product marketing manager with experience in product positioning, go-to-market strategy, and product launches. Strong analytical and strategic thinking.",
        "Email marketing specialist with expertise in email campaign development, email automation, and email analytics. Proficient in email marketing platforms.",
        "Growth marketing manager with experience in growth hacking, user acquisition, and conversion optimization. Strong analytical and experimentation skills.",
        "Marketing director with expertise in marketing strategy, team leadership, and marketing operations. Strong background in integrated marketing campaigns."
    ]
    texts.extend(marketing_texts)
    labels.extend(['marketing'] * len(marketing_texts))
    
    return texts, labels


def train_model():
    """Train the ML domain classifier"""
    print("🚀 Starting ML Domain Classifier training...")
    
    # Initialize model
    model_path = os.path.join("model", "domain_classifier.pkl")
    classifier = MLDomainClassifier(model_path=model_path, use_bert=False)  # Set use_bert=True if you have transformers
    
    if not classifier.rf_classifier:
        print("❌ Error: ML classifier not initialized. Install scikit-learn: pip install scikit-learn")
        return
    
    # Prepare training data
    print("📊 Preparing training data...")
    texts, labels = prepare_training_data()
    
    if len(texts) < 10:
        print("⚠️ Warning: Training data is very small. For best results, collect at least 100+ examples per domain.")
        print("   The model will still train, but may not generalize well.")
    
    print(f"📈 Training on {len(texts)} examples across {len(set(labels))} domains...")
    
    # Show distribution
    from collections import Counter
    label_counts = Counter(labels)
    print("📊 Domain distribution:")
    for domain, count in label_counts.items():
        print(f"   {domain}: {count} examples")
    
    # Train model
    try:
        classifier.train(
            texts=texts,
            labels=labels,
            test_size=0.2  # 20% for testing
        )
        
        print("✅ Model training completed!")
        
        # Save model
        classifier.save_model(model_path)
        print(f"💾 Model saved to {model_path}")
        
        # Show feature importance
        importance = classifier.get_feature_importance()
        if importance:
            print("\n📊 Top 10 Most Important Features:")
            for i, (feature, imp) in enumerate(list(importance.items())[:10]):
                print(f"   {i+1}. {feature}: {imp:.4f}")
        
        # Test on some examples
        print("\n🧪 Testing classifier on sample texts...")
        test_texts = [
            "Python developer with AWS experience",
            "Registered Nurse with ICU experience",
            "Financial analyst with CFA certification",
            "High school mathematics teacher",
            "Digital marketing manager with SEO expertise"
        ]
        
        for text in test_texts:
            domain, confidence = classifier.classify_domain(text)
            print(f"   '{text[:50]}...' -> {domain} (confidence: {confidence:.3f})")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()


def load_and_test_model():
    """Load trained model and test it"""
    model_path = os.path.join("model", "domain_classifier.pkl")
    
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        print("   Please train the model first using train_model()")
        return
    
    print("📥 Loading trained model...")
    classifier = MLDomainClassifier(model_path=model_path)
    
    if not classifier.is_trained:
        print("❌ Model is not trained")
        return
    
    # Test query
    test_texts = [
        "Senior Python developer with AWS and Docker experience",
        "Registered Nurse with 5 years of ICU experience",
        "Financial analyst with CFA certification and investment experience",
        "High school teacher with 10 years of mathematics instruction",
        "Digital marketing manager with SEO and social media expertise",
        "Unknown profession with mixed skills"
    ]
    
    print("\n🔍 Testing Domain Classification:")
    print("-" * 60)
    for text in test_texts:
        domain, confidence = classifier.classify_domain(text)
        print(f"Text: {text[:50]}...")
        print(f"Domain: {domain} (confidence: {confidence:.3f})")
        print("-" * 60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train or test ML Domain Classifier")
    parser.add_argument("--mode", choices=["train", "test"], default="train",
                       help="Mode: train or test the model")
    
    args = parser.parse_args()
    
    if args.mode == "train":
        train_model()
    else:
        load_and_test_model()

