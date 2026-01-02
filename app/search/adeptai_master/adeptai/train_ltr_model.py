"""
Training script for Learning-to-Rank model

This script demonstrates how to train the LTR model using feedback data.
You can collect feedback data from:
- User clicks (clicked candidates vs. not clicked)
- Hire/reject decisions
- Interview outcomes
- User ratings
"""

import os
import sys
from typing import List, Tuple, Dict, Any

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from learning_to_rank import LearningToRankModel, LTRFeatures


def prepare_training_data() -> Tuple[List[Tuple[str, Dict[str, Any]]], List[float], List[Dict[str, float]]]:
    """
    Prepare training data from feedback.
    
    In a real scenario, you would:
    1. Load user interaction data (clicks, hires, rejects)
    2. Extract query-candidate pairs
    3. Assign relevance labels (0-4 scale):
       - 4: Perfect match (hired, highly rated)
       - 3: Good match (interviewed, clicked)
       - 2: Fair match (viewed, considered)
       - 1: Poor match (not clicked, rejected quickly)
       - 0: Irrelevant (not viewed, filtered out)
    4. Extract feature scores for each pair
    
    Returns:
        Tuple of (query_candidate_pairs, labels, feature_scores)
    """
    # Example training data
    # In production, load this from your database/logs
    
    query_candidate_pairs = []
    labels = []
    feature_scores = []
    
    # Example 1: Good match - Python developer with AWS experience
    query1 = "Senior Python developer with AWS"
    candidate1 = {
        'email': 'john.doe@example.com',
        'full_name': 'John Doe',
        'skills': ['Python', 'AWS', 'Docker', 'Kubernetes', 'Django'],
        'total_experience_years': 8,
        'resume_text': 'Senior Python developer with 8 years of experience in AWS cloud services...'
    }
    query_candidate_pairs.append((query1, candidate1))
    labels.append(4.0)  # High relevance - matches perfectly
    feature_scores.append({
        'keyword_score': 0.85,
        'semantic_score': 0.90,
        'cross_encoder_score': 0.88
    })
    
    # Example 2: Fair match - Python developer but less AWS experience
    candidate2 = {
        'email': 'jane.smith@example.com',
        'full_name': 'Jane Smith',
        'skills': ['Python', 'Django', 'PostgreSQL'],
        'total_experience_years': 5,
        'resume_text': 'Python developer with 5 years of experience in web development...'
    }
    query_candidate_pairs.append((query1, candidate2))
    labels.append(2.5)  # Medium relevance - has Python but lacks AWS
    feature_scores.append({
        'keyword_score': 0.60,
        'semantic_score': 0.65,
        'cross_encoder_score': 0.55
    })
    
    # Example 3: Poor match - Java developer instead of Python
    candidate3 = {
        'email': 'bob.jones@example.com',
        'full_name': 'Bob Jones',
        'skills': ['Java', 'Spring', 'AWS'],
        'total_experience_years': 7,
        'resume_text': 'Senior Java developer with AWS experience...'
    }
    query_candidate_pairs.append((query1, candidate3))
    labels.append(1.0)  # Low relevance - wrong language
    feature_scores.append({
        'keyword_score': 0.40,
        'semantic_score': 0.45,
        'cross_encoder_score': 0.35
    })
    
    # Add more examples here...
    # In production, you would have thousands of examples
    
    return query_candidate_pairs, labels, feature_scores


def train_model():
    """Train the Learning-to-Rank model"""
    print("🚀 Starting Learning-to-Rank model training...")
    
    # Initialize model
    model_path = os.path.join("model", "ltr_model.pkl")
    model = LearningToRankModel(model_path=model_path)
    
    if not model.model:
        print("❌ Error: LTR model not initialized. Install lightgbm: pip install lightgbm")
        return
    
    # Prepare training data
    print("📊 Preparing training data...")
    query_candidate_pairs, labels, feature_scores = prepare_training_data()
    
    if len(query_candidate_pairs) < 10:
        print("⚠️ Warning: Training data is very small. For best results, collect at least 100+ examples.")
        print("   The model will still train, but may not generalize well.")
    
    print(f"📈 Training on {len(query_candidate_pairs)} examples...")
    
    # Generate query groups (each query may have multiple candidates)
    query_groups = []
    current_group = 0
    prev_query = None
    for query, _ in query_candidate_pairs:
        if query != prev_query:
            current_group += 1
            prev_query = query
        query_groups.append(current_group)
    
    # Train model
    try:
        model.train(
            query_candidate_pairs=query_candidate_pairs,
            labels=labels,
            query_groups=query_groups,
            feature_scores=feature_scores
        )
        
        print("✅ Model training completed!")
        
        # Save model
        model.save_model(model_path)
        print(f"💾 Model saved to {model_path}")
        
        # Show feature importance
        importance = model.get_feature_importance()
        if importance:
            print("\n📊 Feature Importance:")
            for feature, imp in sorted(importance.items(), key=lambda x: x[1], reverse=True):
                print(f"   {feature}: {imp:.4f}")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()


def load_and_test_model():
    """Load trained model and test it"""
    model_path = os.path.join("model", "ltr_model.pkl")
    
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        print("   Please train the model first using train_model()")
        return
    
    print("📥 Loading trained model...")
    model = LearningToRankModel(model_path=model_path)
    
    if not model.is_trained:
        print("❌ Model is not trained")
        return
    
    # Test query
    query = "Senior Python developer with AWS"
    candidates = [
        {
            'email': 'test1@example.com',
            'full_name': 'Test Candidate 1',
            'skills': ['Python', 'AWS', 'Docker'],
            'total_experience_years': 8,
            'resume_text': 'Senior Python developer with AWS experience'
        },
        {
            'email': 'test2@example.com',
            'full_name': 'Test Candidate 2',
            'skills': ['Python', 'Django'],
            'total_experience_years': 5,
            'resume_text': 'Python developer with web development experience'
        }
    ]
    
    feature_scores = [
        {'keyword_score': 0.85, 'semantic_score': 0.90, 'cross_encoder_score': 0.88},
        {'keyword_score': 0.60, 'semantic_score': 0.65, 'cross_encoder_score': 0.55}
    ]
    
    print(f"\n🔍 Testing query: '{query}'")
    scores = model.predict(query, candidates, feature_scores=feature_scores)
    
    print("\n📊 Predicted Scores:")
    for i, (candidate, score) in enumerate(zip(candidates, scores)):
        print(f"   {i+1}. {candidate['full_name']}: {score:.4f}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train or test Learning-to-Rank model")
    parser.add_argument("--mode", choices=["train", "test"], default="train",
                       help="Mode: train or test the model")
    
    args = parser.parse_args()
    
    if args.mode == "train":
        train_model()
    else:
        load_and_test_model()

