"""
Training Data Preparation Scripts for SageMaker

This module handles preparation of training data for SageMaker training jobs,
including data extraction, formatting, and validation.
"""

import os
import json
import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime
import logging

# Import custom LLM models
from custom_llm_models import CustomTokenizer, CustomEmbeddingModel
from search_system import get_candidates_with_fallback

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrainingDataPreparer:
    """Prepare training data for SageMaker training jobs"""
    
    def __init__(self, output_dir: str = './training_data'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        self.training_dir = self.output_dir / 'training'
        self.validation_dir = self.output_dir / 'validation'
        self.training_dir.mkdir(exist_ok=True)
        self.validation_dir.mkdir(exist_ok=True)
    
    def extract_candidate_texts(self, candidates: List[Dict[str, Any]]) -> List[str]:
        """Extract text content from candidates for training"""
        texts = []
        
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            
            # Combine all text fields
            text_parts = []
            
            # Add skills
            skills = candidate.get('skills', [])
            if skills:
                text_parts.append(' '.join(skills))
            
            # Add resume text
            resume_text = candidate.get('resume_text', '')
            if resume_text:
                text_parts.append(resume_text)
            
            # Add full name
            full_name = candidate.get('full_name', '')
            if full_name:
                text_parts.append(full_name)
            
            # Combine and clean
            combined_text = ' '.join(text_parts).strip()
            if combined_text:
                texts.append(combined_text)
        
        return texts
    
    def create_similarity_pairs(self, texts: List[str], num_pairs: int = 1000) -> List[Tuple[str, str, float]]:
        """Create similarity pairs for training"""
        pairs = []
        
        # Use simple tokenizer for similarity calculation
        tokenizer = CustomTokenizer()
        
        for i in range(min(num_pairs, len(texts) * 2)):
            # Randomly select two texts
            idx1, idx2 = np.random.choice(len(texts), 2, replace=False)
            text1, text2 = texts[idx1], texts[idx2]
            
            # Calculate similarity using token overlap
            tokens1 = set(tokenizer.tokenize(text1))
            tokens2 = set(tokenizer.tokenize(text2))
            
            if tokens1 and tokens2:
                intersection = len(tokens1.intersection(tokens2))
                union = len(tokens1.union(tokens2))
                similarity = intersection / union if union > 0 else 0.0
                
                pairs.append((text1, text2, similarity))
        
        return pairs
    
    def split_data(self, texts: List[str], train_ratio: float = 0.8) -> Tuple[List[str], List[str]]:
        """Split data into training and validation sets"""
        np.random.shuffle(texts)
        
        split_idx = int(len(texts) * train_ratio)
        train_texts = texts[:split_idx]
        val_texts = texts[split_idx:]
        
        return train_texts, val_texts
    
    def save_training_data(self, texts: List[str], filename: str = 'training_texts.txt'):
        """Save training texts to file"""
        file_path = self.training_dir / filename
        
        with open(file_path, 'w', encoding='utf-8') as f:
            for text in texts:
                f.write(text + '\n')
        
        logger.info(f"Saved {len(texts)} training texts to {file_path}")
    
    def save_validation_data(self, texts: List[str], filename: str = 'validation_texts.txt'):
        """Save validation texts to file"""
        file_path = self.validation_dir / filename
        
        with open(file_path, 'w', encoding='utf-8') as f:
            for text in texts:
                f.write(text + '\n')
        
        logger.info(f"Saved {len(texts)} validation texts to {file_path}")
    
    def save_similarity_pairs(self, pairs: List[Tuple[str, str, float]], filename: str = 'similarity_pairs.json'):
        """Save similarity pairs to JSON file"""
        file_path = self.training_dir / filename
        
        data = []
        for text1, text2, similarity in pairs:
            data.append({
                'text1': text1,
                'text2': text2,
                'similarity': similarity
            })
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved {len(pairs)} similarity pairs to {file_path}")
    
    def create_query_candidate_pairs(self, candidates: List[Dict[str, Any]], num_pairs: int = 500) -> List[Dict[str, Any]]:
        """Create query-candidate pairs for training"""
        pairs = []
        
        # Generate synthetic queries based on candidate data
        query_templates = [
            "{skill} developer",
            "Senior {skill} engineer",
            "{skill} specialist",
            "{domain} professional",
            "Experienced {skill} developer"
        ]
        
        skills = set()
        domains = {'technology', 'healthcare', 'finance', 'education', 'marketing'}
        
        # Extract skills from candidates
        for candidate in candidates:
            candidate_skills = candidate.get('skills', [])
            skills.update(skill.lower() for skill in candidate_skills)
        
        # Create query-candidate pairs
        for i in range(min(num_pairs, len(candidates) * 2)):
            candidate = np.random.choice(candidates)
            skill = np.random.choice(list(skills)) if skills else 'developer'
            domain = np.random.choice(list(domains))
            
            # Generate query
            template = np.random.choice(query_templates)
            query = template.format(skill=skill, domain=domain)
            
            # Create pair
            pair = {
                'query': query,
                'candidate': candidate,
                'relevance_score': np.random.uniform(0.3, 1.0)  # Random relevance for now
            }
            
            pairs.append(pair)
        
        return pairs
    
    def save_query_candidate_pairs(self, pairs: List[Dict[str, Any]], filename: str = 'query_candidate_pairs.json'):
        """Save query-candidate pairs to JSON file"""
        file_path = self.training_dir / filename
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(pairs, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved {len(pairs)} query-candidate pairs to {file_path}")
    
    def prepare_all_training_data(self, candidates: Optional[List[Dict[str, Any]]] = None):
        """Prepare all training data files"""
        logger.info("Starting training data preparation...")
        
        # Load candidates if not provided
        if candidates is None:
            logger.info("Loading candidates from database...")
            candidates = get_candidates_with_fallback()
        
        logger.info(f"Loaded {len(candidates)} candidates")
        
        # Extract texts
        texts = self.extract_candidate_texts(candidates)
        logger.info(f"Extracted {len(texts)} text samples")
        
        # Split into train/validation
        train_texts, val_texts = self.split_data(texts)
        logger.info(f"Split: {len(train_texts)} training, {len(val_texts)} validation")
        
        # Save text data
        self.save_training_data(train_texts)
        self.save_validation_data(val_texts)
        
        # Create and save similarity pairs
        logger.info("Creating similarity pairs...")
        similarity_pairs = self.create_similarity_pairs(train_texts, num_pairs=1000)
        self.save_similarity_pairs(similarity_pairs)
        
        # Create and save query-candidate pairs
        logger.info("Creating query-candidate pairs...")
        query_candidate_pairs = self.create_query_candidate_pairs(candidates, num_pairs=500)
        self.save_query_candidate_pairs(query_candidate_pairs)
        
        # Create metadata
        metadata = {
            'created_at': datetime.now().isoformat(),
            'total_candidates': len(candidates),
            'total_texts': len(texts),
            'training_texts': len(train_texts),
            'validation_texts': len(val_texts),
            'similarity_pairs': len(similarity_pairs),
            'query_candidate_pairs': len(query_candidate_pairs),
            'output_directory': str(self.output_dir)
        }
        
        metadata_path = self.output_dir / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Training data preparation completed!")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Metadata saved to: {metadata_path}")
        
        return metadata


class TrainingDataValidator:
    """Validate training data quality"""
    
    @staticmethod
    def validate_text_quality(texts: List[str]) -> Dict[str, Any]:
        """Validate quality of text data"""
        if not texts:
            return {'error': 'No texts provided'}
        
        stats = {
            'total_texts': len(texts),
            'avg_length': np.mean([len(text) for text in texts]),
            'min_length': min(len(text) for text in texts),
            'max_length': max(len(text) for text in texts),
            'empty_texts': sum(1 for text in texts if not text.strip()),
            'very_short_texts': sum(1 for text in texts if len(text.strip()) < 10),
            'very_long_texts': sum(1 for text in texts if len(text.strip()) > 1000)
        }
        
        # Quality checks
        quality_issues = []
        
        if stats['empty_texts'] > 0:
            quality_issues.append(f"{stats['empty_texts']} empty texts found")
        
        if stats['very_short_texts'] > len(texts) * 0.1:
            quality_issues.append(f"Too many very short texts: {stats['very_short_texts']}")
        
        if stats['very_long_texts'] > len(texts) * 0.05:
            quality_issues.append(f"Too many very long texts: {stats['very_long_texts']}")
        
        stats['quality_issues'] = quality_issues
        stats['quality_score'] = max(0, 1.0 - len(quality_issues) * 0.2)
        
        return stats
    
    @staticmethod
    def validate_similarity_pairs(pairs: List[Tuple[str, str, float]]) -> Dict[str, Any]:
        """Validate similarity pairs"""
        if not pairs:
            return {'error': 'No pairs provided'}
        
        similarities = [pair[2] for pair in pairs]
        
        stats = {
            'total_pairs': len(pairs),
            'avg_similarity': np.mean(similarities),
            'min_similarity': min(similarities),
            'max_similarity': max(similarities),
            'std_similarity': np.std(similarities),
            'high_similarity_pairs': sum(1 for sim in similarities if sim > 0.8),
            'low_similarity_pairs': sum(1 for sim in similarities if sim < 0.2)
        }
        
        return stats


def main():
    """Main function for training data preparation"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Prepare training data for SageMaker')
    parser.add_argument('--output-dir', default='./training_data', help='Output directory')
    parser.add_argument('--validate', action='store_true', help='Validate data quality')
    parser.add_argument('--num-pairs', type=int, default=1000, help='Number of similarity pairs to create')
    
    args = parser.parse_args()
    
    # Prepare training data
    preparer = TrainingDataPreparer(args.output_dir)
    metadata = preparer.prepare_all_training_data()
    
    # Validate if requested
    if args.validate:
        logger.info("Validating training data quality...")
        
        # Load and validate texts
        train_texts = []
        train_file = preparer.training_dir / 'training_texts.txt'
        if train_file.exists():
            with open(train_file, 'r', encoding='utf-8') as f:
                train_texts = [line.strip() for line in f if line.strip()]
        
        val_texts = []
        val_file = preparer.validation_dir / 'validation_texts.txt'
        if val_file.exists():
            with open(val_file, 'r', encoding='utf-8') as f:
                val_texts = [line.strip() for line in f if line.strip()]
        
        # Validate training texts
        train_stats = TrainingDataValidator.validate_text_quality(train_texts)
        logger.info(f"Training text quality: {train_stats}")
        
        # Validate validation texts
        val_stats = TrainingDataValidator.validate_text_quality(val_texts)
        logger.info(f"Validation text quality: {val_stats}")
        
        # Validate similarity pairs
        pairs_file = preparer.training_dir / 'similarity_pairs.json'
        if pairs_file.exists():
            with open(pairs_file, 'r') as f:
                pairs_data = json.load(f)
                pairs = [(item['text1'], item['text2'], item['similarity']) for item in pairs_data]
                pair_stats = TrainingDataValidator.validate_similarity_pairs(pairs)
                logger.info(f"Similarity pairs quality: {pair_stats}")
    
    logger.info("Training data preparation completed successfully!")


if __name__ == '__main__':
    main()
