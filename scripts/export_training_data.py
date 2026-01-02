"""
Export Training Data from Auto-Evolution System
Prepares data for SageMaker training
"""

import json
import os
from app.llm.auto_evolution import AutoEvolutionSystem

def export_training_data():
    """Export high-quality training data"""
    
    print("=" * 60)
    print("Exporting Training Data")
    print("=" * 60)
    
    # Initialize evolution system
    evolution = AutoEvolutionSystem()
    
    # Get statistics
    stats = evolution.get_statistics()
    print(f"\nTotal interactions: {stats['total_interactions']}")
    print(f"High quality interactions: {stats['high_quality_interactions']}")
    print(f"Recent (7 days): {stats['recent_interactions_7d']}")
    print(f"\nBy task type:")
    for task_type, data in stats['by_task_type'].items():
        print(f"  {task_type}: {data['count']} examples (avg quality: {data['avg_quality']:.2f})")
    
    # Get training data
    print("\n" + "=" * 60)
    print("Collecting training data...")
    print("=" * 60)
    
    training_data = evolution.get_training_data(
        min_quality=0.7,
        limit=2000
    )
    
    print(f"Found {len(training_data)} high-quality examples")
    
    if len(training_data) < 100:
        print("\n⚠️  Warning: Less than 100 examples found.")
        print("   Consider:")
        print("   1. Using the system more to collect data")
        print("   2. Lowering min_quality threshold")
        print("   3. Adding manual training examples")
    
    # Format for SageMaker
    formatted = []
    for item in training_data:
        task_type = item["task_type"]
        input_text = item["input"]
        output_text = item["output"]
        
        # Determine instruction based on task type
        if task_type == "job_extraction":
            instruction = "Extract structured job data from the following requirement:"
        elif task_type == "candidate_analysis":
            instruction = "Analyze this candidate based on the provided context:"
        else:
            instruction = "Answer the following question:"
        
        formatted.append({
            "instruction": instruction,
            "input": input_text,
            "output": output_text
        })
    
    # Save to file
    output_file = "backend/data/sagemaker_training/training_data.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(formatted, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Exported {len(formatted)} training examples")
    print(f"   Saved to: {output_file}")
    print()
    print("Next steps:")
    print(f"1. Upload to S3:")
    print(f"   aws s3 cp {output_file} s3://YOUR-BUCKET/training-data/")
    print()
    print("2. Set environment variable:")
    print(f"   export TRAINING_DATA_S3=s3://YOUR-BUCKET/training-data/training_data.json")
    print()
    print("3. Start training:")
    print("   python scripts/sagemaker_train.py")
    
    return output_file

if __name__ == "__main__":
    export_training_data()

