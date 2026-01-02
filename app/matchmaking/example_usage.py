"""
Example usage of the matchmaking system.
Demonstrates how to use the system to match candidates to jobs.
"""

import json
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.matchmaking.pipelines.matcher import match_candidates, CandidateJobMatcher

# Example job description
job_description = """
We are looking for a Senior Python Developer with the following requirements:

Required Skills:
- Python (5+ years experience)
- Django or Flask framework
- PostgreSQL or MySQL
- REST API development
- Git version control

Preferred Skills:
- AWS cloud services
- Docker containerization
- CI/CD pipelines

Experience: Minimum 5 years of software development experience
Education: Bachelor's degree in Computer Science or related field
"""

# Example candidates
candidates = [
    {
        "candidate_id": "C1",
        "name": "John Doe",
        "email": "john.doe@example.com",
        "resume_text": """
        Senior Software Engineer with 7 years of experience in Python development.
        
        Skills: Python, Django, Flask, PostgreSQL, MySQL, REST APIs, Git, AWS, Docker
        
        Experience:
        - 7 years of Python development
        - Built scalable web applications using Django
        - Deployed applications on AWS
        - Used Docker for containerization
        
        Education: Bachelor's degree in Computer Science
        Certifications: AWS Certified Developer
        """,
        "skills": ["Python", "Django", "PostgreSQL", "AWS", "Docker", "Git"],
        "experience": "7 years",
        "education": "Bachelor's in Computer Science"
    },
    {
        "candidate_id": "C2",
        "name": "Jane Smith",
        "email": "jane.smith@example.com",
        "resume_text": """
        Software Developer with 3 years of experience in web development.
        
        Skills: Python, Flask, MongoDB, JavaScript, React
        
        Experience:
        - 3 years of Python development
        - Built REST APIs using Flask
        - Frontend development with React
        
        Education: Bachelor's degree in Software Engineering
        """,
        "skills": ["Python", "Flask", "MongoDB", "JavaScript", "React"],
        "experience": "3 years",
        "education": "Bachelor's in Software Engineering"
    },
    {
        "candidate_id": "C3",
        "name": "Bob Johnson",
        "email": "bob.johnson@example.com",
        "resume_text": """
        Full Stack Developer with 6 years of experience.
        
        Skills: Java, Spring Boot, MySQL, JavaScript, Angular, Azure
        
        Experience:
        - 6 years of Java development
        - Spring Boot framework
        - Azure cloud deployment
        
        Education: Master's degree in Computer Science
        """,
        "skills": ["Java", "Spring Boot", "MySQL", "JavaScript", "Angular", "Azure"],
        "experience": "6 years",
        "education": "Master's in Computer Science"
    }
]


def main():
    """Run example matchmaking."""
    print("=" * 60)
    print("Candidate-Job Matchmaking System - Example Usage")
    print("=" * 60)
    print()
    
    print("Job Description:")
    print(job_description)
    print()
    print("=" * 60)
    print()
    
    # Run matching
    print("Matching candidates to job...")
    print()
    
    results = match_candidates(job_description, candidates, top_k=10)
    
    # Display results
    print(f"Found {len(results)} matches:")
    print()
    
    for i, result in enumerate(results, 1):
        print(f"Rank {i}: Candidate {result['candidate_id']}")
        print(f"  Score: {result['score']:.2%}")
        print(f"  Matched Skills: {', '.join(result['matched_skills'][:5])}")
        if result['missing_skills']:
            print(f"  Missing Skills: {', '.join(result['missing_skills'][:5])}")
        print(f"  Score Breakdown:")
        print(f"    - Skill Match: {result['details']['skill_score']:.2%}")
        print(f"    - Experience: {result['details']['experience_score']:.2%}")
        print(f"    - Semantic: {result['details']['semantic_score']:.2%}")
        print(f"    - Additional: {result['details']['additional_score']:.2%}")
        print(f"  Explanation: {result['explanation']}")
        print()
    
    # Display as JSON
    print("=" * 60)
    print("Results as JSON:")
    print("=" * 60)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

