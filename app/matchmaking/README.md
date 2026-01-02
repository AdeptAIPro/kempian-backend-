# Production-Ready Candidate-Job Matchmaking System

A complete, production-ready AI-based job-candidate matching engine with modular architecture, skill ontology matching, semantic embeddings, and explainable multi-factor scoring.

## Features

- **Resume/Text Normalization**: Cleans and normalizes resume and job description text
- **Job JD Parsing & Normalization**: Extracts structured requirements from job descriptions
- **Skill Extraction**: Uses comprehensive skill ontology with aliases, synonyms, and fuzzy matching
- **SentenceTransformer Embeddings**: Semantic similarity using `all-MiniLM-L6-v2` model
- **Multi-Factor Weighted Scoring**: 
  - 40% Skill Match
  - 20% Experience Fit
  - 30% Semantic Embedding Score
  - 10% Additional factors (education, certifications)
- **Ranking with Explainability**: Returns detailed score breakdowns and human-readable explanations
- **Modular Architecture**: Clean, maintainable, and extensible code structure
- **Evaluation Metrics**: Precision, Recall, F1, NDCG, and MAP metrics

## Installation

### Prerequisites

```bash
pip install sentence-transformers numpy
```

### Optional Dependencies

For enhanced performance:
```bash
pip install torch  # For GPU acceleration (optional)
```

## Quick Start

### Basic Usage

```python
from app.matchmaking import match_candidates

# Job description
job_description = """
We are looking for a Python Developer with Django experience.
Required: Python, Django, PostgreSQL
Experience: 3+ years
"""

# Candidates
candidates = [
    {
        "candidate_id": "C1",
        "name": "John Doe",
        "resume_text": "Python developer with 5 years of Django experience...",
        "skills": ["Python", "Django", "PostgreSQL"],
        "experience": "5 years"
    }
]

# Match candidates
results = match_candidates(job_description, candidates)

# Results are sorted by score (highest first)
for result in results:
    print(f"Candidate {result['candidate_id']}: {result['score']:.2%}")
    print(f"Matched Skills: {result['matched_skills']}")
    print(f"Missing Skills: {result['missing_skills']}")
```

### Advanced Usage

```python
from app.matchmaking.pipelines.matcher import CandidateJobMatcher

# Create matcher instance
matcher = CandidateJobMatcher()

# Match with custom top_k
results = matcher.match_candidates(
    job_description=job_description,
    candidates=candidates,
    top_k=10
)

# Access detailed results
for result in results:
    print(result.candidate_id)
    print(result.score)
    print(result.details.skill_score)
    print(result.explanation)
```

## Architecture

```
matchmaking/
├── data/
│   └── skill_ontology.json      # Skill ontology with aliases
├── collectors/
│   ├── resume_parser.py         # Resume parsing and extraction
│   ├── job_parser.py            # Job description parsing
│   └── skill_extractor.py       # Skill extraction using ontology
├── embeddings/
│   └── embedder.py              # SentenceTransformer wrapper
├── scoring/
│   └── scorer.py                # Multi-factor scoring engine
├── pipelines/
│   └── matcher.py               # Main pipeline orchestrator
├── evaluation/
│   ├── metrics.py                # Evaluation metrics
│   └── benchmark_runner.py     # Benchmark testing
└── utils/
    ├── text_cleaner.py          # Text normalization
    └── similarity.py             # Similarity calculations
```

## Components

### Skill Extractor

Extracts and canonicalizes skills from text using a comprehensive ontology:

```python
from app.matchmaking.collectors.skill_extractor import SkillExtractor

extractor = SkillExtractor()
skills = extractor.extract_skills("I have experience with Python, Django, and PostgreSQL")
# Returns: [ExtractedSkill(skill_id='python', ...), ...]
```

### Resume Parser

Parses candidate data into structured format:

```python
from app.matchmaking.collectors.resume_parser import ResumeParser

parser = ResumeParser()
resume_data = parser.parse(candidate_dict)
# Returns: ResumeData with skills, experience, education, etc.
```

### Job Parser

Parses job descriptions into structured requirements:

```python
from app.matchmaking.collectors.job_parser import JobParser

parser = JobParser()
job_data = parser.parse(job_dict)
# Returns: JobData with required_skills, preferred_skills, experience_required, etc.
```

### Scorer

Calculates multi-factor match scores:

```python
from app.matchmaking.scoring.scorer import Scorer

scorer = Scorer()
match_score = scorer.calculate_score(resume_data, job_data)
# Returns: MatchScore with skill_score, experience_score, semantic_score, etc.
```

## Scoring Formula

The overall match score is calculated as:

```
overall_score = (
    skill_score * 0.40 +
    experience_score * 0.20 +
    semantic_score * 0.30 +
    additional_score * 0.10
)
```

Where:
- **skill_score**: Ratio of matched required/preferred skills
- **experience_score**: min(1.0, candidate_years / required_years)
- **semantic_score**: Cosine similarity of embeddings
- **additional_score**: Education match, certifications, etc.

## Output Format

Results are returned as a list of dictionaries:

```json
[
  {
    "candidate_id": "C1",
    "score": 0.83,
    "matched_skills": ["Python", "Django", "PostgreSQL"],
    "missing_skills": ["AWS"],
    "details": {
      "skill_score": 0.80,
      "experience_score": 1.0,
      "semantic_score": 0.78,
      "additional_score": 0.75
    },
    "explanation": "Overall match score: 83%. Matched 3 required/preferred skills..."
  }
]
```

## Evaluation

### Running Benchmarks

```python
from app.matchmaking.evaluation.benchmark_runner import BenchmarkRunner

runner = BenchmarkRunner()

test_cases = [
    {
        "job_description": "...",
        "candidates": [...],
        "expected_top_candidates": ["C1", "C2"]
    }
]

results = runner.run_benchmark(test_cases)
print(f"Average Precision: {results['average_metrics']['precision']}")
print(f"MAP: {results['map']}")
```

### Calculating Metrics

```python
from app.matchmaking.evaluation.metrics import calculate_metrics

predicted = ["C1", "C2", "C3"]
relevant = ["C1", "C3"]

metrics = calculate_metrics(predicted, relevant, k=10)
print(f"Precision: {metrics['precision']}")
print(f"Recall: {metrics['recall']}")
print(f"NDCG: {metrics['ndcg']}")
```

## Skill Ontology

The system uses a JSON-based skill ontology located at `data/skill_ontology.json`. Each skill entry includes:

- `skill_id`: Unique identifier
- `canonical_name`: Standardized skill name
- `category`: Skill category (frontend, backend, cloud, etc.)
- `level`: Skill level (language, framework, tool, etc.)
- `aliases`: Alternative names and abbreviations
- `synonyms`: Related terms
- `parent_skill_id`: Parent skill in hierarchy

To add new skills, edit the JSON file and the system will automatically load them.

## Performance Considerations

- **Lazy Loading**: Embedding model loads on first use
- **Batch Processing**: Supports batch encoding for multiple candidates
- **Caching**: Embeddings can be pre-computed and reused
- **Efficient Matching**: Skill matching uses hash maps for O(1) lookup

## Error Handling

The system gracefully handles:
- Missing or invalid input data
- Unavailable embedding models (falls back to TF-IDF)
- Missing skills in ontology (keeps original skill names)
- Parsing errors (returns minimal valid structures)

## Integration with FastAPI

```python
from fastapi import FastAPI
from app.matchmaking import match_candidates

app = FastAPI()

@app.post("/match")
async def match_candidates_endpoint(job_description: str, candidates: list):
    results = match_candidates(job_description, candidates)
    return {"results": results}
```

## Extending the System

### Adding New Scoring Factors

1. Modify `scoring/scorer.py` to add new factor calculation
2. Update weight distribution in `Scorer` class
3. Add factor to `MatchScore` dataclass

### Adding New Skill Categories

1. Edit `data/skill_ontology.json`
2. Add skills with appropriate category
3. System automatically loads new skills

### Custom Embedding Models

```python
from app.matchmaking.embeddings.embedder import Embedder

# Use custom model
embedder = Embedder(model_name="your-model-name")
```

## Testing

Run the example usage script:

```bash
python -m app.matchmaking.example_usage
```

## License

This system is part of the Kempian recruitment platform.

## Support

For issues or questions, please refer to the main project documentation.

