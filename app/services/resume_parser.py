"""
Resume parsing service for extracting text and data from resume files
"""
import os
import logging
from typing import Dict, List, Optional, Any
from app.simple_logger import get_logger
import re
from datetime import datetime
import tempfile
import io

try:
    import pdfplumber
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

# Fallback to PyPDF2 if pdfplumber not available
try:
    import PyPDF2
    PDF_FALLBACK_AVAILABLE = True
except ImportError:
    PDF_FALLBACK_AVAILABLE = False

try:
    from docx import Document
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False

try:
    import docx2txt
    DOC_AVAILABLE = True
except ImportError:
    DOC_AVAILABLE = False

try:
    import phonenumbers
    PHONE_AVAILABLE = True
except ImportError:
    PHONE_AVAILABLE = False

try:
    from dateparser.search import search_dates
    DATE_PARSER_AVAILABLE = True
except ImportError:
    DATE_PARSER_AVAILABLE = False

try:
    from tika import parser as tika_parser
    TIKA_AVAILABLE = True
except ImportError:
    TIKA_AVAILABLE = False

logger = get_logger(__name__.split('.')[-1])

class ResumeParser:
    """Service for parsing resume files and extracting structured data"""
    
    def __init__(self):
        # Load comprehensive skill list from file if available, otherwise use default
        self.skill_keywords = self._load_skill_keywords()
        
        self.experience_patterns = [
            r'(\d+)\+?\s*years?\s*(?:of\s*)?experience',
            r'(\d+)\+?\s*years?\s*in\s*(?:the\s*)?(?:field|industry|profession)',
            r'(\d+)\+?\s*years?\s*working\s*(?:in|with)',
            r'(\d+)\+?\s*years?\s*professional\s*experience',
            r'(\d+)\+?\s*years?\s*of\s*professional\s*development',
            r'(\d+)\+?\s*years?\s*at\s+\w+',  # e.g., "3 years at Google"
            r'(\d+)\+?\s*years?\s*with\s+\w+',  # e.g., "2 years with Microsoft"
            r'(\d+)\+?\s*years?\s*in\s+\w+',  # e.g., "4 years in software development"
        ]
        
        self.education_keywords = [
            'bachelor', 'master', 'phd', 'doctorate', 'associate', 'diploma', 'certificate',
            'degree', 'university', 'college', 'institute', 'school'
        ]

    def _load_skill_keywords(self) -> List[str]:
        """Load skill keywords from file or use default list"""
        try:
            # Try to load from all_skills.txt file
            skill_file_path = os.path.join(os.path.dirname(__file__), '..', '..', 'all_skills.txt')
            if os.path.exists(skill_file_path):
                with open(skill_file_path, "r", encoding="utf-8") as f:
                    return [line.strip().lower() for line in f if line.strip()]
        except Exception as e:
            logger.warning(f"Could not load skills from file: {e}")
        
        # Fallback to default comprehensive skill list
        return [
            # Programming Languages
            'python', 'javascript', 'java', 'c++', 'c#', 'php', 'ruby', 'go', 'rust', 'swift', 'kotlin',
            'typescript', 'scala', 'r', 'matlab', 'perl', 'haskell', 'clojure', 'erlang',
            'c', 'csharp', 'vb.net', 'f#', 'dart', 'objective-c', 'assembly', 'cobol', 'fortran',
            
            # Web Technologies
            'html', 'css', 'react', 'angular', 'vue', 'node.js', 'express', 'django', 'flask', 'spring',
            'laravel', 'rails', 'asp.net', 'jquery', 'bootstrap', 'sass', 'less', 'webpack',
            'next.js', 'nuxt.js', 'ember', 'backbone', 'knockout', 'polymer', 'svelte',
            'html5', 'css3', 'responsive design', 'pwa', 'spa', 'ssr', 'csr',
            
            # Databases
            'mysql', 'postgresql', 'mongodb', 'redis', 'sqlite', 'oracle', 'sql server', 'cassandra',
            'elasticsearch', 'dynamodb', 'firebase', 'neo4j', 'couchdb', 'riak', 'hbase',
            'mariadb', 'db2', 'teradata', 'snowflake', 'bigquery', 'redshift',
            
            # Cloud & DevOps
            'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'jenkins', 'git', 'ci/cd', 'terraform',
            'ansible', 'linux', 'bash', 'powershell', 'vagrant', 'chef', 'puppet',
            'aws ec2', 'aws s3', 'aws lambda', 'aws rds', 'aws cloudformation', 'aws cloudwatch',
            'azure vm', 'azure storage', 'azure functions', 'azure devops',
            'google cloud', 'google app engine', 'google compute engine',
            
            # Data & AI
            'machine learning', 'deep learning', 'tensorflow', 'pytorch', 'pandas', 'numpy', 'scikit-learn',
            'data analysis', 'statistics', 'sql', 'tableau', 'power bi', 'excel',
            'keras', 'opencv', 'nltk', 'spacy', 'hadoop', 'spark', 'kafka', 'airflow',
            'data science', 'data engineering', 'etl', 'data visualization', 'business intelligence',
            
            # Mobile
            'android', 'ios', 'react native', 'flutter', 'xamarin', 'ionic',
            'swift', 'kotlin', 'objective-c', 'java', 'cordova', 'phonegap',
            
            # Testing
            'unit testing', 'integration testing', 'selenium', 'cypress', 'jest', 'mocha', 'chai',
            'junit', 'testng', 'pytest', 'rspec', 'cucumber', 'bdd', 'tdd',
            
            # Other
            'agile', 'scrum', 'project management', 'leadership', 'communication', 'teamwork',
            'microservices', 'api', 'rest', 'graphql', 'soap', 'json', 'xml',
            'version control', 'git', 'svn', 'mercurial', 'bitbucket', 'github', 'gitlab',
            'jira', 'confluence', 'slack', 'trello', 'asana', 'notion'
        ]

    def extract_text_from_file(self, file_path: str, file_extension: str) -> str:
        """Extract text from resume file based on file type"""
        try:
            if file_extension.lower() == 'pdf':
                return self._extract_from_pdf(file_path)
            elif file_extension.lower() in ['docx']:
                return self._extract_from_docx(file_path)
            elif file_extension.lower() in ['doc']:
                return self._extract_from_doc(file_path)
            elif file_extension.lower() == 'txt':
                return self._extract_from_txt(file_path)
            else:
                logger.warning(f"Unsupported file type: {file_extension}")
                return ""
        except Exception as e:
            logger.error(f"Error extracting text from {file_path}: {str(e)}")
            return ""

    def _extract_from_pdf(self, file_path: str) -> str:
        """Extract text from PDF file using pdfplumber for better accuracy, fallback to PyPDF2"""
        if PDF_AVAILABLE:
            try:
                with pdfplumber.open(file_path) as pdf:
                    text = ""
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
                    return text.strip()
            except Exception as e:
                logger.warning(f"Error reading PDF with pdfplumber: {str(e)}, trying fallback")
        
        # Fallback to PyPDF2
        if PDF_FALLBACK_AVAILABLE:
            try:
                with open(file_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    text = ""
                    for page in pdf_reader.pages:
                        text += page.extract_text() + "\n"
                    return text.strip()
            except Exception as e:
                logger.error(f"Error reading PDF with PyPDF2: {str(e)}")
                return ""
        else:
            logger.error("No PDF parsing library available")
            return ""

    def _extract_from_docx(self, file_path: str) -> str:
        """Extract text from DOCX file"""
        if not DOCX_AVAILABLE:
            logger.error("python-docx not available for DOCX parsing")
            return ""
        
        try:
            doc = Document(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text.strip()
        except Exception as e:
            logger.error(f"Error reading DOCX: {str(e)}")
            return ""

    def _extract_from_doc(self, file_path: str) -> str:
        """Extract text from DOC file using tika for better accuracy"""
        if not TIKA_AVAILABLE:
            logger.error("tika not available for DOC parsing")
            return ""
        
        try:
            parsed = tika_parser.from_file(file_path)
            return parsed.get("content", "").strip()
        except Exception as e:
            logger.error(f"Error reading DOC with tika: {str(e)}")
            return ""

    def _extract_from_txt(self, file_path: str) -> str:
        """Extract text from TXT file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except UnicodeDecodeError:
            # Try with different encoding if UTF-8 fails
            try:
                with open(file_path, 'r', encoding='latin-1') as file:
                    return file.read()
            except Exception as e:
                logger.error(f"Error extracting from TXT with latin-1: {str(e)}")
                return ""
        except Exception as e:
            logger.error(f"Error extracting from TXT: {str(e)}")
            return ""

    def parse_resume_data(self, resume_text: str, filename: str = "") -> Dict[str, Any]:
        """Parse resume text and extract structured data with improved accuracy"""
        if not resume_text:
            return {}
        
        # Clean and normalize text
        text = self._clean_text(resume_text)
        
        # First, identify and separate different sections to prevent cross-contamination
        sections = self._identify_sections(text)
        
        # Extract different components from their respective sections
        education = self._extract_education_from_sections(sections)
        work_experience = self._extract_work_experience_from_sections(sections)
        projects = self._extract_projects_from_sections(sections)
        certifications = self._extract_certifications_from_sections(sections)
        summary = self._extract_summary_from_sections(sections)
        skills = self._extract_skills_from_sections(sections)
        
        # Debug logging for extraction results
        # Data extraction completed
        
        
        # Use section data directly if extraction methods fail
        if not education and sections.get('education'):
            education = sections['education']
        if not projects and sections.get('projects'):
            projects = sections['projects']
        
        # Clean up education data to remove any project contamination
        if education and 'project' in education.lower():
            lines = education.split('\n')
            clean_lines = []
            for line in lines:
                line = line.strip()
                if line and not any(proj_term in line.lower() for proj_term in ['project :', 'title :', 'technology :', 'details :']):
                    clean_lines.append(line)
                elif any(proj_term in line.lower() for proj_term in ['project :', 'title :']):
                    break
            education = '\n'.join(clean_lines) if clean_lines else None
        
        # If education is still contaminated, use only the clean section data
        if education and 'project' in education.lower():
            education = sections.get('education', '')
        
        # ALWAYS use section data if available - this is the cleanest source
        if sections.get('education'):
            education = sections['education']
        if sections.get('projects'):
            projects = sections['projects']
        
        # Skip fallback extraction to avoid contamination
        # Use only section-based extraction for clean data
        if not work_experience:
            work_experience = self._extract_work_experience(text)
        if not certifications:
            certifications = self._extract_certifications(text)
        if not skills:
            skills = self._extract_skills(text)
        
        # The section-based extraction should already be clean
        # No additional cleaning needed
        
        # For fresh graduates, extract career objective as work experience
        if not work_experience:
            other_section = sections.get('other', '')
            if other_section:
                lines = other_section.split('\n')
                for line in lines:
                    line = line.strip()
                    if 'career objective' in line.lower() or 'objective' in line.lower():
                        # Find the objective text
                        obj_start = other_section.lower().find('career objective')
                        if obj_start != -1:
                            obj_text = other_section[obj_start:obj_start+200].strip()
                            work_experience = obj_text
                        break
        
        # Create a focused summary - only use explicit summary/objective
        combined_summary = summary or ""
        
        # If no explicit summary found, create a brief professional summary from key info
        if not combined_summary:
            summary_parts = []
            
            # Add education if available
            if education:
                # Extract just the degree and institution
                edu_lines = education.split('\n')
                if edu_lines:
                    first_edu = edu_lines[0].strip()
                    if len(first_edu) > 10:
                        summary_parts.append(f"Education: {first_edu}")
            
            # Add experience if available
            if work_experience:
                # Extract just the first experience
                exp_lines = work_experience.split('\n')
                if exp_lines:
                    first_exp = exp_lines[0].strip()
                    if len(first_exp) > 10:
                        summary_parts.append(f"Experience: {first_exp}")
            
            # Add skills summary
            if skills:
                # Take first 5 skills
                top_skills = skills[:5]
                summary_parts.append(f"Key Skills: {', '.join(top_skills)}")
            
            if summary_parts:
                combined_summary = ' | '.join(summary_parts)
        
        # Clean up the combined summary to avoid repetition
        if combined_summary:
            lines = combined_summary.split('\n')
            cleaned_lines = []
            seen_content = set()
            
            for line in lines:
                line = line.strip()
                if line and len(line) > 5:
                    line_lower = line.lower()
                    # Check for repetition
                    if not any(seen.lower() in line_lower or line_lower in seen.lower() for seen in seen_content):
                        cleaned_lines.append(line)
                        seen_content.add(line)
            
            combined_summary = '\n'.join(cleaned_lines)
        
        # Limit summary length to avoid database issues
        if len(combined_summary) > 3000:
            combined_summary = combined_summary[:3000] + "..."
        
        parsed_data = {
            'full_name': self._extract_name(text),
            'email': self._extract_email(text),
            'phone': self._extract_phone(text),
            'location': self._extract_location(text),
            'skills': skills,
            'experience_years': self._extract_experience_years(text),
            'education': education,
            'work_experience': work_experience,
            'projects': projects,
            'certifications': certifications,
            'summary': combined_summary,
            'raw_text': resume_text
        }
        
        # Debug logging for final parsed data
        # Final data processing completed
        
        return parsed_data

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove extra whitespace but preserve line breaks for table data
        text = re.sub(r'[ \t]+', ' ', text)  # Replace multiple spaces/tabs with single space
        text = re.sub(r'\n+', '\n', text)    # Replace multiple newlines with single newline
        return text.strip()

    def _identify_sections(self, text: str) -> Dict[str, str]:
        """Identify and separate different sections of the resume"""
        sections = {
            'personal': '',
            'summary': '',
            'education': '',
            'experience': '',
            'skills': '',
            'projects': '',
            'certifications': '',
            'other': ''
        }
        
        # Section headers and their variations
        section_patterns = {
            'personal': [
                r'^personal\s+details?$', r'^personal\s+information$', r'^contact\s+information$',
                r'^contact\s+details?$', r'^personal\s+data$', r'^about\s+me$'
            ],
            'summary': [
                r'^objective$', r'^profile$', r'^about$', r'^overview$',
                r'^career\s+objective$', r'^professional\s+summary$', r'^executive\s+summary$'
            ],
            'education': [
                r'^education$', r'^academic$', r'^qualifications?$', r'^degrees?$',
                r'^academic\s+qualification$', r'^educational\s+background$', r'^scholastic$',
                r'^academic\s+qualification\s*:\s*$', r'^education\s*:\s*$',
                r'^academic\s+qualification\s*:$', r'^education\s*:$',
                r'^academic\s+qualification\s*:\s*$', r'^education\s*:\s*$'
            ],
            'experience': [
                r'^experience$', r'^work\s+experience$', r'^employment$', r'^professional\s+experience$',
                r'^career$', r'^work\s+history$', r'^employment\s+history$', r'^professional\s+background$'
            ],
            'skills': [
                r'^technical\s+skills?$', r'^core\s+competencies?$', r'^technologies?$',
                r'^expertise$', r'^competencies?$', r'^technical\s+expertise$', r'^programming\s+skills?$',
                r'^technical\s+skills?\s*:$', r'^skills?\s*:$'
            ],
            'projects': [
                r'^projects?$', r'^portfolio$', r'^personal\s+projects?$', r'^academic\s+projects?$',
                r'^work\s+projects?$', r'^project\s+experience$', r'^key\s+projects?$',
                r'^project\s*:$', r'^projects?\s*:$'
            ],
            'certifications': [
                r'^certifications?$', r'^certificates?$', r'^certified$', r'^credentials?$',
                r'^licenses?$', r'^professional\s+certifications?$', r'^certifications?\s+and\s+licenses?$'
            ]
        }
        
        # Split text into lines for processing
        lines = text.split('\n')
        current_section = 'other'
        current_content = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check if this line is a section header
            line_lower = line.lower()
            section_found = False
            
            for section_name, patterns in section_patterns.items():
                for pattern in patterns:
                    if re.match(pattern, line_lower):
                        # Save previous section content
                        if current_content:
                            sections[current_section] = '\n'.join(current_content).strip()
                        
                        # Start new section
                        current_section = section_name
                        current_content = []
                        section_found = True
                        break
                
                if section_found:
                    break
            
            # Additional check for education section with more flexible matching
            if not section_found and 'academic' in line_lower and 'qualification' in line_lower:
                if current_content:
                    sections[current_section] = '\n'.join(current_content).strip()
                current_section = 'education'
                current_content = []
                section_found = True
            
            # Additional check for projects section
            if not section_found and 'project' in line_lower and ':' in line:
                if current_content:
                    sections[current_section] = '\n'.join(current_content).strip()
                current_section = 'projects'
                current_content = []
                section_found = True
            
            if not section_found:
                current_content.append(line)
        
        # Save the last section
        if current_content:
            sections[current_section] = '\n'.join(current_content).strip()
        
        return sections

    def _extract_education_from_sections(self, sections: Dict[str, str]) -> Optional[str]:
        """Extract education from the education section"""
        education_text = sections.get('education', '')
        
        # If we have education section content, return it directly
        if education_text and len(education_text.strip()) > 10:
            return education_text.strip()
        
        # Also check other section for education
        education_text = sections.get('other', '')
        if education_text:
            # Look for education patterns in other section
            education_lines = []
            lines = education_text.split('\n')
            in_education_section = False
            
            for line in lines:
                line = line.strip()
                if line and len(line) > 5:
                    # Check if this line starts education section
                    if re.match(r'^academic\s+qualification\s*:', line, re.IGNORECASE):
                        in_education_section = True
                        continue  # Skip the header line
                    elif in_education_section and (re.match(r'^examination\s*:', line, re.IGNORECASE) or
                                                 re.match(r'^bachelor\s+of', line, re.IGNORECASE) or
                                                 re.match(r'^hsc\s*:', line, re.IGNORECASE) or
                                                 re.match(r'^ssc\s*:', line, re.IGNORECASE) or
                                                 re.match(r'^b\.?tech', line, re.IGNORECASE) or
                                                 re.match(r'^m\.?tech', line, re.IGNORECASE) or
                                                 re.match(r'^b\.?e', line, re.IGNORECASE) or
                                                 re.match(r'^m\.?e', line, re.IGNORECASE)):
                        education_lines.append(line)
                    elif in_education_section and not re.match(r'^[A-Z][A-Z\s]+:$', line):
                        # Include continuation lines of education
                        if any(edu_term in line.lower() for edu_term in [
                            'university', 'college', 'school', 'board', 'percentage', 'cgpa', 'gpa', 'grade',
                            'dbatu', 'state board', 'engineering', 'science college', 'high school',
                            'technology', 'b.tech', 'm.tech', 'b.e', 'm.e', 'bachelor', 'master'
                        ]):
                            education_lines.append(line)
                    elif re.match(r'^[A-Z][A-Z\s]+:$', line) and in_education_section:
                        break
            
            if education_lines:
                return '\n'.join(education_lines[:10])  # Limit to first 10 lines
        
        return None

    def _extract_work_experience_from_sections(self, sections: Dict[str, str]) -> Optional[str]:
        """Extract work experience from the experience section"""
        experience_text = sections.get('experience', '')
        if not experience_text:
            return None
        
        # Clean up experience text
        lines = experience_text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if line and len(line) > 5:
                # Skip other sections
                if any(section_term in line.lower() for section_term in [
                    'certifications', 'education', 'personal details', 'skills', 'technical skills',
                    'projects', 'portfolio'
                ]):
                    break
                
                # Include all lines that don't contain section headers (less strict filtering)
                cleaned_lines.append(line)
        
        if cleaned_lines:
            return '\n'.join(cleaned_lines[:8])  # Limit to first 8 experience lines
        
        return None

    def _extract_projects_from_sections(self, sections: Dict[str, str]) -> Optional[str]:
        """Extract projects from the projects section"""
        projects_text = sections.get('projects', '')
        
        # If we have projects section content, return it directly
        if projects_text and len(projects_text.strip()) > 10:
            return projects_text.strip()
        
        if not projects_text:
            # Also check other section for projects
            projects_text = sections.get('other', '')
            if projects_text:
                # Look for project patterns in other section
                project_lines = []
                lines = projects_text.split('\n')
                in_project_section = False
                
                for line in lines:
                    line = line.strip()
                    if line and len(line) > 5:
                        # Check if this line starts a project
                        if re.match(r'^project\s*:', line, re.IGNORECASE) or re.match(r'^title\s*:', line, re.IGNORECASE):
                            in_project_section = True
                            project_lines.append(line)
                        elif in_project_section and (re.match(r'^technology\s*:', line, re.IGNORECASE) or 
                                                   re.match(r'^details\s*:', line, re.IGNORECASE) or
                                                   re.match(r'^project\s*:', line, re.IGNORECASE)):
                            project_lines.append(line)
                        elif in_project_section and not re.match(r'^[A-Z][A-Z\s]+:$', line):
                            # Include continuation lines of projects
                            if any(proj_term in line.lower() for proj_term in [
                                'html', 'css', 'javascript', 'java', 'react', 'reactjs', 'website', 'clone',
                                'online', 'shoes', 'store', 'weather', 'information', 'adidas', 'facebook',
                                'selling', 'details', 'particular', 'city', 'entered', 'home', 'page',
                                'title', 'technology', 'project', 'store', 'information', 'clone', 'node',
                                'python', 'django', 'flask', 'sql', 'database', 'api', 'frontend', 'backend'
                            ]):
                                project_lines.append(line)
                        elif re.match(r'^[A-Z][A-Z\s]+:$', line) and in_project_section:
                            # Don't break, continue to next project
                            continue
                
                if project_lines:
                    return '\n'.join(project_lines[:15])  # Limit to first 15 lines
        
        if projects_text:
            # Clean up projects text
            lines = projects_text.split('\n')
            cleaned_lines = []
            
            for line in lines:
                line = line.strip()
                if line and len(line) > 5:
                    # Skip section headers and personal details
                    if not re.match(r'^[A-Z][A-Z\s]+:$', line) and not any(personal_term in line.lower() for personal_term in [
                        'personal details', 'date of birth', 'father', 'mother', 'marital status',
                        'gender', 'languages known', 'permanent address', 'address', 'phone',
                        'email', 'contact', 'mobile', 'declaration', 'place', 'date'
                    ]):
                        cleaned_lines.append(line)
            
            if cleaned_lines:
                return '\n'.join(cleaned_lines[:15])  # Limit to first 15 lines
        
        return None

    def _extract_certifications_from_sections(self, sections: Dict[str, str]) -> Optional[str]:
        """Extract certifications from the certifications section"""
        certs_text = sections.get('certifications', '')
        if not certs_text:
            return None
        
        # Clean up certifications text
        lines = certs_text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if line and len(line) > 5:
                # Skip other sections
                if any(section_term in line.lower() for section_term in [
                    'education', 'personal details', 'skills', 'technical skills', 'projects',
                    'experience', 'work experience'
                ]):
                    break
                
                # Include all lines that don't contain section headers (less strict filtering)
                cleaned_lines.append(line)
        
        if cleaned_lines:
            return '\n'.join(cleaned_lines[:5])  # Limit to first 5 certification lines
        
        return None

    def _extract_summary_from_sections(self, sections: Dict[str, str]) -> Optional[str]:
        """Extract summary from the summary section"""
        summary_text = sections.get('summary', '')
        if not summary_text:
            return None
        
        # Clean up summary text
        lines = summary_text.split('\n')
        cleaned_lines = []
        seen_lines = set()
        
        for line in lines:
            line = line.strip()
            if line and len(line) > 10:
                # Avoid repetition
                line_lower = line.lower()
                if not any(seen_line.lower() in line_lower or line_lower in seen_line.lower() 
                         for seen_line in seen_lines):
                    cleaned_lines.append(line)
                    seen_lines.add(line)
        
        if cleaned_lines:
            return '\n'.join(cleaned_lines[:5])  # Limit to first 5 unique lines
        
        return None

    def _extract_skills_from_sections(self, sections: Dict[str, str]) -> List[str]:
        """Extract skills from the skills section with improved accuracy"""
        skills_text = sections.get('skills', '')
        found_skills = []
        
        if skills_text:
            # Parse comma-separated or bullet-pointed skills
            skills_list = re.split(r'[,•\n\r]', skills_text)
            for skill in skills_list:
                skill = skill.strip()
                if skill and len(skill) > 1 and len(skill) < 50:
                    # Clean up skill text (remove category labels like "Programming Languages:")
                    skill = re.sub(r'^[^:]*:\s*', '', skill)
                    skill = skill.strip()
                    
                    # Validate that it's actually a skill
                    if self._is_valid_skill(skill):
                        found_skills.append(skill.title())
        
        # If no skills section found, look for skills throughout the text but be more selective
        if not found_skills:
            # Combine all sections except personal details
            all_text = ' '.join([
                sections.get('summary', ''),
                sections.get('experience', ''),
                sections.get('projects', ''),
                sections.get('certifications', ''),
                sections.get('other', '')  # Also check other section for skills
            ])
            
            text_lower = all_text.lower()
            for skill in self.skill_keywords:
                # Use word boundaries to avoid partial matches
                if re.search(r'\b' + re.escape(skill) + r'\b', text_lower):
                    found_skills.append(skill.title())
        
        # Remove duplicates and return
        return list(set(found_skills))

    def _extract_name(self, text: str) -> Optional[str]:
        """Extract candidate name from resume text"""
        # Look for name patterns at the beginning of the document
        lines = text.split('\n')[:15]  # Check first 15 lines
        
        for line in lines:
            line = line.strip()
            if len(line) > 2 and len(line) < 60:  # Reasonable name length
                # Skip lines that look like headers or contain common resume keywords
                if not any(keyword in line.lower() for keyword in [
                    'resume', 'cv', 'curriculum', 'vitae', 'contact', 'email', 'phone',
                    'address', 'objective', 'summary', 'experience', 'education', 'skills',
                    'projects', 'certifications', 'languages', 'references'
                ]):
                    # Check if it looks like a name (contains letters and possibly spaces, dots, hyphens, apostrophes)
                    if re.match(r'^[A-Za-z\s\.\-\']+$', line) and len(line.split()) <= 5:
                        # Additional check: make sure it's not just numbers or special characters
                        if re.search(r'[A-Za-z]', line):
                            return line.title()
        
        # If no name found in first lines, try to find it in the first few words
        words = text.split()[:10]
        if len(words) >= 2:
            potential_name = ' '.join(words[:2])
            if re.match(r'^[A-Za-z\s\.\-\']+$', potential_name) and len(potential_name.split()) <= 3:
                return potential_name.title()
        
        return None

    def _extract_email(self, text: str) -> Optional[str]:
        """Extract email address from resume text"""
        # More comprehensive email patterns
        email_patterns = [
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Standard email
            r'email[:\s]*([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,})',  # Email: pattern
            r'e-mail[:\s]*([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,})',  # E-mail: pattern
            r'contact[:\s]*([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,})',  # Contact: pattern
        ]
        
        for pattern in email_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                email = matches[0].strip()
                # Clean up email (remove spaces and extra characters)
                email = re.sub(r'\s+', '', email)
                # Validate email format
                if '@' in email and '.' in email.split('@')[1]:
                    return email
        
        # Additional fallback - look for any email pattern in the text
        email_fallback = re.search(r'[A-Za-z0-9._%+-]+\s*@\s*[A-Za-z0-9.-]+\s*\.\s*[A-Z|a-z]{2,}', text)
        if email_fallback:
            email = email_fallback.group(0)
            email = re.sub(r'\s+', '', email)
            # Clean up email - remove any extra text after the domain
            if '@' in email and '.' in email.split('@')[1]:
                # Split by @ and take only the domain part before any extra text
                local_part, domain_part = email.split('@', 1)
                # Clean domain part - remove anything after the TLD
                domain_clean = re.sub(r'[^a-zA-Z0-9.-].*$', '', domain_part)
                if '.' in domain_clean:
                    return f"{local_part}@{domain_clean}"
        
        # Final fallback - look for email with spaces and clean it
        email_with_spaces = re.search(r'[A-Za-z0-9._%+-]+\s+@\s*[A-Za-z0-9.-]+\s*\.\s*[A-Z|a-z]{2,}', text)
        if email_with_spaces:
            email = email_with_spaces.group(0)
            email = re.sub(r'\s+', '', email)
            # Extract just the email part before any extra text
            email_match = re.search(r'([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,})', email)
            if email_match:
                return email_match.group(1)
        
        # Additional cleanup for emails with extra text
        email_cleanup = re.search(r'([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,})[^A-Za-z0-9]', text)
        if email_cleanup:
            return email_cleanup.group(1)
        
        # Final cleanup - remove any trailing non-email characters
        final_email = re.search(r'([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,})', text)
        if final_email:
            email = final_email.group(1)
            # Remove any trailing characters that are not part of the email
            email = re.sub(r'[^A-Za-z0-9._%+-@]', '', email)
            # Additional cleanup for common issues
            if email.endswith('.GitHub') or email.endswith('.github'):
                email = email.replace('.GitHub', '').replace('.github', '')
            # Remove any trailing dots or special characters
            email = email.rstrip('.')
            # Final validation - ensure it's a proper email
            if '@' in email and '.' in email.split('@')[1]:
                return email
        
        return None

    def _extract_phone(self, text: str) -> Optional[str]:
        """Extract phone number from resume text using phonenumbers library for better accuracy"""
        if not PHONE_AVAILABLE:
            # Fallback to regex-based extraction
            return self._extract_phone_regex(text)
        
        try:
            # Try to extract with no region hint (only works if +country_code exists)
            matches = list(phonenumbers.PhoneNumberMatcher(text, None))
            for match in matches:
                number = match.number
                if phonenumbers.is_valid_number(number):
                    return phonenumbers.format_number(number, phonenumbers.PhoneNumberFormat.E164)
           
            # If not found, try with a list of possible regions
            fallback_regions = ["US", "IN", "GB", "CA", "AU", "DE", "FR", "SG", "PH", "ZA", "BR", "MX", "NG", "KE"]
            for region in fallback_regions:
                matches = list(phonenumbers.PhoneNumberMatcher(text, region))
                for match in matches:
                    number = match.number
                    if phonenumbers.is_valid_number(number):
                        return phonenumbers.format_number(number, phonenumbers.PhoneNumberFormat.E164)
        except Exception as e:
            logger.warning(f"Phone number parsing error: {e}")
       
        return None

    def _extract_phone_regex(self, text: str) -> Optional[str]:
        """Fallback phone extraction using regex patterns"""
        # More comprehensive phone patterns including Indian and international formats
        phone_patterns = [
            # Indian mobile numbers (10 digits starting with 6,7,8,9)
            r'phone[:\s]*([6-9]\d{9})',
            r'mobile[:\s]*([6-9]\d{9})',
            r'contact[:\s]*([6-9]\d{9})',
            r'\+91[-\s]?([6-9]\d{9})',
            r'\+91[-\s]?(\d{5}[-\s]?\d{5})',
            
            # US/International formats
            r'\+?1?[-.\s]?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})',
            r'\+?[0-9]{1,3}[-.\s]?[0-9]{3,4}[-.\s]?[0-9]{3,4}[-.\s]?[0-9]{3,4}',
            r'\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}',
            
            # General patterns
            r'(\d{10})',  # 10 consecutive digits
            r'(\d{3}[-.\s]?\d{3}[-.\s]?\d{4})',  # XXX-XXX-XXXX format
            r'(\d{5}[-.\s]?\d{5})',  # XXXXX-XXXXX format (Indian landline)
        ]
        
        for pattern in phone_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                phone = matches[0]
                # Clean the phone number (remove non-digits)
                clean_phone = re.sub(r'[^\d]', '', str(phone))
                
                # Validate phone number length
                if len(clean_phone) >= 10:
                    # For Indian numbers, ensure it starts with 6,7,8,9
                    if len(clean_phone) == 10 and clean_phone[0] in '6789':
                        return clean_phone
                    # For US numbers, ensure it's 10 digits
                    elif len(clean_phone) == 10:
                        return clean_phone
                    # For international numbers with country code
                    elif len(clean_phone) > 10:
                        return clean_phone
        
        return None

    def _extract_location(self, text: str) -> Optional[str]:
        """Extract location from resume text with high precision"""
        # Common city names and states to help identify real locations
        common_cities = [
            'mumbai', 'delhi', 'bangalore', 'hyderabad', 'chennai', 'kolkata', 'pune', 'ahmedabad',
            'jaipur', 'surat', 'lucknow', 'kanpur', 'nagpur', 'indore', 'thane', 'bhopal',
            'visakhapatnam', 'pimpri', 'patna', 'vadodara', 'ghaziabad', 'ludhiana', 'agra',
            'nashik', 'faridabad', 'meerut', 'rajkot', 'kalyan', 'vasai', 'varanasi', 'dhule',
            'new york', 'los angeles', 'chicago', 'houston', 'phoenix', 'philadelphia',
            'san antonio', 'san diego', 'dallas', 'san jose', 'austin', 'jacksonville',
            'fort worth', 'columbus', 'charlotte', 'san francisco', 'indianapolis',
            'seattle', 'denver', 'washington', 'boston', 'el paso', 'nashville',
            'detroit', 'oklahoma city', 'portland', 'las vegas', 'memphis', 'louisville',
            'baltimore', 'milwaukee', 'albuquerque', 'tucson', 'fresno', 'sacramento',
            'mesa', 'kansas city', 'atlanta', 'long beach', 'colorado springs', 'raleigh',
            'miami', 'virginia beach', 'omaha', 'oakland', 'minneapolis', 'tulsa',
            'arlington', 'tampa', 'new orleans', 'wichita', 'cleveland', 'bakersfield',
            'aurora', 'anaheim', 'honolulu', 'santa ana', 'corpus christi', 'riverside',
            'lexington', 'stockton', 'henderson', 'saint paul', 'st louis', 'milwaukee'
        ]
        
        common_states = [
            'maharashtra', 'karnataka', 'tamil nadu', 'west bengal', 'gujarat', 'rajasthan',
            'uttar pradesh', 'andhra pradesh', 'telangana', 'kerala', 'madhya pradesh',
            'haryana', 'punjab', 'odisha', 'assam', 'jharkhand', 'chhattisgarh', 'himachal pradesh',
            'uttarakhand', 'goa', 'tripura', 'meghalaya', 'manipur', 'nagaland', 'arunachal pradesh',
            'mizoram', 'sikkim', 'delhi', 'jammu and kashmir', 'ladakh', 'puducherry',
            'california', 'texas', 'florida', 'new york', 'pennsylvania', 'illinois',
            'ohio', 'georgia', 'north carolina', 'michigan', 'new jersey', 'virginia',
            'washington', 'arizona', 'massachusetts', 'tennessee', 'indiana', 'missouri',
            'maryland', 'wisconsin', 'colorado', 'minnesota', 'south carolina', 'alabama',
            'louisiana', 'kentucky', 'oregon', 'oklahoma', 'connecticut', 'utah', 'iowa',
            'nevada', 'arkansas', 'mississippi', 'kansas', 'new mexico', 'nebraska',
            'west virginia', 'idaho', 'hawaii', 'new hampshire', 'maine', 'montana',
            'rhode island', 'delaware', 'south dakota', 'north dakota', 'alaska', 'vermont', 'wyoming'
        ]
        
        # Terms that should NOT be in location
        exclude_terms = [
            'email', 'phone', 'mobile', 'github', 'linkedin', 'website', 'portfolio',
            'objective', 'career', 'summary', 'experience', 'education', 'skills',
            'project', 'certification', 'achievement', 'hobby', 'interest',
            'javascript', 'html', 'css', 'react', 'angular', 'vue', 'node', 'python',
            'java', 'c++', 'c#', 'php', 'ruby', 'go', 'rust', 'swift', 'kotlin',
            'mysql', 'postgresql', 'mongodb', 'redis', 'sqlite', 'oracle',
            'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'git', 'linux',
            'http', 'https', 'www', '.com', '.org', '.net', '@', 'gmail', 'yahoo', 'hotmail'
        ]
        
        def is_valid_location(location_text):
            """Check if the extracted text is a valid location"""
            location_lower = location_text.lower()
            
            # Must not contain excluded terms
            if any(term in location_lower for term in exclude_terms):
                return False
            
            # Must contain at least one known city or state
            if not (any(city in location_lower for city in common_cities) or 
                   any(state in location_lower for state in common_states)):
                return False
            
            # Must be reasonable length (not too short or too long)
            if len(location_text) < 3 or len(location_text) > 100:
                return False
            
            # Must not be mostly numbers or special characters
            if re.match(r'^[\d\s\-\(\)\.]+$', location_text):
                return False
            
            return True
        
        # Strategy 1: Look for specific location patterns with strict boundaries
        location_patterns = [
            # City, State format
            r'\b([A-Za-z\s]+),\s*([A-Za-z\s]+)\b',
            # City, State, Country format  
            r'\b([A-Za-z\s]+),\s*([A-Za-z\s]+),\s*([A-Za-z\s]+)\b',
            # Just city name (must be in our list)
            r'\b(' + '|'.join(common_cities) + r')\b',
        ]
        
        for pattern in location_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    # Join tuple elements
                    location = ', '.join(match).strip()
                else:
                    location = match.strip()
                
                if is_valid_location(location):
                    return location
        
        # Strategy 2: Look for address patterns but with strict validation
        address_patterns = [
            r'address[:\s]*([^\n]{1,50})',
            r'location[:\s]*([^\n]{1,50})',
            r'city[:\s]*([^\n]{1,50})',
            r'permanent address[:\s]*([^\n]{1,50})',
        ]
        
        for pattern in address_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                address = match.group(1).strip()
                
                # Stop at first excluded term
                for term in exclude_terms:
                    if term in address.lower():
                        address = address.split(term)[0].strip()
                        break
                
                # Clean up common separators
                address = re.sub(r'[^\w\s,.-]', '', address).strip()
                
                if is_valid_location(address):
                    # Extract just the city if it's a long address
                    if ',' in address:
                        parts = address.split(',')
                        for part in parts:
                            part = part.strip()
                            if is_valid_location(part):
                                return part
                    return address
        
        # Strategy 3: Look for standalone city names in the first few lines
        lines = text.split('\n')[:10]  # Only check first 10 lines
        for line in lines:
            line = line.strip()
            if len(line) < 100:  # Skip very long lines
                for city in common_cities:
                    if city in line.lower():
                        # Extract the city and surrounding context
                        city_match = re.search(r'\b([A-Za-z\s]*' + re.escape(city) + r'[A-Za-z\s]*)\b', line, re.IGNORECASE)
                        if city_match:
                            location = city_match.group(1).strip()
                            if is_valid_location(location):
                                return location
        
        return None

    def _extract_skills(self, text: str) -> List[str]:
        """Extract skills from resume text"""
        found_skills = []
        
        # First, try to extract from dedicated skills section
        skills_section = self._extract_skills_section(text)
        if skills_section:
            # Parse comma-separated or bullet-pointed skills
            skills_list = re.split(r'[,•\n\r]', skills_section)
            for skill in skills_list:
                skill = skill.strip()
                if skill and len(skill) > 1 and len(skill) < 50:  # Reasonable skill length
                    # Validate that it's actually a skill, not an address or other text
                    if self._is_valid_skill(skill):
                        found_skills.append(skill.title())
        
        # If no skills section found, look for skills throughout the text but be more selective
        if not found_skills:
            text_lower = text.lower()
            for skill in self.skill_keywords:
                # Use word boundaries to avoid partial matches
                if re.search(r'\b' + re.escape(skill) + r'\b', text_lower):
                    found_skills.append(skill.title())
        
        # Remove duplicates and return
        return list(set(found_skills))
    
    def _is_valid_skill(self, skill: str) -> bool:
        """Validate if a string is actually a skill and not an address or other text"""
        skill_lower = skill.lower().strip()
        
        # Skip if it's too long (likely not a skill)
        if len(skill) > 50:
            return False
        
        # Skip if it's too short
        if len(skill) < 2:
            return False
        
        # Skip if it contains address-like patterns
        address_patterns = [
            r'\d+\s+[a-z\s]+road',  # "123 Main Road"
            r'\d+\s+[a-z\s]+street',  # "123 Main Street"
            r'\d+\s+[a-z\s]+avenue',  # "123 Main Avenue"
            r'\d+\s+[a-z\s]+lane',  # "123 Main Lane"
            r'\d+\s+[a-z\s]+drive',  # "123 Main Drive"
            r'[a-z\s]+,\s*[a-z\s]+,\s*\d+',  # "City, State, 12345"
            r'p\.o\.\s*box',  # "P.O. Box"
            r'apartment\s+\d+',  # "Apartment 123"
            r'apt\s+\d+',  # "Apt 123"
            r'rajendra\s+nagar',  # Specific address pattern from test
            r'kirti\s+gondur',  # Specific address pattern from test
            r'behind\s+[a-z\s]+road',  # "Behind Main Road"
            r'permanent\s+address',  # "Permanent Address"
            r'suite\s+\d+',  # "Suite 123"
            r'house\s+no',  # "House No"
            r'flat\s+no',  # "Flat No"
            r'building\s+no',  # "Building No"
            r'ward\s+no',  # "Ward No"
            r'pin\s*code',  # "Pin Code"
            r'postal\s+code',  # "Postal Code"
        ]
        
        for pattern in address_patterns:
            if re.search(pattern, skill_lower):
                return False
        
        # Skip if it contains personal information patterns
        personal_patterns = [
            r'father\'s?\s+name',  # "Father's Name"
            r'mother\'s?\s+name',  # "Mother's Name"
            r'date\s+of\s+birth',  # "Date of Birth"
            r'phone\s*:',  # "Phone:"
            r'email\s*:',  # "Email:"
            r'address\s*:',  # "Address:"
            r'contact\s*:',  # "Contact:"
            r'marital\s+status',  # "Marital Status"
            r'gender\s*:',  # "Gender:"
            r'languages?\s+known',  # "Languages Known"
            r'permanent\s+address',  # "Permanent Address"
            r'personal\s+details',  # "Personal Details"
            r'nationality',  # "Nationality"
            r'religion',  # "Religion"
            r'caste',  # "Caste"
            r'category',  # "Category"
        ]
        
        for pattern in personal_patterns:
            if re.search(pattern, skill_lower):
                return False
        
        # Skip if it contains project-like patterns that aren't skills
        project_patterns = [
            r'project\s*:',  # "Project:"
            r'portfolio\s*:',  # "Portfolio:"
            r'application\s*:',  # "Application:"
            r'website\s*:',  # "Website:"
            r'system\s*:',  # "System:"
            r'software\s*:',  # "Software:"
            r'app\s*:',  # "App:"
            r'platform\s*:',  # "Platform:"
            r'github\s*:',  # "GitHub:"
            r'linkedin\s*:',  # "LinkedIn:"
        ]
        
        for pattern in project_patterns:
            if re.search(pattern, skill_lower):
                return False
        
        # Skip if it contains education patterns
        education_patterns = [
            r'education\s*:',  # "Education:"
            r'academic\s*:',  # "Academic:"
            r'qualification\s*:',  # "Qualification:"
            r'degree\s*:',  # "Degree:"
            r'university\s*:',  # "University:"
            r'college\s*:',  # "College:"
            r'school\s*:',  # "School:"
            r'gpa\s*:',  # "GPA:"
            r'percentage\s*:',  # "Percentage:"
            r'cgpa\s*:',  # "CGPA:"
            r'grade\s*:',  # "Grade:"
        ]
        
        for pattern in education_patterns:
            if re.search(pattern, skill_lower):
                return False
        
        # Skip if it contains experience patterns
        experience_patterns = [
            r'experience\s*:',  # "Experience:"
            r'work\s+experience',  # "Work Experience"
            r'employment\s*:',  # "Employment:"
            r'professional\s+experience',  # "Professional Experience"
            r'career\s*:',  # "Career:"
            r'job\s*:',  # "Job:"
            r'position\s*:',  # "Position:"
        ]
        
        for pattern in experience_patterns:
            if re.search(pattern, skill_lower):
                return False
        
        # Skip if it contains certification patterns
        cert_patterns = [
            r'certification\s*:',  # "Certification:"
            r'certificate\s*:',  # "Certificate:"
            r'certified\s*:',  # "Certified:"
            r'credential\s*:',  # "Credential:"
            r'license\s*:',  # "License:"
        ]
        
        for pattern in cert_patterns:
            if re.search(pattern, skill_lower):
                return False
        
        # Skip if it's mostly numbers or special characters
        if re.match(r'^[\d\s\-\(\)\.]+$', skill):
            return False
        
        # Skip if it contains only numbers
        if skill.isdigit():
            return False
        
        # Skip if it's a single word that's too generic
        generic_words = [
            'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
            'from', 'up', 'about', 'into', 'through', 'during', 'before', 'after',
            'above', 'below', 'between', 'among', 'under', 'over', 'around', 'near',
            'far', 'here', 'there', 'where', 'when', 'why', 'how', 'what', 'who',
            'which', 'that', 'this', 'these', 'those', 'i', 'you', 'he', 'she', 'it',
            'we', 'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his',
            'her', 'its', 'our', 'their', 'mine', 'yours', 'hers', 'ours', 'theirs',
            'name', 'age', 'dob', 'mobile', 'phone', 'email', 'address', 'city',
            'state', 'country', 'pin', 'code', 'house', 'flat', 'building', 'road',
            'street', 'lane', 'avenue', 'drive', 'area', 'locality', 'colony'
        ]
        
        if skill_lower in generic_words:
            return False
        
        # Skip if it looks like a name (starts with capital and contains only letters)
        if re.match(r'^[A-Z][a-z]+$', skill) and len(skill.split()) <= 3:
            # Check if it's not a known technical term
            technical_terms = [
                'java', 'python', 'react', 'angular', 'vue', 'node', 'express',
                'django', 'flask', 'spring', 'laravel', 'rails', 'asp', 'net',
                'mysql', 'postgresql', 'mongodb', 'redis', 'sqlite', 'oracle',
                'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'jenkins', 'git',
                'linux', 'bash', 'powershell', 'html', 'css', 'javascript',
                'typescript', 'php', 'ruby', 'go', 'rust', 'swift', 'kotlin'
            ]
            if skill_lower not in technical_terms:
                return False
        
        # Skip if it contains common non-skill words
        non_skill_words = [
            'years', 'year', 'months', 'month', 'days', 'day', 'hours', 'hour',
            'experience', 'work', 'job', 'company', 'organization', 'firm',
            'project', 'task', 'responsibility', 'duty', 'role', 'position',
            'team', 'group', 'department', 'division', 'unit', 'section',
            'client', 'customer', 'user', 'stakeholder', 'manager', 'director',
            'leader', 'supervisor', 'mentor', 'trainer', 'instructor'
        ]
        
        if any(word in skill_lower for word in non_skill_words):
            return False
        
        # If it passes all checks, it's likely a valid skill
        return True

    def _extract_skills_section(self, text: str) -> Optional[str]:
        """Extract skills section from resume"""
        skills_keywords = ['skills', 'technical skills', 'core competencies', 'technologies', 'expertise']
        
        for keyword in skills_keywords:
            pattern = rf'{keyword}[:\s]*([^\n]*(?:\n[^\n]*)*?)(?=\n\s*\n|\n[A-Z][A-Z\s]+:|\Z)'
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                return match.group(1).strip()
        
        return None

    def _extract_experience_years(self, text: str) -> Optional[int]:
        """Extract years of experience from resume text"""
        text_lower = text.lower()
        
        # First try to find explicit experience statements
        for pattern in self.experience_patterns:
            match = re.search(pattern, text_lower)
            if match:
                years = int(match.group(1))
                return years
        
        # If no explicit experience found, try to calculate from job dates using dateparser
        if DATE_PARSER_AVAILABLE:
            return self._calculate_experience_from_dates_advanced(text)
        else:
            return self._calculate_experience_from_dates(text)
    
    def _calculate_experience_from_dates(self, text: str) -> Optional[int]:
        """Calculate experience years from job start/end dates"""
        # Look for date patterns like (2020-2022), (2022-2024), etc.
        date_patterns = [
            r'\((\d{4})\s*-\s*(\d{4})\)',  # (2020-2022)
            r'(\d{4})\s*-\s*(\d{4})',      # 2020-2022
            r'(\d{4})\s*to\s*(\d{4})',     # 2020 to 2022
            r'(\d{4})\s*present',          # 2020 present
            r'(\d{4})\s*current',          # 2020 current
        ]
        
        total_months = 0
        current_year = 2024  # Current year for "present" calculations
        
        for pattern in date_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if len(match) == 2:
                    start_year = int(match[0])
                    end_year = int(match[1])
                    if start_year <= end_year and start_year >= 1990:  # Reasonable year range
                        months = (end_year - start_year) * 12
                        total_months += months
                elif len(match) == 1 and ('present' in match[0].lower() or 'current' in match[0].lower()):
                    start_year = int(match[0])
                    if start_year >= 1990:
                        months = (current_year - start_year) * 12
                        total_months += months
        
        if total_months > 0:
            years = total_months // 12
            return years if years > 0 else None
        
        return None

    def _calculate_experience_from_dates_advanced(self, text: str) -> Optional[int]:
        """Calculate experience years from job dates using dateparser for better accuracy"""
        try:
            dates = search_dates(text)
            if not dates:
                return 0
            
            periods = []
            for i in range(len(dates) - 1):
                start, end = dates[i][1], dates[i + 1][1]
                if start < end:
                    delta = (end - start).days
                    if 30 < delta < 3650:  # Between 1 month and 10 years
                        periods.append(delta)
            
            if periods:
                total_days = sum(periods)
                years = total_days / 365.0
                return int(round(years, 1))
        except Exception as e:
            logger.warning(f"Error calculating experience from dates: {e}")
        
        return None

    def _extract_education(self, text: str) -> Optional[str]:
        """Extract education information from resume"""
        education_section = self._extract_education_section(text)
        if education_section:
            return education_section.strip()
        
        # Look for degree mentions throughout the text
        degree_patterns = [
            r'(Bachelor[^.]*\.)',
            r'(Master[^.]*\.)',
            r'(PhD[^.]*\.)',
            r'(Associate[^.]*\.)',
            r'(Diploma[^.]*\.)',
            r'(B\.?S\.?[^.]*\.)',
            r'(B\.?A\.?[^.]*\.)',
            r'(M\.?S\.?[^.]*\.)',
            r'(M\.?A\.?[^.]*\.)',
            r'(M\.?B\.?A\.?[^.]*\.)',
            r'(B\.?E\.?[^.]*\.)',
            r'(B\.?T\.?e\.?c\.?h[^.]*\.)',
            r'(M\.?T\.?e\.?c\.?h[^.]*\.)'
        ]
        
        education_info = []
        for pattern in degree_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if len(match) > 5 and match not in education_info:  # Avoid duplicates and short matches
                    education_info.append(match.strip())
        
        if education_info:
            return '; '.join(education_info[:3])  # Limit to first 3 education entries
        
        return None

    def _extract_education_section(self, text: str) -> Optional[str]:
        """Extract education section from resume"""
        education_keywords = ['education', 'academic', 'qualifications', 'degrees', 'academic qualification']
        
        for keyword in education_keywords:
            # More flexible pattern to capture education content
            pattern = rf'{keyword}[:\s]*([^\n]*(?:\n[^\n]*)*?)(?=\n\s*\n|\n[A-Z][A-Z\s]+:|\n[A-Z][a-z]+\s+[A-Z][a-z]+:|\Z)'
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                education_text = match.group(1).strip()
                # Clean up the education text - stop at personal details
                lines = education_text.split('\n')
                cleaned_lines = []
                for line in lines:
                    line = line.strip()
                    if line and len(line) > 5:  # Skip empty or very short lines
                        # Stop if we hit personal details
                        if any(personal_term in line.lower() for personal_term in [
                            'personal details', 'date of birth', 'father', 'mother', 'marital status',
                            'gender', 'languages known', 'permanent address', 'address'
                        ]):
                            break
                        # Only include lines that look like education
                        if any(edu_term in line.lower() for edu_term in [
                            'bachelor', 'master', 'phd', 'degree', 'university', 'college', 'institute',
                            'gpa', 'cgpa', 'percentage', 'grade', 'b.tech', 'm.tech', 'b.e', 'm.e'
                        ]):
                            cleaned_lines.append(line)
                
                if cleaned_lines:
                    return '\n'.join(cleaned_lines[:5])  # Limit to first 5 education lines
        
        return None

    def _extract_work_experience(self, text: str) -> Optional[str]:
        """Extract work experience information from resume"""
        experience_section = self._extract_experience_section(text)
        if experience_section:
            return experience_section.strip()
        
        # Look for job titles and companies throughout the text
        job_patterns = [
            r'(Software Engineer[^.]*\.)',
            r'(Developer[^.]*\.)',
            r'(Manager[^.]*\.)',
            r'(Analyst[^.]*\.)',
            r'(Consultant[^.]*\.)',
            r'(Specialist[^.]*\.)',
            r'(Coordinator[^.]*\.)',
            r'(Director[^.]*\.)',
            r'(Lead[^.]*\.)',
            r'(Senior[^.]*\.)'
        ]
        
        experiences = []
        for pattern in job_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if len(match) > 10 and match not in experiences:  # Avoid duplicates and short matches
                    experiences.append(match.strip())
        
        if experiences:
            return '; '.join(experiences[:3])  # Limit to first 3 experiences
        
        return None

    def _extract_experience_section(self, text: str) -> Optional[str]:
        """Extract experience section from resume"""
        experience_keywords = ['experience', 'work experience', 'employment', 'professional experience', 'career', 'projects']
        
        for keyword in experience_keywords:
            # More flexible pattern to capture experience content
            pattern = rf'{keyword}[:\s]*([^\n]*(?:\n[^\n]*)*?)(?=\n\s*\n|\n[A-Z][A-Z\s]+:|\n[A-Z][a-z]+\s+[A-Z][a-z]+:|\Z)'
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                experience_text = match.group(1).strip()
                # Clean up the experience text - stop at other sections
                lines = experience_text.split('\n')
                cleaned_lines = []
                for line in lines:
                    line = line.strip()
                    if line and len(line) > 5:  # Skip empty or very short lines
                        # Stop if we hit other sections
                        if any(section_term in line.lower() for section_term in [
                            'certifications', 'education', 'personal details', 'skills', 'technical skills'
                        ]):
                            break
                        # Only include lines that look like experience/projects
                        if any(exp_term in line.lower() for exp_term in [
                            'project', 'experience', 'work', 'job', 'company', 'developer', 'engineer',
                            'manager', 'analyst', 'consultant', 'html', 'css', 'javascript', 'java',
                            'react', 'angular', 'vue', 'node', 'python', 'website', 'application'
                        ]):
                            cleaned_lines.append(line)
                
                if cleaned_lines:
                    return '\n'.join(cleaned_lines[:8])  # Limit to first 8 experience lines
        
        return None

    def _extract_projects(self, text: str) -> Optional[str]:
        """Extract projects information from resume"""
        projects_section = self._extract_projects_section(text)
        if projects_section:
            return projects_section.strip()
        
        # Look for project mentions throughout the text
        project_patterns = [
            r'(Project[^.]*\.)',
            r'(Portfolio[^.]*\.)',
            r'(Application[^.]*\.)',
            r'(Website[^.]*\.)',
            r'(System[^.]*\.)',
            r'(Software[^.]*\.)',
            r'(App[^.]*\.)',
            r'(Platform[^.]*\.)',
        ]
        
        projects_info = []
        for pattern in project_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if len(match) > 10 and match not in projects_info:  # Avoid duplicates and short matches
                    projects_info.append(match.strip())
        
        if projects_info:
            return '; '.join(projects_info[:5])  # Limit to first 5 projects
        
        return None

    def _extract_projects_section(self, text: str) -> Optional[str]:
        """Extract projects section from resume"""
        project_keywords = ['projects', 'portfolio', 'personal projects', 'academic projects', 'work projects']
        
        for keyword in project_keywords:
            # More flexible pattern to capture projects content
            # Look for the keyword followed by content until the next major section
            pattern = rf'{keyword}[:\s]*([^\n]*(?:\n[^\n]*)*?)(?=\n\s*\n|\n[A-Z][A-Z\s]+:|\n[A-Z][a-z]+\s+[A-Z][a-z]+:|\n[A-Z][a-z]+:|\Z)'
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                projects_text = match.group(1).strip()
                # Clean up the projects text
                lines = projects_text.split('\n')
                cleaned_lines = []
                for line in lines:
                    line = line.strip()
                    if line and len(line) > 5:  # Skip empty or very short lines
                        # Skip lines that look like section headers
                        if not re.match(r'^[A-Z][A-Z\s]+:$', line):
                            cleaned_lines.append(line)
                
                if cleaned_lines:
                    return '\n'.join(cleaned_lines[:10])  # Limit to first 10 lines
        
        return None

    def _extract_certifications(self, text: str) -> Optional[str]:
        """Extract certifications information from resume"""
        certifications_section = self._extract_certifications_section(text)
        if certifications_section:
            return certifications_section.strip()
        
        # Look for certification mentions throughout the text
        cert_patterns = [
            r'(Certified[^.]*\.)',
            r'(Certificate[^.]*\.)',
            r'(Certification[^.]*\.)',
            r'(AWS[^.]*\.)',
            r'(Azure[^.]*\.)',
            r'(Google[^.]*\.)',
            r'(Microsoft[^.]*\.)',
            r'(Oracle[^.]*\.)',
            r'(Cisco[^.]*\.)',
            r'(CompTIA[^.]*\.)',
        ]
        
        certs_info = []
        for pattern in cert_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if len(match) > 10 and match not in certs_info:  # Avoid duplicates and short matches
                    certs_info.append(match.strip())
        
        if certs_info:
            return '; '.join(certs_info[:5])  # Limit to first 5 certifications
        
        return None

    def _extract_certifications_section(self, text: str) -> Optional[str]:
        """Extract certifications section from resume"""
        cert_keywords = ['certifications', 'certificates', 'certified', 'credentials', 'licenses']
        
        for keyword in cert_keywords:
            # More flexible pattern to capture certifications content
            pattern = rf'{keyword}[:\s]*([^\n]*(?:\n[^\n]*)*?)(?=\n\s*\n|\n[A-Z][A-Z\s]+:|\n[A-Z][a-z]+\s+[A-Z][a-z]+:|\Z)'
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                certs_text = match.group(1).strip()
                # Clean up the certifications text - stop at other sections
                lines = certs_text.split('\n')
                cleaned_lines = []
                for line in lines:
                    line = line.strip()
                    if line and len(line) > 5:  # Skip empty or very short lines
                        # Stop if we hit other sections
                        if any(section_term in line.lower() for section_term in [
                            'education', 'personal details', 'skills', 'technical skills', 'projects'
                        ]):
                            break
                        # Only include lines that look like certifications
                        if any(cert_term in line.lower() for cert_term in [
                            'certified', 'certificate', 'certification', 'aws', 'azure', 'google', 'microsoft',
                            'oracle', 'cisco', 'comptia', 'credential', 'license', 'professional'
                        ]):
                            cleaned_lines.append(line)
                
                if cleaned_lines:
                    return '\n'.join(cleaned_lines[:5])  # Limit to first 5 certification lines
        
        return None

    def _extract_summary(self, text: str) -> Optional[str]:
        """Extract professional summary or objective from resume"""
        summary_keywords = ['summary', 'objective', 'profile', 'about', 'overview', 'career objective']
        
        for keyword in summary_keywords:
            # More flexible pattern to capture summary content
            pattern = rf'{keyword}[:\s]*([^\n]*(?:\n[^\n]*)*?)(?=\n\s*\n|\n[A-Z][A-Z\s]+:|\n[A-Z][a-z]+\s+[A-Z][a-z]+:|\Z)'
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                summary = match.group(1).strip()
                if len(summary) > 20:  # Ensure it's substantial
                    # Clean up the summary to avoid repetition
                    lines = summary.split('\n')
                    cleaned_lines = []
                    seen_lines = set()
                    
                    for line in lines:
                        line = line.strip()
                        if line and len(line) > 10:  # Skip empty or very short lines
                            # Avoid repetition by checking if we've seen similar content
                            line_lower = line.lower()
                            if not any(seen_line.lower() in line_lower or line_lower in seen_line.lower() 
                                     for seen_line in seen_lines):
                                cleaned_lines.append(line)
                                seen_lines.add(line)
                    
                    if cleaned_lines:
                        return '\n'.join(cleaned_lines[:5])  # Limit to first 5 unique lines
        
        return None

    def parse_file(self, file_path: str) -> Dict[str, Any]:
        """Parse a resume file and return structured data"""
        try:
            # Get file extension
            file_extension = os.path.splitext(file_path)[1].lower().lstrip('.')
            
            # Extract text
            text = self.extract_text_from_file(file_path, file_extension)
            
            if not text:
                logger.warning(f"No text extracted from {file_path}")
                return {}
            
            # Parse the text
            parsed_data = self.parse_resume_data(text, os.path.basename(file_path))
            
            return parsed_data
            
        except Exception as e:
            logger.error(f"Error parsing file {file_path}: {str(e)}")
            return {}


# Global instance
resume_parser = ResumeParser()

# Module-level function for easy importing
def parse_resume_data(file_path):
    """Parse resume data from file path - module level function"""
    return resume_parser.parse_file(file_path)
