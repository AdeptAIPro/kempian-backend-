import React, { useState } from 'react';
import styled from 'styled-components';
import { 
  User, 
  Mail, 
  Phone, 
  MapPin, 
  Calendar, 
  Star, 
  Brain, 
  Shield,
  ExternalLink,
  ChevronDown,
  ChevronUp,
  Award,
  Target,
  Zap
} from 'lucide-react';
import { formatCandidateData, calculateScoreColor, getGradeColor } from '../services/api';

const CardContainer = styled.div`
  background: white;
  border: 1px solid #e5e7eb;
  border-radius: 0.75rem;
  overflow: hidden;
  transition: all 0.2s ease-in-out;
  box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);

  &:hover {
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    transform: translateY(-2px);
  }

  ${props => props.viewMode === 'list' && `
    display: flex;
    align-items: center;
    gap: 1rem;
    padding: 1.5rem;
  `}
`;

const CardHeader = styled.div`
  padding: 1.5rem;
  border-bottom: 1px solid #f3f4f6;
  background: #f8fafc;
  display: flex;
  align-items: center;
  justify-content: space-between;

  ${props => props.viewMode === 'list' && `
    padding: 0;
    border-bottom: none;
    background: transparent;
    flex: 1;
  `}
`;

const CandidateInfo = styled.div`
  display: flex;
  align-items: center;
  gap: 1rem;
  flex: 1;
  min-width: 0;

  ${props => props.viewMode === 'list' && `
    gap: 1.5rem;
  `}
`;

const Avatar = styled.div`
  width: 48px;
  height: 48px;
  border-radius: 50%;
  background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
  color: white;
  display: flex;
  align-items: center;
  justify-content: center;
  font-weight: 600;
  font-size: 1.125rem;
  flex-shrink: 0;

  ${props => props.viewMode === 'list' && `
    width: 56px;
    height: 56px;
    font-size: 1.25rem;
  `}
`;

const CandidateDetails = styled.div`
  flex: 1;
  min-width: 0;
`;

const CandidateName = styled.h3`
  font-size: 1.125rem;
  font-weight: 600;
  color: #1e293b;
  margin: 0 0 0.25rem 0;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;

  ${props => props.viewMode === 'list' && `
    font-size: 1.25rem;
  `}
`;

const CandidateTitle = styled.p`
  font-size: 0.875rem;
  color: #6b7280;
  margin: 0 0 0.5rem 0;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
`;

const SkillsContainer = styled.div`
  display: flex;
  flex-wrap: wrap;
  gap: 0.25rem;
  margin-bottom: 0.5rem;
`;

const SkillTag = styled.span`
  background: #eff6ff;
  color: #1d4ed8;
  font-size: 0.75rem;
  font-weight: 500;
  padding: 0.25rem 0.5rem;
  border-radius: 0.375rem;
  white-space: nowrap;
`;

const ScoreContainer = styled.div`
  display: flex;
  align-items: center;
  gap: 0.5rem;
`;

const ScoreBadge = styled.div`
  background: ${props => props.color}20;
  color: ${props => props.color};
  padding: 0.5rem 1rem;
  border-radius: 0.5rem;
  font-weight: 600;
  font-size: 0.875rem;
  display: flex;
  align-items: center;
  gap: 0.25rem;
`;

const GradeBadge = styled.div`
  background: ${props => props.color}20;
  color: ${props => props.color};
  width: 32px;
  height: 32px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-weight: 700;
  font-size: 0.875rem;
`;

const CardBody = styled.div`
  padding: 1.5rem;
  display: flex;
  flex-direction: column;
  gap: 1rem;

  ${props => props.viewMode === 'list' && `
    flex: 2;
    padding: 0;
  `}
`;

const ContactInfo = styled.div`
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
`;

const ContactItem = styled.div`
  display: flex;
  align-items: center;
  gap: 0.5rem;
  font-size: 0.875rem;
  color: #6b7280;
`;

const ExperienceInfo = styled.div`
  display: flex;
  align-items: center;
  gap: 0.5rem;
  font-size: 0.875rem;
  color: #6b7280;
`;

const ExpandButton = styled.button`
  background: none;
  border: none;
  color: #6b7280;
  cursor: pointer;
  padding: 0.5rem;
  border-radius: 0.375rem;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: all 0.2s ease-in-out;

  &:hover {
    background: #f3f4f6;
    color: #374151;
  }
`;

const ExpandedContent = styled.div`
  padding: 1.5rem;
  border-top: 1px solid #f3f4f6;
  background: #f8fafc;
`;

const ResumePreview = styled.div`
  background: white;
  border: 1px solid #e5e7eb;
  border-radius: 0.5rem;
  padding: 1rem;
  margin-bottom: 1rem;
`;

const ResumeText = styled.p`
  font-size: 0.875rem;
  color: #374151;
  line-height: 1.6;
  margin: 0;
  max-height: 120px;
  overflow-y: auto;
`;

const AIExplanation = styled.div`
  background: white;
  border: 1px solid #e5e7eb;
  border-radius: 0.5rem;
  padding: 1rem;
  margin-bottom: 1rem;
`;

const AIExplanationTitle = styled.h4`
  font-size: 0.875rem;
  font-weight: 600;
  color: #1e293b;
  margin: 0 0 0.5rem 0;
  display: flex;
  align-items: center;
  gap: 0.5rem;
`;

const AIExplanationText = styled.p`
  font-size: 0.875rem;
  color: #374151;
  line-height: 1.6;
  margin: 0;
`;

const FeatureIndicators = styled.div`
  display: flex;
  align-items: center;
  gap: 0.5rem;
  margin-top: 0.5rem;
`;

const FeatureIndicator = styled.div`
  display: flex;
  align-items: center;
  gap: 0.25rem;
  font-size: 0.75rem;
  color: #6b7280;
`;

const FeatureIcon = styled.div`
  width: 16px;
  height: 16px;
  border-radius: 50%;
  background: ${props => props.color}20;
  color: ${props => props.color};
  display: flex;
  align-items: center;
  justify-content: center;
`;

function CandidateCard({ candidate, viewMode = 'grid' }) {
  const [isExpanded, setIsExpanded] = useState(false);
  const formattedCandidate = formatCandidateData(candidate);
  
  const {
    fullName,
    email,
    phone,
    skills,
    experienceYears,
    resumeText,
    score,
    grade,
    domain,
    aiExplanation,
    sourceURL
  } = formattedCandidate;

  const getInitials = (name) => {
    return name
      .split(' ')
      .map(word => word.charAt(0))
      .join('')
      .toUpperCase()
      .slice(0, 2);
  };

  const scoreColor = calculateScoreColor(score);
  const gradeColor = getGradeColor(grade);

  const displaySkills = skills.slice(0, viewMode === 'list' ? 6 : 4);

  return (
    <CardContainer viewMode={viewMode}>
      <CardHeader viewMode={viewMode}>
        <CandidateInfo viewMode={viewMode}>
          <Avatar viewMode={viewMode}>
            {getInitials(fullName)}
          </Avatar>
          <CandidateDetails>
            <CandidateName viewMode={viewMode}>{fullName}</CandidateName>
            <CandidateTitle>
              {domain === 'healthcare' ? 'Healthcare Professional' : 
               domain === 'technology' ? 'Technology Professional' : 
               'Professional'}
            </CandidateTitle>
            {viewMode === 'list' && (
              <SkillsContainer>
                {displaySkills.map((skill, index) => (
                  <SkillTag key={index}>{skill}</SkillTag>
                ))}
                {skills.length > displaySkills.length && (
                  <SkillTag>+{skills.length - displaySkills.length} more</SkillTag>
                )}
              </SkillsContainer>
            )}
          </CandidateDetails>
        </CandidateInfo>
        
        <ScoreContainer>
          <ScoreBadge color={scoreColor}>
            <Target size={16} />
            {(score * 100).toFixed(1)}%
          </ScoreBadge>
          <GradeBadge color={gradeColor}>
            {grade}
          </GradeBadge>
        </ScoreContainer>
      </CardHeader>

      {viewMode === 'grid' && (
        <CardBody viewMode={viewMode}>
          <ContactInfo>
            <ContactItem>
              <Mail size={16} />
              {email}
            </ContactItem>
            <ContactItem>
              <Phone size={16} />
              {phone}
            </ContactItem>
            <ExperienceInfo>
              <Calendar size={16} />
              {experienceYears} years experience
            </ExperienceInfo>
          </ContactInfo>

          <SkillsContainer>
            {displaySkills.map((skill, index) => (
              <SkillTag key={index}>{skill}</SkillTag>
            ))}
            {skills.length > displaySkills.length && (
              <SkillTag>+{skills.length - displaySkills.length} more</SkillTag>
            )}
          </SkillsContainer>

          <FeatureIndicators>
            <FeatureIndicator>
              <FeatureIcon color="#3b82f6">
                <Zap size={10} />
              </FeatureIcon>
              AI Matched
            </FeatureIndicator>
            {aiExplanation && (
              <FeatureIndicator>
                <FeatureIcon color="#10b981">
                  <Brain size={10} />
                </FeatureIcon>
                Explained
              </FeatureIndicator>
            )}
            <FeatureIndicator>
              <FeatureIcon color="#f59e0b">
                <Shield size={10} />
              </FeatureIcon>
              Bias Checked
            </FeatureIndicator>
          </FeatureIndicators>

          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <ExpandButton onClick={() => setIsExpanded(!isExpanded)}>
              {isExpanded ? <ChevronUp size={20} /> : <ChevronDown size={20} />}
              {isExpanded ? 'Less' : 'More'}
            </ExpandButton>
            
            {sourceURL && (
              <a 
                href={sourceURL} 
                target="_blank" 
                rel="noopener noreferrer"
                style={{ 
                  display: 'flex', 
                  alignItems: 'center', 
                  gap: '0.25rem',
                  color: '#3b82f6',
                  textDecoration: 'none',
                  fontSize: '0.875rem'
                }}
              >
                View Profile
                <ExternalLink size={16} />
              </a>
            )}
          </div>
        </CardBody>
      )}

      {isExpanded && (
        <ExpandedContent>
          {resumeText && (
            <ResumePreview>
              <h4 style={{ margin: '0 0 0.5rem 0', fontSize: '0.875rem', fontWeight: '600', color: '#1e293b' }}>
                Resume Preview
              </h4>
              <ResumeText>{resumeText}</ResumeText>
            </ResumePreview>
          )}

          {aiExplanation && (
            <AIExplanation>
              <AIExplanationTitle>
                <Brain size={16} />
                AI Explanation
              </AIExplanationTitle>
              <AIExplanationText>{aiExplanation}</AIExplanationText>
            </AIExplanation>
          )}

          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <div style={{ display: 'flex', gap: '1rem' }}>
              <button style={{
                background: '#3b82f6',
                color: 'white',
                border: 'none',
                borderRadius: '0.5rem',
                padding: '0.5rem 1rem',
                fontSize: '0.875rem',
                fontWeight: '500',
                cursor: 'pointer'
              }}>
                Contact Candidate
              </button>
              <button style={{
                background: 'white',
                color: '#374151',
                border: '1px solid #d1d5db',
                borderRadius: '0.5rem',
                padding: '0.5rem 1rem',
                fontSize: '0.875rem',
                fontWeight: '500',
                cursor: 'pointer'
              }}>
                Save to List
              </button>
            </div>
          </div>
        </ExpandedContent>
      )}
    </CardContainer>
  );
}

export default CandidateCard;
