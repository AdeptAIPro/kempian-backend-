import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import styled from 'styled-components';
import { Search, Filter, Zap, Brain, Shield } from 'lucide-react';
import { useApp } from '../context/AppContext';

const SearchCardContainer = styled.div`
  background: white;
  border: 1px solid #e5e7eb;
  border-radius: 1rem;
  padding: 2rem;
  box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
`;

const SearchForm = styled.form`
  display: flex;
  gap: 1rem;
  margin-bottom: 1.5rem;

  @media (max-width: 768px) {
    flex-direction: column;
  }
`;

const SearchInput = styled.input`
  flex: 1;
  padding: 1rem;
  border: 2px solid #e5e7eb;
  border-radius: 0.75rem;
  font-size: 1rem;
  transition: all 0.2s ease-in-out;

  &:focus {
    outline: none;
    border-color: #3b82f6;
    box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
  }

  &::placeholder {
    color: #9ca3af;
  }
`;

const SearchButton = styled.button`
  background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
  color: white;
  border: none;
  border-radius: 0.75rem;
  padding: 1rem 2rem;
  font-size: 1rem;
  font-weight: 600;
  cursor: pointer;
  display: flex;
  align-items: center;
  gap: 0.5rem;
  transition: all 0.2s ease-in-out;
  min-width: 140px;
  justify-content: center;

  &:hover:not(:disabled) {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4);
  }

  &:disabled {
    opacity: 0.6;
    cursor: not-allowed;
    transform: none;
  }
`;

const AdvancedOptions = styled.div`
  display: flex;
  flex-wrap: wrap;
  gap: 1rem;
  margin-bottom: 1.5rem;
`;

const OptionToggle = styled.label`
  display: flex;
  align-items: center;
  gap: 0.5rem;
  cursor: pointer;
  font-size: 0.875rem;
  color: #374151;
  user-select: none;
`;

const Checkbox = styled.input`
  width: 1rem;
  height: 1rem;
  accent-color: #3b82f6;
`;

const QuickSearchButtons = styled.div`
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
`;

const QuickButton = styled.button`
  background: #f8fafc;
  border: 1px solid #e5e7eb;
  border-radius: 0.5rem;
  padding: 0.5rem 1rem;
  font-size: 0.875rem;
  color: #374151;
  cursor: pointer;
  transition: all 0.2s ease-in-out;

  &:hover {
    background: #e5e7eb;
    border-color: #d1d5db;
  }
`;

const FeatureIndicators = styled.div`
  display: flex;
  align-items: center;
  gap: 1rem;
  margin-top: 1rem;
  padding-top: 1rem;
  border-top: 1px solid #e5e7eb;
`;

const FeatureIndicator = styled.div`
  display: flex;
  align-items: center;
  gap: 0.5rem;
  font-size: 0.875rem;
  color: #6b7280;
`;

const FeatureIcon = styled.div`
  width: 20px;
  height: 20px;
  border-radius: 50%;
  background: ${props => props.color}20;
  color: ${props => props.color};
  display: flex;
  align-items: center;
  justify-content: center;
`;

const quickSearches = [
  'Senior Python Developer',
  'React Frontend Engineer',
  'Registered Nurse ICU',
  'Full Stack Developer',
  'DevOps Engineer',
  'Data Scientist',
  'Product Manager',
  'UX Designer'
];

function SearchCard() {
  const { state, actions } = useApp();
  const { searchQuery, isSearching, settings } = state;
  const [localQuery, setLocalQuery] = useState(searchQuery);
  const navigate = useNavigate();

  const handleSubmit = (e) => {
    e.preventDefault();
    if (localQuery.trim()) {
      actions.performSearch(localQuery);
      // Navigate to search page to show results
      navigate('/search');
    }
  };

  const handleQuickSearch = (query) => {
    setLocalQuery(query);
    actions.performSearch(query);
    // Navigate to search page to show results
    navigate('/search');
  };

  const handleSettingsChange = (setting, value) => {
    actions.updateSettings({ [setting]: value });
  };

  return (
    <SearchCardContainer>
      <SearchForm onSubmit={handleSubmit}>
        <SearchInput
          type="text"
          placeholder="Search for candidates... (e.g., 'Senior Python Developer with AWS experience')"
          value={localQuery}
          onChange={(e) => setLocalQuery(e.target.value)}
          disabled={isSearching}
        />
        <SearchButton type="submit" disabled={isSearching || !localQuery.trim()}>
          {isSearching ? (
            <>
              <div className="loading" />
              Searching...
            </>
          ) : (
            <>
              <Search size={20} />
              Search
            </>
          )}
        </SearchButton>
      </SearchForm>

      <AdvancedOptions>
        <OptionToggle>
          <Checkbox
            type="checkbox"
            checked={settings.domainFiltering}
            onChange={(e) => handleSettingsChange('domainFiltering', e.target.checked)}
          />
          Domain-Aware Search
        </OptionToggle>
        
        <OptionToggle>
          <Checkbox
            type="checkbox"
            checked={settings.includeExplainableAI}
            onChange={(e) => handleSettingsChange('includeExplainableAI', e.target.checked)}
          />
          Explainable AI
        </OptionToggle>
        
        <OptionToggle>
          <Checkbox
            type="checkbox"
            checked={settings.includeBehavioralAnalysis}
            onChange={(e) => handleSettingsChange('includeBehavioralAnalysis', e.target.checked)}
          />
          Behavioral Analysis
        </OptionToggle>
      </AdvancedOptions>

      <div>
        <h4 style={{ margin: '0 0 0.75rem 0', fontSize: '0.875rem', fontWeight: '600', color: '#374151' }}>
          Quick Searches
        </h4>
        <QuickSearchButtons>
          {quickSearches.map((query, index) => (
            <QuickButton
              key={index}
              onClick={() => handleQuickSearch(query)}
              disabled={isSearching}
            >
              {query}
            </QuickButton>
          ))}
        </QuickSearchButtons>
      </div>

      <FeatureIndicators>
        <FeatureIndicator>
          <FeatureIcon color="#3b82f6">
            <Zap size={12} />
          </FeatureIcon>
          Ultra-Fast Search
        </FeatureIndicator>
        
        <FeatureIndicator>
          <FeatureIcon color="#10b981">
            <Brain size={12} />
          </FeatureIcon>
          AI-Powered Matching
        </FeatureIndicator>
        
        <FeatureIndicator>
          <FeatureIcon color="#f59e0b">
            <Shield size={12} />
          </FeatureIcon>
          Bias Prevention
        </FeatureIndicator>
      </FeatureIndicators>
    </SearchCardContainer>
  );
}

export default SearchCard;
