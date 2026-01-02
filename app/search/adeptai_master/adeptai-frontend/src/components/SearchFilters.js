import React from 'react';
import styled from 'styled-components';
import { Filter, SortAsc, SortDesc } from 'lucide-react';

const FiltersContainer = styled.div`
  background: white;
  border: 1px solid #e5e7eb;
  border-radius: 0.75rem;
  padding: 1.5rem;
  box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
`;

const FiltersTitle = styled.h3`
  font-size: 1rem;
  font-weight: 600;
  color: #1e293b;
  margin: 0 0 1.5rem 0;
  display: flex;
  align-items: center;
  gap: 0.5rem;
`;

const FilterSection = styled.div`
  margin-bottom: 1.5rem;

  &:last-child {
    margin-bottom: 0;
  }
`;

const FilterLabel = styled.label`
  display: block;
  font-size: 0.875rem;
  font-weight: 500;
  color: #374151;
  margin-bottom: 0.5rem;
`;

const FilterSelect = styled.select`
  width: 100%;
  padding: 0.75rem;
  border: 1px solid #d1d5db;
  border-radius: 0.5rem;
  font-size: 0.875rem;
  color: #374151;
  background: white;
  cursor: pointer;
  transition: border-color 0.2s ease-in-out;

  &:focus {
    outline: none;
    border-color: #3b82f6;
    box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
  }
`;

const FilterCheckbox = styled.input`
  width: 1rem;
  height: 1rem;
  accent-color: #3b82f6;
  margin-right: 0.5rem;
`;

const FilterCheckboxLabel = styled.label`
  display: flex;
  align-items: center;
  font-size: 0.875rem;
  color: #374151;
  cursor: pointer;
  margin-bottom: 0.5rem;

  &:last-child {
    margin-bottom: 0;
  }
`;

const SortContainer = styled.div`
  display: flex;
  align-items: center;
  gap: 0.5rem;
  margin-bottom: 1rem;
`;

const SortLabel = styled.label`
  font-size: 0.875rem;
  font-weight: 500;
  color: #374151;
`;

const SortSelect = styled.select`
  flex: 1;
  padding: 0.75rem;
  border: 1px solid #d1d5db;
  border-radius: 0.5rem;
  font-size: 0.875rem;
  color: #374151;
  background: white;
  cursor: pointer;
  transition: border-color 0.2s ease-in-out;

  &:focus {
    outline: none;
    border-color: #3b82f6;
    box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
  }
`;

const SortIcon = styled.div`
  color: #6b7280;
  display: flex;
  align-items: center;
  justify-content: center;
`;

const ClearFiltersButton = styled.button`
  width: 100%;
  background: #f8fafc;
  color: #6b7280;
  border: 1px solid #e5e7eb;
  border-radius: 0.5rem;
  padding: 0.75rem;
  font-size: 0.875rem;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.2s ease-in-out;

  &:hover {
    background: #f1f5f9;
    color: #374151;
  }
`;

const FilterStats = styled.div`
  background: #f8fafc;
  border: 1px solid #e5e7eb;
  border-radius: 0.5rem;
  padding: 1rem;
  margin-bottom: 1.5rem;
`;

const FilterStatsTitle = styled.h4`
  font-size: 0.875rem;
  font-weight: 600;
  color: #1e293b;
  margin: 0 0 0.5rem 0;
`;

const FilterStatsText = styled.p`
  font-size: 0.75rem;
  color: #6b7280;
  margin: 0;
`;

function SearchFilters({ filters, onFilterChange, sortBy, onSortChange }) {
  const handleFilterChange = (filterName, value) => {
    onFilterChange({
      ...filters,
      [filterName]: value
    });
  };

  const handleSortChange = (value) => {
    onSortChange(value);
  };

  const clearFilters = () => {
    onFilterChange({
      domain: 'all',
      experience: 'all',
      score: 'all'
    });
    onSortChange('relevance');
  };

  const hasActiveFilters = Object.values(filters).some(value => value !== 'all') || sortBy !== 'relevance';

  return (
    <FiltersContainer>
      <FiltersTitle>
        <Filter size={20} />
        Filters & Sort
      </FiltersTitle>

      <FilterStats>
        <FilterStatsTitle>Active Filters</FilterStatsTitle>
        <FilterStatsText>
          {Object.entries(filters).filter(([_, value]) => value !== 'all').length} filters applied
        </FilterStatsText>
      </FilterStats>

      <FilterSection>
        <FilterLabel>Domain</FilterLabel>
        <FilterSelect
          value={filters.domain}
          onChange={(e) => handleFilterChange('domain', e.target.value)}
        >
          <option value="all">All Domains</option>
          <option value="technology">Technology</option>
          <option value="healthcare">Healthcare</option>
          <option value="general">General</option>
        </FilterSelect>
      </FilterSection>

      <FilterSection>
        <FilterLabel>Experience Level</FilterLabel>
        <FilterSelect
          value={filters.experience}
          onChange={(e) => handleFilterChange('experience', e.target.value)}
        >
          <option value="all">All Experience Levels</option>
          <option value="entry">Entry Level (0-2 years)</option>
          <option value="mid">Mid Level (3-5 years)</option>
          <option value="senior">Senior Level (5+ years)</option>
        </FilterSelect>
      </FilterSection>

      <FilterSection>
        <FilterLabel>Match Score</FilterLabel>
        <FilterSelect
          value={filters.score}
          onChange={(e) => handleFilterChange('score', e.target.value)}
        >
          <option value="all">All Scores</option>
          <option value="high">High (80%+)</option>
          <option value="medium">Medium (60-79%)</option>
          <option value="low">Low (Below 60%)</option>
        </FilterSelect>
      </FilterSection>

      <FilterSection>
        <SortContainer>
          <SortLabel>Sort by:</SortLabel>
          <SortSelect
            value={sortBy}
            onChange={(e) => handleSortChange(e.target.value)}
          >
            <option value="relevance">Relevance</option>
            <option value="score">Match Score</option>
            <option value="experience">Experience</option>
            <option value="name">Name</option>
          </SortSelect>
          <SortIcon>
            {sortBy === 'name' ? <SortAsc size={16} /> : <SortDesc size={16} />}
          </SortIcon>
        </SortContainer>
      </FilterSection>

      <FilterSection>
        <FilterCheckboxLabel>
          <FilterCheckbox
            type="checkbox"
            defaultChecked
          />
          Include AI Explanations
        </FilterCheckboxLabel>
        <FilterCheckboxLabel>
          <FilterCheckbox
            type="checkbox"
            defaultChecked
          />
          Bias Prevention
        </FilterCheckboxLabel>
        <FilterCheckboxLabel>
          <FilterCheckbox
            type="checkbox"
          />
          Behavioral Analysis
        </FilterCheckboxLabel>
      </FilterSection>

      {hasActiveFilters && (
        <ClearFiltersButton onClick={clearFilters}>
          Clear All Filters
        </ClearFiltersButton>
      )}
    </FiltersContainer>
  );
}

export default SearchFilters;
