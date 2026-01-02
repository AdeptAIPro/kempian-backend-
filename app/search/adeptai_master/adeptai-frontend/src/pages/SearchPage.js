import React, { useState, useEffect } from 'react';
import styled from 'styled-components';
import { motion } from 'framer-motion';
import { Search, Filter, SortAsc, Grid, List, Download } from 'lucide-react';
import { useApp } from '../context/AppContext';
import SearchCard from '../components/SearchCard';
import CandidateCard from '../components/CandidateCard';
import SearchFilters from '../components/SearchFilters';
import SearchResultsHeader from '../components/SearchResultsHeader';
import LoadingSkeleton from '../components/LoadingSkeleton';

const SearchPageContainer = styled.div`
  display: flex;
  flex-direction: column;
  gap: 2rem;
`;

const SearchSection = styled.div`
  background: white;
  border: 1px solid #e5e7eb;
  border-radius: 1rem;
  padding: 2rem;
  box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
`;

const ResultsSection = styled.div`
  display: flex;
  gap: 2rem;

  @media (max-width: 1024px) {
    flex-direction: column;
  }
`;

const FiltersColumn = styled.div`
  width: 300px;
  flex-shrink: 0;

  @media (max-width: 1024px) {
    width: 100%;
  }
`;

const ResultsColumn = styled.div`
  flex: 1;
  min-width: 0;
`;

const ResultsHeader = styled.div`
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 1.5rem;
  padding: 1rem;
  background: #f8fafc;
  border-radius: 0.75rem;
  border: 1px solid #e5e7eb;
`;

const ResultsTitle = styled.h2`
  font-size: 1.25rem;
  font-weight: 600;
  color: #1e293b;
  margin: 0;
`;

const ViewControls = styled.div`
  display: flex;
  align-items: center;
  gap: 0.5rem;
`;

const ViewButton = styled.button`
  background: ${props => props.active ? '#3b82f6' : 'white'};
  color: ${props => props.active ? 'white' : '#6b7280'};
  border: 1px solid #e5e7eb;
  border-radius: 0.5rem;
  padding: 0.5rem;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: all 0.2s ease-in-out;

  &:hover {
    background: ${props => props.active ? '#2563eb' : '#f8fafc'};
  }
`;

const ResultsGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(400px, 1fr));
  gap: 1.5rem;

  @media (max-width: 768px) {
    grid-template-columns: 1fr;
  }
`;

const ResultsList = styled.div`
  display: flex;
  flex-direction: column;
  gap: 1rem;
`;

const EmptyState = styled.div`
  text-align: center;
  padding: 4rem 2rem;
  color: #6b7280;
`;

const EmptyIcon = styled.div`
  width: 64px;
  height: 64px;
  border-radius: 50%;
  background: #f3f4f6;
  color: #9ca3af;
  display: flex;
  align-items: center;
  justify-content: center;
  margin: 0 auto 1rem;
`;

const EmptyTitle = styled.h3`
  font-size: 1.25rem;
  font-weight: 600;
  color: #374151;
  margin: 0 0 0.5rem 0;
`;

const EmptyDescription = styled.p`
  font-size: 1rem;
  color: #6b7280;
  margin: 0 0 1.5rem 0;
`;

const LoadMoreButton = styled.button`
  background: #3b82f6;
  color: white;
  border: none;
  border-radius: 0.75rem;
  padding: 1rem 2rem;
  font-size: 1rem;
  font-weight: 600;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 0.5rem;
  margin: 2rem auto 0;
  transition: all 0.2s ease-in-out;

  &:hover:not(:disabled) {
    background: #2563eb;
    transform: translateY(-2px);
  }

  &:disabled {
    opacity: 0.6;
    cursor: not-allowed;
    transform: none;
  }
`;

function SearchPage() {
  const { state, actions } = useApp();
  const { searchQuery, searchResults, isSearching, settings } = state;
  const [viewMode, setViewMode] = useState('grid');
  const [sortBy, setSortBy] = useState('relevance');
  const [filters, setFilters] = useState({
    domain: 'all',
    experience: 'all',
    score: 'all'
  });

  useEffect(() => {
    // Load candidates if not already loaded
    if (state.candidates.length === 0) {
      actions.loadCandidates();
    }
  }, [actions, state.candidates.length]);

  const handleSearch = (query) => {
    actions.performSearch(query);
  };

  const handleFilterChange = (newFilters) => {
    setFilters(newFilters);
    // Apply filters to current results
    // This would typically trigger a new search with filters
  };

  const handleSortChange = (newSortBy) => {
    setSortBy(newSortBy);
    // Apply sorting to current results
  };

  const handleLoadMore = () => {
    // Implement load more functionality
    console.log('Load more functionality not implemented yet');
  };

  const filteredResults = searchResults.filter(result => {
    if (filters.domain !== 'all' && result.domain !== filters.domain) {
      return false;
    }
    if (filters.experience !== 'all') {
      const exp = result.experienceYears || 0;
      switch (filters.experience) {
        case 'entry': return exp <= 2;
        case 'mid': return exp > 2 && exp <= 5;
        case 'senior': return exp > 5;
        default: return true;
      }
    }
    if (filters.score !== 'all') {
      const score = result.score || 0;
      switch (filters.score) {
        case 'high': return score >= 0.8;
        case 'medium': return score >= 0.6 && score < 0.8;
        case 'low': return score < 0.6;
        default: return true;
      }
    }
    return true;
  });

  const sortedResults = [...filteredResults].sort((a, b) => {
    switch (sortBy) {
      case 'score':
        return (b.score || 0) - (a.score || 0);
      case 'experience':
        return (b.experienceYears || 0) - (a.experienceYears || 0);
      case 'name':
        return (a.fullName || '').localeCompare(b.fullName || '');
      default:
        return 0; // Keep original order for relevance
    }
  });

  return (
    <SearchPageContainer>
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
      >
        <SearchSection>
          <SearchCard onSearch={handleSearch} />
        </SearchSection>
      </motion.div>

      <ResultsSection>
        <motion.div
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.6, delay: 0.2 }}
        >
          <FiltersColumn>
            <SearchFilters
              filters={filters}
              onFilterChange={handleFilterChange}
              sortBy={sortBy}
              onSortChange={handleSortChange}
            />
          </FiltersColumn>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.6, delay: 0.3 }}
        >
          <ResultsColumn>
            <SearchResultsHeader
              totalResults={searchResults.length}
              filteredResults={filteredResults.length}
              isSearching={isSearching}
              searchQuery={searchQuery}
            />

            {isSearching ? (
              <LoadingSkeleton count={6} />
            ) : searchResults.length === 0 ? (
              <EmptyState>
                <EmptyIcon>
                  <Search size={32} />
                </EmptyIcon>
                <EmptyTitle>No search results</EmptyTitle>
                <EmptyDescription>
                  Try searching for candidates using different keywords or adjust your filters
                </EmptyDescription>
              </EmptyState>
            ) : (
              <>
                {viewMode === 'grid' ? (
                  <ResultsGrid>
                    {sortedResults.map((candidate, index) => (
                      <motion.div
                        key={candidate.id || index}
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ duration: 0.4, delay: index * 0.1 }}
                      >
                        <CandidateCard candidate={candidate} viewMode="grid" />
                      </motion.div>
                    ))}
                  </ResultsGrid>
                ) : (
                  <ResultsList>
                    {sortedResults.map((candidate, index) => (
                      <motion.div
                        key={candidate.id || index}
                        initial={{ opacity: 0, x: 20 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ duration: 0.4, delay: index * 0.05 }}
                      >
                        <CandidateCard candidate={candidate} viewMode="list" />
                      </motion.div>
                    ))}
                  </ResultsList>
                )}

                {filteredResults.length > 0 && (
                  <LoadMoreButton onClick={handleLoadMore}>
                    Load More Results
                  </LoadMoreButton>
                )}
              </>
            )}
          </ResultsColumn>
        </motion.div>
      </ResultsSection>
    </SearchPageContainer>
  );
}

export default SearchPage;
