import React from 'react';
import styled from 'styled-components';
import { Search, Filter, Clock, Users, Zap } from 'lucide-react';

const HeaderContainer = styled.div`
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 1.5rem;
  padding: 1rem;
  background: #f8fafc;
  border-radius: 0.75rem;
  border: 1px solid #e5e7eb;

  @media (max-width: 768px) {
    flex-direction: column;
    gap: 1rem;
    align-items: flex-start;
  }
`;

const ResultsInfo = styled.div`
  display: flex;
  align-items: center;
  gap: 1rem;
  flex: 1;
  min-width: 0;

  @media (max-width: 768px) {
    width: 100%;
    justify-content: space-between;
  }
`;

const ResultsCount = styled.div`
  display: flex;
  align-items: center;
  gap: 0.5rem;
  font-size: 1.125rem;
  font-weight: 600;
  color: #1e293b;
`;

const ResultsCountNumber = styled.span`
  background: #3b82f6;
  color: white;
  padding: 0.25rem 0.75rem;
  border-radius: 0.375rem;
  font-size: 0.875rem;
  font-weight: 700;
`;

const SearchQuery = styled.div`
  font-size: 0.875rem;
  color: #6b7280;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  max-width: 300px;

  @media (max-width: 768px) {
    max-width: 200px;
  }
`;

const PerformanceInfo = styled.div`
  display: flex;
  align-items: center;
  gap: 1rem;
  font-size: 0.875rem;
  color: #6b7280;

  @media (max-width: 768px) {
    width: 100%;
    justify-content: space-between;
  }
`;

const PerformanceItem = styled.div`
  display: flex;
  align-items: center;
  gap: 0.25rem;
`;

const PerformanceIcon = styled.div`
  display: flex;
  align-items: center;
  justify-content: center;
`;

const LoadingIndicator = styled.div`
  display: flex;
  align-items: center;
  gap: 0.5rem;
  color: #3b82f6;
  font-size: 0.875rem;
  font-weight: 500;
`;

const LoadingSpinner = styled.div`
  width: 16px;
  height: 16px;
  border: 2px solid #e5e7eb;
  border-top: 2px solid #3b82f6;
  border-radius: 50%;
  animation: spin 1s linear infinite;

  @keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
  }
`;

const FilterBadge = styled.div`
  background: #3b82f6;
  color: white;
  padding: 0.25rem 0.75rem;
  border-radius: 0.375rem;
  font-size: 0.75rem;
  font-weight: 600;
  display: flex;
  align-items: center;
  gap: 0.25rem;
`;

const EmptyState = styled.div`
  text-align: center;
  padding: 2rem;
  color: #6b7280;
`;

const EmptyIcon = styled.div`
  width: 48px;
  height: 48px;
  border-radius: 50%;
  background: #f3f4f6;
  color: #9ca3af;
  display: flex;
  align-items: center;
  justify-content: center;
  margin: 0 auto 1rem;
`;

const EmptyTitle = styled.h3`
  font-size: 1.125rem;
  font-weight: 600;
  color: #374151;
  margin: 0 0 0.5rem 0;
`;

const EmptyDescription = styled.p`
  font-size: 0.875rem;
  color: #6b7280;
  margin: 0;
`;

function SearchResultsHeader({ 
  totalResults, 
  filteredResults, 
  isSearching, 
  searchQuery,
  searchTime,
  performance 
}) {
  if (isSearching) {
    return (
      <HeaderContainer>
        <LoadingIndicator>
          <LoadingSpinner />
          Searching for candidates...
        </LoadingIndicator>
      </HeaderContainer>
    );
  }

  if (totalResults === 0) {
    return (
      <EmptyState>
        <EmptyIcon>
          <Search size={24} />
        </EmptyIcon>
        <EmptyTitle>No results found</EmptyTitle>
        <EmptyDescription>
          Try adjusting your search terms or filters
        </EmptyDescription>
      </EmptyState>
    );
  }

  const hasFilters = filteredResults !== totalResults;
  const resultsText = hasFilters 
    ? `${filteredResults} of ${totalResults} results`
    : `${totalResults} result${totalResults !== 1 ? 's' : ''}`;

  return (
    <HeaderContainer>
      <ResultsInfo>
        <ResultsCount>
          <Users size={20} />
          <ResultsCountNumber>{filteredResults}</ResultsCountNumber>
          {resultsText}
        </ResultsCount>
        
        {searchQuery && (
          <SearchQuery>
            for "{searchQuery}"
          </SearchQuery>
        )}
      </ResultsInfo>

      <PerformanceInfo>
        {searchTime && (
          <PerformanceItem>
            <PerformanceIcon>
              <Clock size={16} />
            </PerformanceIcon>
            {searchTime}ms
          </PerformanceItem>
        )}
        
        <PerformanceItem>
          <PerformanceIcon>
            <Zap size={16} />
          </PerformanceIcon>
          AI-Powered
        </PerformanceItem>
        
        {hasFilters && (
          <FilterBadge>
            <Filter size={12} />
            Filtered
          </FilterBadge>
        )}
      </PerformanceInfo>
    </HeaderContainer>
  );
}

export default SearchResultsHeader;
