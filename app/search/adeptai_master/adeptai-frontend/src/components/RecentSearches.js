import React from 'react';
import styled from 'styled-components';
import { Clock, Search, Users, ArrowRight } from 'lucide-react';
import { useApp } from '../context/AppContext';

const RecentSearchesContainer = styled.div`
  background: white;
  border: 1px solid #e5e7eb;
  border-radius: 0.75rem;
  overflow: hidden;
  box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
`;

const SearchItem = styled.div`
  display: flex;
  align-items: center;
  gap: 1rem;
  padding: 1rem 1.5rem;
  border-bottom: 1px solid #f3f4f6;
  cursor: pointer;
  transition: all 0.2s ease-in-out;

  &:hover {
    background: #f8fafc;
  }

  &:last-child {
    border-bottom: none;
  }
`;

const SearchIcon = styled.div`
  width: 40px;
  height: 40px;
  border-radius: 0.75rem;
  background: #eff6ff;
  color: #3b82f6;
  display: flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
`;

const SearchContent = styled.div`
  flex: 1;
  min-width: 0;
`;

const SearchQuery = styled.div`
  font-size: 0.875rem;
  font-weight: 500;
  color: #1e293b;
  margin-bottom: 0.25rem;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
`;

const SearchMeta = styled.div`
  display: flex;
  align-items: center;
  gap: 1rem;
  font-size: 0.75rem;
  color: #6b7280;
`;

const SearchResultCount = styled.div`
  display: flex;
  align-items: center;
  gap: 0.25rem;
`;

const SearchTime = styled.div`
  display: flex;
  align-items: center;
  gap: 0.25rem;
`;

const SearchArrow = styled.div`
  color: #9ca3af;
  display: flex;
  align-items: center;
  justify-content: center;
`;

const EmptyState = styled.div`
  padding: 3rem 1.5rem;
  text-align: center;
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
  font-size: 1rem;
  font-weight: 500;
  color: #374151;
  margin: 0 0 0.5rem 0;
`;

const EmptyDescription = styled.p`
  font-size: 0.875rem;
  color: #6b7280;
  margin: 0;
`;

// Mock data for recent searches
const recentSearches = [
  {
    id: 1,
    query: 'Senior Python Developer with AWS experience',
    results: 23,
    timestamp: '2 minutes ago',
    domain: 'technology'
  },
  {
    id: 2,
    query: 'Registered Nurse ICU experience',
    results: 15,
    timestamp: '15 minutes ago',
    domain: 'healthcare'
  },
  {
    id: 3,
    query: 'React Frontend Developer',
    results: 31,
    timestamp: '1 hour ago',
    domain: 'technology'
  },
  {
    id: 4,
    query: 'DevOps Engineer Kubernetes',
    results: 18,
    timestamp: '2 hours ago',
    domain: 'technology'
  },
  {
    id: 5,
    query: 'Data Scientist Machine Learning',
    results: 27,
    timestamp: '3 hours ago',
    domain: 'technology'
  }
];

function RecentSearches() {
  const { actions } = useApp();

  const handleSearchClick = (query) => {
    actions.performSearch(query);
  };

  if (recentSearches.length === 0) {
    return (
      <RecentSearchesContainer>
        <EmptyState>
          <EmptyIcon>
            <Search size={24} />
          </EmptyIcon>
          <EmptyTitle>No recent searches</EmptyTitle>
          <EmptyDescription>
            Start searching for candidates to see your recent searches here
          </EmptyDescription>
        </EmptyState>
      </RecentSearchesContainer>
    );
  }

  return (
    <RecentSearchesContainer>
      {recentSearches.map((search) => (
        <SearchItem
          key={search.id}
          onClick={() => handleSearchClick(search.query)}
        >
          <SearchIcon>
            <Search size={20} />
          </SearchIcon>
          
          <SearchContent>
            <SearchQuery>{search.query}</SearchQuery>
            <SearchMeta>
              <SearchResultCount>
                <Users size={12} />
                {search.results} results
              </SearchResultCount>
              <SearchTime>
                <Clock size={12} />
                {search.timestamp}
              </SearchTime>
            </SearchMeta>
          </SearchContent>
          
          <SearchArrow>
            <ArrowRight size={16} />
          </SearchArrow>
        </SearchItem>
      ))}
    </RecentSearchesContainer>
  );
}

export default RecentSearches;
