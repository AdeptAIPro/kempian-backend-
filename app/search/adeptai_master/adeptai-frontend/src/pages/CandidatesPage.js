import React, { useEffect, useState } from 'react';
import styled from 'styled-components';
import { motion } from 'framer-motion';
import { Users, Search, Filter, Download, RefreshCw, Grid, List } from 'lucide-react';
import { useApp } from '../context/AppContext';
import CandidateCard from '../components/CandidateCard';
import LoadingSkeleton from '../components/LoadingSkeleton';
import SearchFilters from '../components/SearchFilters';

const CandidatesPageContainer = styled.div`
  display: flex;
  flex-direction: column;
  gap: 2rem;
`;

const PageHeader = styled.div`
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 1rem;

  @media (max-width: 768px) {
    flex-direction: column;
    gap: 1rem;
    align-items: flex-start;
  }
`;

const PageTitle = styled.h1`
  font-size: 2rem;
  font-weight: 700;
  color: #1e293b;
  margin: 0;
  display: flex;
  align-items: center;
  gap: 0.75rem;
`;

const PageActions = styled.div`
  display: flex;
  align-items: center;
  gap: 1rem;
`;

const ActionButton = styled.button`
  background: ${props => props.primary ? '#3b82f6' : 'white'};
  color: ${props => props.primary ? 'white' : '#374151'};
  border: 1px solid #e5e7eb;
  border-radius: 0.5rem;
  padding: 0.75rem 1rem;
  font-size: 0.875rem;
  font-weight: 500;
  cursor: pointer;
  display: flex;
  align-items: center;
  gap: 0.5rem;
  transition: all 0.2s ease-in-out;

  &:hover:not(:disabled) {
    background: ${props => props.primary ? '#2563eb' : '#f8fafc'};
    transform: translateY(-1px);
  }

  &:disabled {
    opacity: 0.6;
    cursor: not-allowed;
    transform: none;
  }
`;

const StatsContainer = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 1rem;
  margin-bottom: 2rem;
`;

const StatCard = styled.div`
  background: white;
  border: 1px solid #e5e7eb;
  border-radius: 0.75rem;
  padding: 1.5rem;
  box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
`;

const StatValue = styled.div`
  font-size: 2rem;
  font-weight: 700;
  color: #1e293b;
  margin-bottom: 0.5rem;
`;

const StatLabel = styled.div`
  font-size: 0.875rem;
  color: #6b7280;
  font-weight: 500;
`;

const MainContent = styled.div`
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

function CandidatesPage() {
  const { state, actions } = useApp();
  const { candidates, isCandidatesLoading } = state;
  const [viewMode, setViewMode] = useState('grid');
  const [filters, setFilters] = useState({
    domain: 'all',
    experience: 'all',
    score: 'all'
  });
  const [sortBy, setSortBy] = useState('name');

  useEffect(() => {
    // Load candidates if not already loaded
    if (candidates.length === 0) {
      actions.loadCandidates();
    }
  }, [actions, candidates.length]);

  const handleRefresh = () => {
    actions.loadCandidates();
  };

  const handleFilterChange = (newFilters) => {
    setFilters(newFilters);
  };

  const handleSortChange = (newSortBy) => {
    setSortBy(newSortBy);
  };

  const handleExport = () => {
    // Implement export functionality
    console.log('Export functionality not implemented yet');
  };

  const filteredCandidates = candidates.filter(candidate => {
    if (filters.domain !== 'all' && candidate.domain !== filters.domain) {
      return false;
    }
    if (filters.experience !== 'all') {
      const exp = candidate.experienceYears || 0;
      switch (filters.experience) {
        case 'entry': return exp <= 2;
        case 'mid': return exp > 2 && exp <= 5;
        case 'senior': return exp > 5;
        default: return true;
      }
    }
    if (filters.score !== 'all') {
      const score = candidate.score || 0;
      switch (filters.score) {
        case 'high': return score >= 0.8;
        case 'medium': return score >= 0.6 && score < 0.8;
        case 'low': return score < 0.6;
        default: return true;
      }
    }
    return true;
  });

  const sortedCandidates = [...filteredCandidates].sort((a, b) => {
    switch (sortBy) {
      case 'score':
        return (b.score || 0) - (a.score || 0);
      case 'experience':
        return (b.experienceYears || 0) - (a.experienceYears || 0);
      case 'name':
        return (a.fullName || '').localeCompare(b.fullName || '');
      default:
        return 0;
    }
  });

  const stats = {
    total: candidates.length,
    filtered: filteredCandidates.length,
    technology: candidates.filter(c => c.domain === 'technology').length,
    healthcare: candidates.filter(c => c.domain === 'healthcare').length
  };

  return (
    <CandidatesPageContainer>
      <PageHeader>
        <PageTitle>
          <Users size={32} />
          All Candidates
        </PageTitle>
        <PageActions>
          <ActionButton onClick={handleRefresh} disabled={isCandidatesLoading}>
            <RefreshCw size={16} />
            Refresh
          </ActionButton>
          <ActionButton onClick={handleExport}>
            <Download size={16} />
            Export
          </ActionButton>
        </PageActions>
      </PageHeader>

      <StatsContainer>
        <StatCard>
          <StatValue>{stats.total}</StatValue>
          <StatLabel>Total Candidates</StatLabel>
        </StatCard>
        <StatCard>
          <StatValue>{stats.filtered}</StatValue>
          <StatLabel>Filtered Results</StatLabel>
        </StatCard>
        <StatCard>
          <StatValue>{stats.technology}</StatValue>
          <StatLabel>Technology</StatLabel>
        </StatCard>
        <StatCard>
          <StatValue>{stats.healthcare}</StatValue>
          <StatLabel>Healthcare</StatLabel>
        </StatCard>
      </StatsContainer>

      <MainContent>
        <motion.div
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.6 }}
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
          transition={{ duration: 0.6, delay: 0.2 }}
        >
          <ResultsColumn>
            <ResultsHeader>
              <ResultsTitle>
                {filteredCandidates.length} Candidate{filteredCandidates.length !== 1 ? 's' : ''}
              </ResultsTitle>
              <ViewControls>
                <ViewButton
                  active={viewMode === 'grid'}
                  onClick={() => setViewMode('grid')}
                >
                  <Grid size={16} />
                </ViewButton>
                <ViewButton
                  active={viewMode === 'list'}
                  onClick={() => setViewMode('list')}
                >
                  <List size={16} />
                </ViewButton>
              </ViewControls>
            </ResultsHeader>

            {isCandidatesLoading ? (
              <LoadingSkeleton count={6} />
            ) : candidates.length === 0 ? (
              <EmptyState>
                <EmptyIcon>
                  <Users size={32} />
                </EmptyIcon>
                <EmptyTitle>No candidates found</EmptyTitle>
                <EmptyDescription>
                  No candidates are currently available. Try refreshing the data.
                </EmptyDescription>
                <ActionButton onClick={handleRefresh}>
                  <RefreshCw size={16} />
                  Refresh Data
                </ActionButton>
              </EmptyState>
            ) : filteredCandidates.length === 0 ? (
              <EmptyState>
                <EmptyIcon>
                  <Filter size={32} />
                </EmptyIcon>
                <EmptyTitle>No candidates match your filters</EmptyTitle>
                <EmptyDescription>
                  Try adjusting your filters to see more candidates.
                </EmptyDescription>
              </EmptyState>
            ) : (
              <>
                {viewMode === 'grid' ? (
                  <ResultsGrid>
                    {sortedCandidates.map((candidate, index) => (
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
                    {sortedCandidates.map((candidate, index) => (
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

                {filteredCandidates.length > 0 && (
                  <LoadMoreButton onClick={() => console.log('Load more')}>
                    Load More Candidates
                  </LoadMoreButton>
                )}
              </>
            )}
          </ResultsColumn>
        </motion.div>
      </MainContent>
    </CandidatesPageContainer>
  );
}

export default CandidatesPage;
