import React from 'react';
import styled from 'styled-components';
import { Search, Bell, Settings, User, Activity, Wifi, WifiOff } from 'lucide-react';
import { useApp } from '../context/AppContext';

const HeaderContainer = styled.header`
  background: white;
  border-bottom: 1px solid #e5e7eb;
  padding: 1rem 2rem;
  display: flex;
  align-items: center;
  justify-content: space-between;
  position: sticky;
  top: 0;
  z-index: 50;
  box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);

  @media (max-width: 768px) {
    padding: 1rem;
  }
`;

const LeftSection = styled.div`
  display: flex;
  align-items: center;
  gap: 1rem;
`;

const Logo = styled.div`
  display: flex;
  align-items: center;
  gap: 0.5rem;
  font-size: 1.5rem;
  font-weight: 700;
  color: #1e293b;
`;

const SearchContainer = styled.div`
  position: relative;
  display: flex;
  align-items: center;
  background: #f8fafc;
  border: 1px solid #e5e7eb;
  border-radius: 0.5rem;
  padding: 0.5rem 1rem;
  min-width: 300px;

  @media (max-width: 768px) {
    min-width: 200px;
  }
`;

const SearchInput = styled.input`
  border: none;
  background: transparent;
  outline: none;
  flex: 1;
  font-size: 0.875rem;
  color: #374151;

  &::placeholder {
    color: #9ca3af;
  }
`;

const SearchButton = styled.button`
  background: none;
  border: none;
  color: #6b7280;
  cursor: pointer;
  padding: 0.25rem;
  display: flex;
  align-items: center;
  justify-content: center;

  &:hover {
    color: #3b82f6;
  }
`;

const RightSection = styled.div`
  display: flex;
  align-items: center;
  gap: 1rem;
`;

const StatusIndicator = styled.div`
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.5rem 1rem;
  border-radius: 0.5rem;
  font-size: 0.875rem;
  font-weight: 500;
  background: ${props => {
    switch (props.status) {
      case 'healthy': return '#d1fae5';
      case 'degraded': return '#fef3c7';
      case 'unhealthy': return '#fee2e2';
      default: return '#f3f4f6';
    }
  }};
  color: ${props => {
    switch (props.status) {
      case 'healthy': return '#065f46';
      case 'degraded': return '#92400e';
      case 'unhealthy': return '#991b1b';
      default: return '#374151';
    }
  }};
`;

const IconButton = styled.button`
  background: none;
  border: none;
  color: #6b7280;
  cursor: pointer;
  padding: 0.5rem;
  border-radius: 0.5rem;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: all 0.2s ease-in-out;

  &:hover {
    background: #f3f4f6;
    color: #374151;
  }
`;

const NotificationBadge = styled.span`
  position: absolute;
  top: -0.25rem;
  right: -0.25rem;
  background: #ef4444;
  color: white;
  border-radius: 50%;
  width: 1.25rem;
  height: 1.25rem;
  font-size: 0.75rem;
  display: flex;
  align-items: center;
  justify-content: center;
  font-weight: 600;
`;

function Header() {
  const { state, actions } = useApp();
  const { searchQuery, systemHealth } = state;

  const handleSearch = (e) => {
    e.preventDefault();
    if (searchQuery.trim()) {
      actions.performSearch(searchQuery);
    }
  };

  const getStatusIcon = () => {
    if (systemHealth.status === 'healthy') {
      return <Wifi size={16} />;
    }
    return <WifiOff size={16} />;
  };

  const getStatusText = () => {
    switch (systemHealth.status) {
      case 'healthy': return 'System Healthy';
      case 'degraded': return 'System Degraded';
      case 'unhealthy': return 'System Unhealthy';
      default: return 'Checking Status...';
    }
  };

  return (
    <HeaderContainer>
      <LeftSection>
        <Logo>
          <Activity size={24} />
          AdeptAI
        </Logo>
        
        <SearchContainer>
          <form onSubmit={handleSearch} style={{ display: 'flex', alignItems: 'center', width: '100%' }}>
            <SearchInput
              type="text"
              placeholder="Search candidates..."
              value={searchQuery}
              onChange={(e) => actions.setSearchQuery(e.target.value)}
            />
            <SearchButton type="submit">
              <Search size={16} />
            </SearchButton>
          </form>
        </SearchContainer>
      </LeftSection>

      <RightSection>
        <StatusIndicator status={systemHealth.status}>
          {getStatusIcon()}
          {getStatusText()}
        </StatusIndicator>

        <div style={{ position: 'relative' }}>
          <IconButton>
            <Bell size={20} />
            <NotificationBadge>3</NotificationBadge>
          </IconButton>
        </div>

        <IconButton onClick={() => actions.setCurrentPage('settings')}>
          <Settings size={20} />
        </IconButton>

        <IconButton>
          <User size={20} />
        </IconButton>
      </RightSection>
    </HeaderContainer>
  );
}

export default Header;
