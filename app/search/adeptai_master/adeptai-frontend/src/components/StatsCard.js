import React from 'react';
import styled from 'styled-components';
import { TrendingUp, TrendingDown } from 'lucide-react';

const StatsCardContainer = styled.div`
  background: white;
  border: 1px solid #e5e7eb;
  border-radius: 0.75rem;
  padding: 1.5rem;
  box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
  transition: all 0.2s ease-in-out;

  &:hover {
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    transform: translateY(-2px);
  }
`;

const StatsHeader = styled.div`
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 1rem;
`;

const StatsTitle = styled.h3`
  font-size: 0.875rem;
  font-weight: 500;
  color: #6b7280;
  margin: 0;
`;

const StatsIcon = styled.div`
  width: 40px;
  height: 40px;
  border-radius: 0.75rem;
  background: ${props => props.color}20;
  color: ${props => props.color};
  display: flex;
  align-items: center;
  justify-content: center;
`;

const StatsValue = styled.div`
  font-size: 2rem;
  font-weight: 700;
  color: #1e293b;
  margin-bottom: 0.5rem;
`;

const StatsChange = styled.div`
  display: flex;
  align-items: center;
  gap: 0.25rem;
  font-size: 0.875rem;
  font-weight: 500;
  color: ${props => {
    switch (props.changeType) {
      case 'positive': return '#059669';
      case 'negative': return '#dc2626';
      default: return '#6b7280';
    }
  }};
`;

const ChangeIcon = styled.div`
  display: flex;
  align-items: center;
  justify-content: center;
`;

function StatsCard({ title, value, change, changeType, icon: Icon, color }) {
  return (
    <StatsCardContainer>
      <StatsHeader>
        <StatsTitle>{title}</StatsTitle>
        <StatsIcon color={color}>
          <Icon size={20} />
        </StatsIcon>
      </StatsHeader>
      
      <StatsValue>{value}</StatsValue>
      
      <StatsChange changeType={changeType}>
        <ChangeIcon>
          {changeType === 'positive' ? (
            <TrendingUp size={16} />
          ) : changeType === 'negative' ? (
            <TrendingDown size={16} />
          ) : null}
        </ChangeIcon>
        {change}
      </StatsChange>
    </StatsCardContainer>
  );
}

export default StatsCard;
