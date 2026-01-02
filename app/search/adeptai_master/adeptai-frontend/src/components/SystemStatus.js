import React from 'react';
import styled from 'styled-components';
import { 
  CheckCircle, 
  AlertCircle, 
  XCircle, 
  Clock, 
  Activity,
  Database,
  Brain,
  Shield,
  Zap
} from 'lucide-react';
import { useApp } from '../context/AppContext';

const SystemStatusContainer = styled.div`
  background: white;
  border: 1px solid #e5e7eb;
  border-radius: 0.75rem;
  overflow: hidden;
  box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
`;

const StatusHeader = styled.div`
  padding: 1.5rem;
  border-bottom: 1px solid #e5e7eb;
  background: #f8fafc;
`;

const StatusTitle = styled.h3`
  font-size: 1rem;
  font-weight: 600;
  color: #1e293b;
  margin: 0 0 0.5rem 0;
  display: flex;
  align-items: center;
  gap: 0.5rem;
`;

const StatusIndicator = styled.div`
  display: inline-flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.25rem 0.75rem;
  border-radius: 0.375rem;
  font-size: 0.75rem;
  font-weight: 600;
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

const StatusIcon = styled.div`
  display: flex;
  align-items: center;
  justify-content: center;
`;

const LastChecked = styled.p`
  font-size: 0.75rem;
  color: #6b7280;
  margin: 0;
`;

const ComponentList = styled.div`
  padding: 1rem 1.5rem;
`;

const ComponentItem = styled.div`
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0.75rem 0;
  border-bottom: 1px solid #f3f4f6;

  &:last-child {
    border-bottom: none;
  }
`;

const ComponentInfo = styled.div`
  display: flex;
  align-items: center;
  gap: 0.75rem;
`;

const ComponentIcon = styled.div`
  width: 32px;
  height: 32px;
  border-radius: 0.5rem;
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
      case 'healthy': return '#059669';
      case 'degraded': return '#d97706';
      case 'unhealthy': return '#dc2626';
      default: return '#6b7280';
    }
  }};
  display: flex;
  align-items: center;
  justify-content: center;
`;

const ComponentDetails = styled.div`
  flex: 1;
`;

const ComponentName = styled.div`
  font-size: 0.875rem;
  font-weight: 500;
  color: #1e293b;
  margin-bottom: 0.25rem;
`;

const ComponentStatus = styled.div`
  font-size: 0.75rem;
  color: #6b7280;
`;

const ComponentStatusBadge = styled.span`
  display: inline-flex;
  align-items: center;
  gap: 0.25rem;
  padding: 0.125rem 0.5rem;
  border-radius: 0.25rem;
  font-size: 0.625rem;
  font-weight: 600;
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

const getStatusIcon = (status) => {
  switch (status) {
    case 'healthy': return <CheckCircle size={16} />;
    case 'degraded': return <AlertCircle size={16} />;
    case 'unhealthy': return <XCircle size={16} />;
    default: return <Clock size={16} />;
  }
};

const getComponentStatus = (component, overallStatus) => {
  if (component === 'enhanced_search' && overallStatus === 'healthy') return 'healthy';
  if (component === 'database' && overallStatus === 'healthy') return 'healthy';
  if (component === 'fallback_search') return 'healthy';
  return 'degraded';
};

const components = [
  {
    name: 'Search System',
    key: 'enhanced_search',
    icon: Zap,
    description: 'Core search functionality'
  },
  {
    name: 'Database',
    key: 'database',
    icon: Database,
    description: 'Candidate data storage'
  },
  {
    name: 'Explainable AI',
    key: 'explainable_ai',
    icon: Brain,
    description: 'AI explanations'
  },
  {
    name: 'Bias Prevention',
    key: 'bias_prevention',
    icon: Shield,
    description: 'Bias detection'
  },
  {
    name: 'Fallback System',
    key: 'fallback_search',
    icon: Activity,
    description: 'Backup search'
  }
];

function SystemStatus() {
  const { state, actions } = useApp();
  const { systemHealth } = state;

  const formatLastChecked = (timestamp) => {
    if (!timestamp) return 'Never';
    const date = new Date(timestamp);
    const now = new Date();
    const diffMs = now - date;
    const diffMins = Math.floor(diffMs / 60000);
    
    if (diffMins < 1) return 'Just now';
    if (diffMins < 60) return `${diffMins}m ago`;
    const diffHours = Math.floor(diffMins / 60);
    if (diffHours < 24) return `${diffHours}h ago`;
    const diffDays = Math.floor(diffHours / 24);
    return `${diffDays}d ago`;
  };

  return (
    <SystemStatusContainer>
      <StatusHeader>
        <StatusTitle>
          <Activity size={20} />
          System Status
        </StatusTitle>
        <StatusIndicator status={systemHealth.status}>
          <StatusIcon>
            {getStatusIcon(systemHealth.status)}
          </StatusIcon>
          {systemHealth.status?.toUpperCase() || 'UNKNOWN'}
        </StatusIndicator>
        <LastChecked>
          Last checked: {formatLastChecked(systemHealth.lastChecked)}
        </LastChecked>
      </StatusHeader>

      <ComponentList>
        {components.map((component, index) => {
          const Icon = component.icon;
          const componentStatus = getComponentStatus(component.key, systemHealth.status);
          
          return (
            <ComponentItem key={index}>
              <ComponentInfo>
                <ComponentIcon status={componentStatus}>
                  <Icon size={16} />
                </ComponentIcon>
                <ComponentDetails>
                  <ComponentName>{component.name}</ComponentName>
                  <ComponentStatus>{component.description}</ComponentStatus>
                </ComponentDetails>
              </ComponentInfo>
              <ComponentStatusBadge status={componentStatus}>
                {getStatusIcon(componentStatus)}
                {componentStatus.toUpperCase()}
              </ComponentStatusBadge>
            </ComponentItem>
          );
        })}
      </ComponentList>
    </SystemStatusContainer>
  );
}

export default SystemStatus;
