import React, { useEffect, useState } from 'react';
import styled from 'styled-components';
import { motion } from 'framer-motion';
import { 
  Activity, 
  CheckCircle, 
  AlertCircle, 
  XCircle, 
  RefreshCw,
  Database,
  Brain,
  Shield,
  Zap,
  Clock,
  Server,
  Cpu,
  HardDrive
} from 'lucide-react';
import { useApp } from '../context/AppContext';

const HealthPageContainer = styled.div`
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

const RefreshButton = styled.button`
  background: #3b82f6;
  color: white;
  border: none;
  border-radius: 0.5rem;
  padding: 0.75rem 1rem;
  font-size: 0.875rem;
  font-weight: 600;
  cursor: pointer;
  display: flex;
  align-items: center;
  gap: 0.5rem;
  transition: all 0.2s ease-in-out;

  &:hover:not(:disabled) {
    background: #2563eb;
    transform: translateY(-1px);
  }

  &:disabled {
    opacity: 0.6;
    cursor: not-allowed;
    transform: none;
  }
`;

const StatusOverview = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: 1rem;
  margin-bottom: 2rem;
`;

const StatusCard = styled.div`
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

const StatusHeader = styled.div`
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 1rem;
`;

const StatusTitle = styled.h3`
  font-size: 1rem;
  font-weight: 600;
  color: #1e293b;
  margin: 0;
`;

const StatusIndicator = styled.div`
  display: flex;
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

const StatusValue = styled.div`
  font-size: 2rem;
  font-weight: 700;
  color: #1e293b;
  margin-bottom: 0.5rem;
`;

const StatusDescription = styled.div`
  font-size: 0.875rem;
  color: #6b7280;
`;

const ComponentsGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: 1.5rem;
  margin-bottom: 2rem;
`;

const ComponentCard = styled.div`
  background: white;
  border: 1px solid #e5e7eb;
  border-radius: 0.75rem;
  padding: 1.5rem;
  box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
`;

const ComponentHeader = styled.div`
  display: flex;
  align-items: center;
  gap: 1rem;
  margin-bottom: 1rem;
`;

const ComponentIcon = styled.div`
  width: 48px;
  height: 48px;
  border-radius: 0.75rem;
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

const ComponentInfo = styled.div`
  flex: 1;
`;

const ComponentName = styled.h4`
  font-size: 1rem;
  font-weight: 600;
  color: #1e293b;
  margin: 0 0 0.25rem 0;
`;

const ComponentStatus = styled.div`
  font-size: 0.875rem;
  color: #6b7280;
`;

const ComponentDetails = styled.div`
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
`;

const DetailItem = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-size: 0.875rem;
`;

const DetailLabel = styled.span`
  color: #6b7280;
`;

const DetailValue = styled.span`
  color: #1e293b;
  font-weight: 500;
`;

const MetricsGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 1rem;
  margin-bottom: 2rem;
`;

const MetricCard = styled.div`
  background: white;
  border: 1px solid #e5e7eb;
  border-radius: 0.75rem;
  padding: 1.5rem;
  box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
  text-align: center;
`;

const MetricValue = styled.div`
  font-size: 2rem;
  font-weight: 700;
  color: #1e293b;
  margin-bottom: 0.5rem;
`;

const MetricLabel = styled.div`
  font-size: 0.875rem;
  color: #6b7280;
  font-weight: 500;
`;

const getStatusIcon = (status) => {
  switch (status) {
    case 'healthy': return <CheckCircle size={16} />;
    case 'degraded': return <AlertCircle size={16} />;
    case 'unhealthy': return <XCircle size={16} />;
    default: return <Clock size={16} />;
  }
};

const getOverallStatus = (health) => {
  if (!health.components) return 'unknown';
  
  const componentStatuses = Object.values(health.components);
  if (componentStatuses.some(status => status === false)) return 'unhealthy';
  if (componentStatuses.some(status => status === 'degraded')) return 'degraded';
  return 'healthy';
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

const mockMetrics = {
  uptime: '99.9%',
  responseTime: '245ms',
  totalSearches: '1,247',
  errorRate: '0.1%',
  memoryUsage: '2.1GB',
  cpuUsage: '15%'
};

function HealthPage() {
  const { state, actions } = useApp();
  const { systemHealth } = state;
  const [isRefreshing, setIsRefreshing] = useState(false);

  useEffect(() => {
    // Initial health check
    actions.refreshSystemHealth();
  }, [actions]);

  const handleRefresh = async () => {
    setIsRefreshing(true);
    await actions.refreshSystemHealth();
    setIsRefreshing(false);
  };

  const overallStatus = getOverallStatus(systemHealth);
  const lastChecked = systemHealth.lastChecked 
    ? new Date(systemHealth.lastChecked).toLocaleString()
    : 'Never';

  return (
    <HealthPageContainer>
      <PageHeader>
        <PageTitle>
          <Activity size={32} />
          System Health
        </PageTitle>
        <RefreshButton onClick={handleRefresh} disabled={isRefreshing}>
          <RefreshCw size={16} className={isRefreshing ? 'animate-spin' : ''} />
          {isRefreshing ? 'Refreshing...' : 'Refresh'}
        </RefreshButton>
      </PageHeader>

      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
      >
        <StatusOverview>
          <StatusCard>
            <StatusHeader>
              <StatusTitle>Overall Status</StatusTitle>
              <StatusIndicator status={overallStatus}>
                {getStatusIcon(overallStatus)}
                {overallStatus.toUpperCase()}
              </StatusIndicator>
            </StatusHeader>
            <StatusValue>
              {overallStatus === 'healthy' ? 'All Systems Operational' :
               overallStatus === 'degraded' ? 'Degraded Performance' :
               overallStatus === 'unhealthy' ? 'System Issues' : 'Unknown'}
            </StatusValue>
            <StatusDescription>
              Last checked: {lastChecked}
            </StatusDescription>
          </StatusCard>

          <StatusCard>
            <StatusHeader>
              <StatusTitle>Search Performance</StatusTitle>
              <StatusIndicator status="healthy">
                <CheckCircle size={16} />
                EXCELLENT
              </StatusIndicator>
            </StatusHeader>
            <StatusValue>245ms</StatusValue>
            <StatusDescription>
              Average response time
            </StatusDescription>
          </StatusCard>

          <StatusCard>
            <StatusHeader>
              <StatusTitle>Uptime</StatusTitle>
              <StatusIndicator status="healthy">
                <CheckCircle size={16} />
                STABLE
              </StatusIndicator>
            </StatusHeader>
            <StatusValue>99.9%</StatusValue>
            <StatusDescription>
              System availability
            </StatusDescription>
          </StatusCard>

          <StatusCard>
            <StatusHeader>
              <StatusTitle>Error Rate</StatusTitle>
              <StatusIndicator status="healthy">
                <CheckCircle size={16} />
                LOW
              </StatusIndicator>
            </StatusHeader>
            <StatusValue>0.1%</StatusValue>
            <StatusDescription>
              Failed requests
            </StatusDescription>
          </StatusCard>
        </StatusOverview>
      </motion.div>

      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, delay: 0.2 }}
      >
        <ComponentsGrid>
          {components.map((component, index) => {
            const Icon = component.icon;
            const componentStatus = systemHealth.components?.[component.key] ? 'healthy' : 'unhealthy';
            
            return (
              <motion.div
                key={index}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: 0.3 + index * 0.1 }}
              >
                <ComponentCard>
                  <ComponentHeader>
                    <ComponentIcon status={componentStatus}>
                      <Icon size={24} />
                    </ComponentIcon>
                    <ComponentInfo>
                      <ComponentName>{component.name}</ComponentName>
                      <ComponentStatus>{component.description}</ComponentStatus>
                    </ComponentInfo>
                    <StatusIndicator status={componentStatus}>
                      {getStatusIcon(componentStatus)}
                      {componentStatus.toUpperCase()}
                    </StatusIndicator>
                  </ComponentHeader>
                  
                  <ComponentDetails>
                    <DetailItem>
                      <DetailLabel>Status</DetailLabel>
                      <DetailValue>
                        {componentStatus === 'healthy' ? 'Operational' : 'Offline'}
                      </DetailValue>
                    </DetailItem>
                    <DetailItem>
                      <DetailLabel>Response Time</DetailLabel>
                      <DetailValue>
                        {componentStatus === 'healthy' ? '< 100ms' : 'N/A'}
                      </DetailValue>
                    </DetailItem>
                    <DetailItem>
                      <DetailLabel>Last Check</DetailLabel>
                      <DetailValue>{lastChecked}</DetailValue>
                    </DetailItem>
                  </ComponentDetails>
                </ComponentCard>
              </motion.div>
            );
          })}
        </ComponentsGrid>
      </motion.div>

      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, delay: 0.4 }}
      >
        <h2 style={{ fontSize: '1.5rem', fontWeight: '600', color: '#1e293b', margin: '0 0 1rem 0' }}>
          System Metrics
        </h2>
        <MetricsGrid>
          {Object.entries(mockMetrics).map(([key, value], index) => (
            <motion.div
              key={key}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.5 + index * 0.1 }}
            >
              <MetricCard>
                <MetricValue>{value}</MetricValue>
                <MetricLabel>
                  {key.replace(/([A-Z])/g, ' $1').replace(/^./, str => str.toUpperCase())}
                </MetricLabel>
              </MetricCard>
            </motion.div>
          ))}
        </MetricsGrid>
      </motion.div>
    </HealthPageContainer>
  );
}

export default HealthPage;
