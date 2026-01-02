import React, { useEffect } from 'react';
import styled from 'styled-components';
import { motion } from 'framer-motion';
import { 
  Search, 
  Users, 
  TrendingUp, 
  Activity, 
  Brain, 
  Shield,
  Clock,
  Target,
  Zap
} from 'lucide-react';
import { useApp } from '../context/AppContext';
import SearchCard from '../components/SearchCard';
import StatsCard from '../components/StatsCard';
import RecentSearches from '../components/RecentSearches';
import SystemStatus from '../components/SystemStatus';
import QuickActions from '../components/QuickActions';

const DashboardContainer = styled.div`
  display: flex;
  flex-direction: column;
  gap: 2rem;
`;

const WelcomeSection = styled.div`
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  padding: 2rem;
  border-radius: 1rem;
  margin-bottom: 1rem;
`;

const WelcomeTitle = styled.h1`
  font-size: 2rem;
  font-weight: 700;
  margin: 0 0 0.5rem 0;
`;

const WelcomeSubtitle = styled.p`
  font-size: 1.125rem;
  opacity: 0.9;
  margin: 0;
`;

const GridContainer = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: 1.5rem;
  margin-bottom: 2rem;
`;

const StatsGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: 1rem;
  margin-bottom: 2rem;
`;

const SectionTitle = styled.h2`
  font-size: 1.5rem;
  font-weight: 600;
  color: #1e293b;
  margin: 0 0 1rem 0;
`;

const TwoColumnGrid = styled.div`
  display: grid;
  grid-template-columns: 2fr 1fr;
  gap: 2rem;

  @media (max-width: 1024px) {
    grid-template-columns: 1fr;
  }
`;

const QuickStats = [
  {
    title: 'Total Candidates',
    value: '1,247',
    change: '+12%',
    changeType: 'positive',
    icon: Users,
    color: '#3b82f6'
  },
  {
    title: 'Searches Today',
    value: '89',
    change: '+23%',
    changeType: 'positive',
    icon: Search,
    color: '#10b981'
  },
  {
    title: 'Avg Response Time',
    value: '245ms',
    change: '-15%',
    changeType: 'positive',
    icon: Clock,
    color: '#f59e0b'
  },
  {
    title: 'Match Accuracy',
    value: '94.2%',
    change: '+2.1%',
    changeType: 'positive',
    icon: Target,
    color: '#8b5cf6'
  }
];

const FeatureCards = [
  {
    title: 'Domain-Aware Search',
    description: 'Intelligent search that understands healthcare vs technology domains',
    icon: Brain,
    color: '#3b82f6',
    status: 'active'
  },
  {
    title: 'Explainable AI',
    description: 'Get detailed explanations for why candidates match your criteria',
    icon: Brain,
    color: '#10b981',
    status: 'beta'
  },
  {
    title: 'Bias Prevention',
    description: 'Advanced algorithms to prevent unconscious bias in hiring',
    icon: Shield,
    color: '#f59e0b',
    status: 'beta'
  },
  {
    title: 'Behavioral Analysis',
    description: 'Analyze candidate personality and work preferences',
    icon: Activity,
    color: '#8b5cf6',
    status: 'experimental'
  }
];

function Dashboard() {
  const { state, actions } = useApp();
  const { systemHealth, performance } = state;

  useEffect(() => {
    // Load initial data
    actions.refreshSystemHealth();
  }, [actions]);

  return (
    <DashboardContainer>
      <WelcomeSection>
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
        >
          <WelcomeTitle>Welcome to AdeptAI</WelcomeTitle>
          <WelcomeSubtitle>
            Your intelligent recruitment platform powered by advanced AI
          </WelcomeSubtitle>
        </motion.div>
      </WelcomeSection>

      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, delay: 0.2 }}
      >
        <SectionTitle>Quick Search</SectionTitle>
        <SearchCard />
      </motion.div>

      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, delay: 0.3 }}
      >
        <SectionTitle>Performance Overview</SectionTitle>
        <StatsGrid>
          {QuickStats.map((stat, index) => (
            <StatsCard
              key={index}
              title={stat.title}
              value={stat.value}
              change={stat.change}
              changeType={stat.changeType}
              icon={stat.icon}
              color={stat.color}
            />
          ))}
        </StatsGrid>
      </motion.div>

      <TwoColumnGrid>
        <motion.div
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.6, delay: 0.4 }}
        >
          <SectionTitle>Recent Searches</SectionTitle>
          <RecentSearches />
        </motion.div>

        <motion.div
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.6, delay: 0.5 }}
        >
          <SectionTitle>System Status</SectionTitle>
          <SystemStatus />
        </motion.div>
      </TwoColumnGrid>

      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, delay: 0.6 }}
      >
        <SectionTitle>AI Features</SectionTitle>
        <GridContainer>
          {FeatureCards.map((feature, index) => (
            <motion.div
              key={index}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.7 + index * 0.1 }}
            >
              <FeatureCard>
                <FeatureIcon color={feature.color}>
                  <feature.icon size={24} />
                </FeatureIcon>
                <FeatureContent>
                  <FeatureTitle>{feature.title}</FeatureTitle>
                  <FeatureDescription>{feature.description}</FeatureDescription>
                  <FeatureStatus status={feature.status}>
                    {feature.status.toUpperCase()}
                  </FeatureStatus>
                </FeatureContent>
              </FeatureCard>
            </motion.div>
          ))}
        </GridContainer>
      </motion.div>

      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, delay: 0.8 }}
      >
        <SectionTitle>Quick Actions</SectionTitle>
        <QuickActions />
      </motion.div>
    </DashboardContainer>
  );
}

const FeatureCard = styled.div`
  background: white;
  border: 1px solid #e5e7eb;
  border-radius: 0.75rem;
  padding: 1.5rem;
  display: flex;
  align-items: flex-start;
  gap: 1rem;
  transition: all 0.2s ease-in-out;
  box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);

  &:hover {
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    transform: translateY(-2px);
  }
`;

const FeatureIcon = styled.div`
  width: 48px;
  height: 48px;
  border-radius: 0.75rem;
  background: ${props => props.color}20;
  color: ${props => props.color};
  display: flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
`;

const FeatureContent = styled.div`
  flex: 1;
`;

const FeatureTitle = styled.h3`
  font-size: 1.125rem;
  font-weight: 600;
  color: #1e293b;
  margin: 0 0 0.5rem 0;
`;

const FeatureDescription = styled.p`
  font-size: 0.875rem;
  color: #6b7280;
  margin: 0 0 1rem 0;
  line-height: 1.5;
`;

const FeatureStatus = styled.span`
  display: inline-block;
  padding: 0.25rem 0.75rem;
  border-radius: 0.375rem;
  font-size: 0.75rem;
  font-weight: 600;
  background: ${props => {
    switch (props.status) {
      case 'active': return '#d1fae5';
      case 'beta': return '#fef3c7';
      case 'experimental': return '#e0e7ff';
      default: return '#f3f4f6';
    }
  }};
  color: ${props => {
    switch (props.status) {
      case 'active': return '#065f46';
      case 'beta': return '#92400e';
      case 'experimental': return '#3730a3';
      default: return '#374151';
    }
  }};
`;

export default Dashboard;
