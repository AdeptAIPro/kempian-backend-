import React, { useState, useEffect } from 'react';
import styled from 'styled-components';
import { motion } from 'framer-motion';
import { useApp } from '../context/AppContext';
import { 
  BarChart3, 
  TrendingUp, 
  Users, 
  Search, 
  Clock, 
  Target,
  Brain,
  Shield,
  Zap
} from 'lucide-react';
import { 
  BarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  LineChart,
  Line
} from 'recharts';

const AnalyticsPageContainer = styled.div`
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

const TimeRangeSelector = styled.select`
  background: white;
  border: 1px solid #e5e7eb;
  border-radius: 0.5rem;
  padding: 0.75rem 1rem;
  font-size: 0.875rem;
  color: #374151;
  cursor: pointer;

  &:focus {
    outline: none;
    border-color: #3b82f6;
    box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
  }
`;

const StatsGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: 1rem;
  margin-bottom: 2rem;
`;

const StatCard = styled.div`
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

const StatHeader = styled.div`
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 1rem;
`;

const StatTitle = styled.h3`
  font-size: 0.875rem;
  font-weight: 500;
  color: #6b7280;
  margin: 0;
`;

const StatIcon = styled.div`
  width: 40px;
  height: 40px;
  border-radius: 0.75rem;
  background: ${props => props.color}20;
  color: ${props => props.color};
  display: flex;
  align-items: center;
  justify-content: center;
`;

const StatValue = styled.div`
  font-size: 2rem;
  font-weight: 700;
  color: #1e293b;
  margin-bottom: 0.5rem;
`;

const StatChange = styled.div`
  display: flex;
  align-items: center;
  gap: 0.25rem;
  font-size: 0.875rem;
  font-weight: 500;
  color: ${props => props.positive ? '#059669' : '#dc2626'};
`;

const ChartsGrid = styled.div`
  display: grid;
  grid-template-columns: 2fr 1fr;
  gap: 2rem;
  margin-bottom: 2rem;

  @media (max-width: 1024px) {
    grid-template-columns: 1fr;
  }
`;

const ChartCard = styled.div`
  background: white;
  border: 1px solid #e5e7eb;
  border-radius: 0.75rem;
  padding: 1.5rem;
  box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
`;

const ChartTitle = styled.h3`
  font-size: 1.125rem;
  font-weight: 600;
  color: #1e293b;
  margin: 0 0 1rem 0;
`;

const ChartContainer = styled.div`
  height: 300px;
  width: 100%;
`;

const FeatureAnalytics = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: 1.5rem;
`;

const FeatureCard = styled.div`
  background: white;
  border: 1px solid #e5e7eb;
  border-radius: 0.75rem;
  padding: 1.5rem;
  box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
`;

const FeatureHeader = styled.div`
  display: flex;
  align-items: center;
  gap: 1rem;
  margin-bottom: 1rem;
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
`;

const FeatureTitle = styled.h4`
  font-size: 1rem;
  font-weight: 600;
  color: #1e293b;
  margin: 0;
`;

const FeatureStats = styled.div`
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 1rem;
`;

const FeatureStat = styled.div`
  text-align: center;
`;

const FeatureStatValue = styled.div`
  font-size: 1.5rem;
  font-weight: 700;
  color: #1e293b;
  margin-bottom: 0.25rem;
`;

const FeatureStatLabel = styled.div`
  font-size: 0.75rem;
  color: #6b7280;
  font-weight: 500;
`;

// Mock data
const searchTrendsData = [
  { month: 'Jan', searches: 120, matches: 45 },
  { month: 'Feb', searches: 150, matches: 60 },
  { month: 'Mar', searches: 180, matches: 72 },
  { month: 'Apr', searches: 200, matches: 85 },
  { month: 'May', searches: 220, matches: 95 },
  { month: 'Jun', searches: 250, matches: 110 }
];

const domainDistributionData = [
  { name: 'Technology', value: 65, color: '#3b82f6' },
  { name: 'Healthcare', value: 25, color: '#10b981' },
  { name: 'General', value: 10, color: '#6b7280' }
];

const performanceData = [
  { name: 'Search Speed', value: 245, target: 200 },
  { name: 'Accuracy', value: 94.2, target: 90 },
  { name: 'User Satisfaction', value: 4.8, target: 4.5 }
];

const stats = [
  {
    title: 'Total Searches',
    value: '1,247',
    change: '+12%',
    positive: true,
    icon: Search,
    color: '#3b82f6'
  },
  {
    title: 'Successful Matches',
    value: '892',
    change: '+18%',
    positive: true,
    icon: Target,
    color: '#10b981'
  },
  {
    title: 'Avg Response Time',
    value: '245ms',
    change: '-15%',
    positive: true,
    icon: Clock,
    color: '#f59e0b'
  },
  {
    title: 'Active Candidates',
    value: '2,156',
    change: '+8%',
    positive: true,
    icon: Users,
    color: '#8b5cf6'
  }
];

const features = [
  {
    name: 'Domain-Aware Search',
    icon: Brain,
    color: '#3b82f6',
    usage: '95%',
    accuracy: '94.2%',
    performance: '245ms'
  },
  {
    name: 'Explainable AI',
    icon: Brain,
    color: '#10b981',
    usage: '78%',
    accuracy: '91.5%',
    performance: '320ms'
  },
  {
    name: 'Bias Prevention',
    icon: Shield,
    color: '#f59e0b',
    usage: '85%',
    accuracy: '96.8%',
    performance: '180ms'
  },
  {
    name: 'Ultra-Fast Search',
    icon: Zap,
    color: '#8b5cf6',
    usage: '92%',
    accuracy: '89.3%',
    performance: '120ms'
  }
];

function AnalyticsPage() {
  const [timeRange, setTimeRange] = useState('30d');
  const { state } = useApp();

  return (
    <AnalyticsPageContainer>
      <PageHeader>
        <PageTitle>
          <BarChart3 size={32} />
          Analytics Dashboard
        </PageTitle>
        <TimeRangeSelector
          value={timeRange}
          onChange={(e) => setTimeRange(e.target.value)}
        >
          <option value="7d">Last 7 days</option>
          <option value="30d">Last 30 days</option>
          <option value="90d">Last 90 days</option>
          <option value="1y">Last year</option>
        </TimeRangeSelector>
      </PageHeader>

      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
      >
        <StatsGrid>
          {stats.map((stat, index) => {
            const Icon = stat.icon;
            return (
              <motion.div
                key={index}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: index * 0.1 }}
              >
                <StatCard>
                  <StatHeader>
                    <StatTitle>{stat.title}</StatTitle>
                    <StatIcon color={stat.color}>
                      <Icon size={20} />
                    </StatIcon>
                  </StatHeader>
                  <StatValue>{stat.value}</StatValue>
                  <StatChange positive={stat.positive}>
                    <TrendingUp size={16} />
                    {stat.change}
                  </StatChange>
                </StatCard>
              </motion.div>
            );
          })}
        </StatsGrid>
      </motion.div>

      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, delay: 0.2 }}
      >
        <ChartsGrid>
          <ChartCard>
            <ChartTitle>Search Trends</ChartTitle>
            <ChartContainer>
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={searchTrendsData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="month" />
                  <YAxis />
                  <Tooltip />
                  <Line 
                    type="monotone" 
                    dataKey="searches" 
                    stroke="#3b82f6" 
                    strokeWidth={2}
                    name="Searches"
                  />
                  <Line 
                    type="monotone" 
                    dataKey="matches" 
                    stroke="#10b981" 
                    strokeWidth={2}
                    name="Matches"
                  />
                </LineChart>
              </ResponsiveContainer>
            </ChartContainer>
          </ChartCard>

          <ChartCard>
            <ChartTitle>Domain Distribution</ChartTitle>
            <ChartContainer>
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={domainDistributionData}
                    cx="50%"
                    cy="50%"
                    innerRadius={60}
                    outerRadius={100}
                    paddingAngle={5}
                    dataKey="value"
                  >
                    {domainDistributionData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </ChartContainer>
          </ChartCard>
        </ChartsGrid>
      </motion.div>

      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, delay: 0.4 }}
      >
        <ChartCard>
          <ChartTitle>Performance Metrics</ChartTitle>
          <ChartContainer>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={performanceData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" />
                <YAxis />
                <Tooltip />
                <Bar dataKey="value" fill="#3b82f6" name="Current" />
                <Bar dataKey="target" fill="#e5e7eb" name="Target" />
              </BarChart>
            </ResponsiveContainer>
          </ChartContainer>
        </ChartCard>
      </motion.div>

      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, delay: 0.6 }}
      >
        <ChartTitle style={{ marginBottom: '1.5rem' }}>AI Features Performance</ChartTitle>
        <FeatureAnalytics>
          {features.map((feature, index) => {
            const Icon = feature.icon;
            return (
              <motion.div
                key={index}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: 0.7 + index * 0.1 }}
              >
                <FeatureCard>
                  <FeatureHeader>
                    <FeatureIcon color={feature.color}>
                      <Icon size={24} />
                    </FeatureIcon>
                    <FeatureTitle>{feature.name}</FeatureTitle>
                  </FeatureHeader>
                  <FeatureStats>
                    <FeatureStat>
                      <FeatureStatValue>{feature.usage}</FeatureStatValue>
                      <FeatureStatLabel>Usage</FeatureStatLabel>
                    </FeatureStat>
                    <FeatureStat>
                      <FeatureStatValue>{feature.accuracy}</FeatureStatValue>
                      <FeatureStatLabel>Accuracy</FeatureStatLabel>
                    </FeatureStat>
                    <FeatureStat>
                      <FeatureStatValue>{feature.performance}</FeatureStatValue>
                      <FeatureStatLabel>Performance</FeatureStatLabel>
                    </FeatureStat>
                    <FeatureStat>
                      <FeatureStatValue>4.8★</FeatureStatValue>
                      <FeatureStatLabel>Rating</FeatureStatLabel>
                    </FeatureStat>
                  </FeatureStats>
                </FeatureCard>
              </motion.div>
            );
          })}
        </FeatureAnalytics>
      </motion.div>
    </AnalyticsPageContainer>
  );
}

export default AnalyticsPage;
