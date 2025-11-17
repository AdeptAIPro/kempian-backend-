import React from 'react';
import styled from 'styled-components';
import { 
  Search, 
  Users, 
  BarChart3, 
  Settings, 
  Download,
  Upload,
  RefreshCw,
  Zap,
  Brain,
  Shield
} from 'lucide-react';
import { useApp } from '../context/AppContext';

const QuickActionsContainer = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 1rem;
`;

const ActionButton = styled.button`
  background: white;
  border: 1px solid #e5e7eb;
  border-radius: 0.75rem;
  padding: 1.5rem;
  cursor: pointer;
  transition: all 0.2s ease-in-out;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 0.75rem;
  text-align: center;
  box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);

  &:hover {
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    transform: translateY(-2px);
    border-color: #3b82f6;
  }

  &:active {
    transform: translateY(0);
  }
`;

const ActionIcon = styled.div`
  width: 48px;
  height: 48px;
  border-radius: 0.75rem;
  background: ${props => props.color}20;
  color: ${props => props.color};
  display: flex;
  align-items: center;
  justify-content: center;
`;

const ActionTitle = styled.h3`
  font-size: 1rem;
  font-weight: 600;
  color: #1e293b;
  margin: 0;
`;

const ActionDescription = styled.p`
  font-size: 0.875rem;
  color: #6b7280;
  margin: 0;
  line-height: 1.4;
`;

const ActionBadge = styled.span`
  background: #3b82f6;
  color: white;
  font-size: 0.625rem;
  font-weight: 600;
  padding: 0.25rem 0.5rem;
  border-radius: 0.25rem;
  margin-top: 0.25rem;
`;

const actions = [
  {
    id: 'search',
    title: 'New Search',
    description: 'Search for candidates with advanced AI',
    icon: Search,
    color: '#3b82f6',
    action: 'search'
  },
  {
    id: 'candidates',
    title: 'View All Candidates',
    description: 'Browse the complete candidate database',
    icon: Users,
    color: '#10b981',
    action: 'candidates'
  },
  {
    id: 'analytics',
    title: 'Analytics Dashboard',
    description: 'View performance metrics and insights',
    icon: BarChart3,
    color: '#8b5cf6',
    action: 'analytics'
  },
  {
    id: 'refresh',
    title: 'Refresh Data',
    description: 'Reload candidates from database',
    icon: RefreshCw,
    color: '#f59e0b',
    action: 'refresh'
  },
  {
    id: 'export',
    title: 'Export Results',
    description: 'Download search results as CSV',
    icon: Download,
    color: '#ef4444',
    action: 'export'
  },
  {
    id: 'settings',
    title: 'Settings',
    description: 'Configure search preferences',
    icon: Settings,
    color: '#6b7280',
    action: 'settings'
  }
];

const featureActions = [
  {
    id: 'explainable-ai',
    title: 'Explainable AI',
    description: 'Understand AI decisions',
    icon: Brain,
    color: '#10b981',
    badge: 'Beta',
    action: 'explainable-ai'
  },
  {
    id: 'bias-prevention',
    title: 'Bias Prevention',
    description: 'Fair hiring practices',
    icon: Shield,
    color: '#f59e0b',
    badge: 'Beta',
    action: 'bias-prevention'
  },
  {
    id: 'ultra-fast',
    title: 'Ultra-Fast Search',
    description: 'Lightning quick results',
    icon: Zap,
    color: '#8b5cf6',
    badge: 'New',
    action: 'ultra-fast'
  }
];

function QuickActions() {
  const { actions: appActions } = useApp();

  const handleAction = (actionId) => {
    switch (actionId) {
      case 'search':
        // Focus on search input or navigate to search page
        appActions.setCurrentPage('search');
        break;
      case 'candidates':
        appActions.setCurrentPage('candidates');
        break;
      case 'analytics':
        appActions.setCurrentPage('analytics');
        break;
      case 'refresh':
        appActions.loadCandidates();
        break;
      case 'export':
        // Implement export functionality
        console.log('Export functionality not implemented yet');
        break;
      case 'settings':
        appActions.setCurrentPage('settings');
        break;
      case 'explainable-ai':
        appActions.setCurrentPage('explainable-ai');
        break;
      case 'bias-prevention':
        appActions.setCurrentPage('bias-prevention');
        break;
      case 'ultra-fast':
        // Enable ultra-fast search mode
        appActions.updateSettings({ ultraFastMode: true });
        break;
      default:
        console.log(`Action ${actionId} not implemented`);
    }
  };

  return (
    <div>
      <QuickActionsContainer>
        {actions.map((action) => {
          const Icon = action.icon;
          return (
            <ActionButton
              key={action.id}
              onClick={() => handleAction(action.action)}
            >
              <ActionIcon color={action.color}>
                <Icon size={24} />
              </ActionIcon>
              <ActionTitle>{action.title}</ActionTitle>
              <ActionDescription>{action.description}</ActionDescription>
            </ActionButton>
          );
        })}
      </QuickActionsContainer>

      <div style={{ marginTop: '2rem' }}>
        <h3 style={{ 
          fontSize: '1.125rem', 
          fontWeight: '600', 
          color: '#1e293b', 
          margin: '0 0 1rem 0' 
        }}>
          AI Features
        </h3>
        <QuickActionsContainer>
          {featureActions.map((action) => {
            const Icon = action.icon;
            return (
              <ActionButton
                key={action.id}
                onClick={() => handleAction(action.action)}
              >
                <ActionIcon color={action.color}>
                  <Icon size={24} />
                </ActionIcon>
                <ActionTitle>{action.title}</ActionTitle>
                <ActionDescription>{action.description}</ActionDescription>
                {action.badge && <ActionBadge>{action.badge}</ActionBadge>}
              </ActionButton>
            );
          })}
        </QuickActionsContainer>
      </div>
    </div>
  );
}

export default QuickActions;
