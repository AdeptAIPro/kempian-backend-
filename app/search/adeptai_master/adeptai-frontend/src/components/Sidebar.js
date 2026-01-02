import React from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import styled from 'styled-components';
import { 
  Home, 
  Search, 
  Users, 
  BarChart3, 
  Settings, 
  Activity,
  ChevronLeft,
  ChevronRight,
  Brain,
  Shield,
  TrendingUp,
  Database
} from 'lucide-react';
import { useApp } from '../context/AppContext';

const SidebarContainer = styled.aside`
  width: 280px;
  background: white;
  border-right: 1px solid #e5e7eb;
  height: 100vh;
  position: fixed;
  left: 0;
  top: 0;
  z-index: 40;
  transition: transform 0.3s ease-in-out;
  transform: ${props => props.collapsed ? 'translateX(-100%)' : 'translateX(0)'};

  @media (max-width: 1024px) {
    transform: ${props => props.collapsed ? 'translateX(-100%)' : 'translateX(0)'};
  }
`;

const SidebarHeader = styled.div`
  padding: 1.5rem;
  border-bottom: 1px solid #e5e7eb;
  display: flex;
  align-items: center;
  justify-content: space-between;
`;

const SidebarTitle = styled.h2`
  font-size: 1.25rem;
  font-weight: 600;
  color: #1e293b;
  margin: 0;
`;

const ToggleButton = styled.button`
  background: none;
  border: none;
  color: #6b7280;
  cursor: pointer;
  padding: 0.5rem;
  border-radius: 0.375rem;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: all 0.2s ease-in-out;

  &:hover {
    background: #f3f4f6;
    color: #374151;
  }

  @media (min-width: 1025px) {
    display: none;
  }
`;

const Navigation = styled.nav`
  padding: 1rem 0;
`;

const NavSection = styled.div`
  margin-bottom: 2rem;
`;

const SectionTitle = styled.h3`
  font-size: 0.75rem;
  font-weight: 600;
  color: #6b7280;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  margin: 0 0 0.5rem 0;
  padding: 0 1.5rem;
`;

const NavItem = styled.button`
  width: 100%;
  display: flex;
  align-items: center;
  gap: 0.75rem;
  padding: 0.75rem 1.5rem;
  border: none;
  background: none;
  color: #6b7280;
  text-align: left;
  cursor: pointer;
  transition: all 0.2s ease-in-out;
  font-size: 0.875rem;
  font-weight: 500;

  &:hover {
    background: #f8fafc;
    color: #374151;
  }

  &.active {
    background: #eff6ff;
    color: #3b82f6;
    border-right: 3px solid #3b82f6;
  }
`;

const NavIcon = styled.div`
  display: flex;
  align-items: center;
  justify-content: center;
  width: 20px;
  height: 20px;
`;

const FeatureBadge = styled.span`
  background: #10b981;
  color: white;
  font-size: 0.625rem;
  font-weight: 600;
  padding: 0.125rem 0.375rem;
  border-radius: 0.25rem;
  margin-left: auto;
`;

const BetaBadge = styled.span`
  background: #f59e0b;
  color: white;
  font-size: 0.625rem;
  font-weight: 600;
  padding: 0.125rem 0.375rem;
  border-radius: 0.25rem;
  margin-left: auto;
`;

const ExperimentalBadge = styled.span`
  background: #8b5cf6;
  color: white;
  font-size: 0.625rem;
  font-weight: 600;
  padding: 0.125rem 0.375rem;
  border-radius: 0.25rem;
  margin-left: auto;
`;

const SidebarFooter = styled.div`
  position: absolute;
  bottom: 0;
  left: 0;
  right: 0;
  padding: 1rem 1.5rem;
  border-top: 1px solid #e5e7eb;
  background: #f8fafc;
`;

const FooterText = styled.p`
  font-size: 0.75rem;
  color: #6b7280;
  margin: 0;
  text-align: center;
`;

const navigationItems = [
  {
    section: 'Main',
    items: [
      { path: '/', label: 'Dashboard', icon: Home },
      { path: '/search', label: 'Search', icon: Search },
      { path: '/candidates', label: 'Candidates', icon: Users },
      { path: '/analytics', label: 'Analytics', icon: BarChart3 },
    ]
  },
  {
    section: 'AI Features',
    items: [
      { path: '/explainable-ai', label: 'Explainable AI', icon: Brain, badge: 'Beta' },
      { path: '/behavioral-analysis', label: 'Behavioral Analysis', icon: Activity, badge: 'Experimental' },
      { path: '/bias-prevention', label: 'Bias Prevention', icon: Shield, badge: 'Beta' },
    ]
  },
  {
    section: 'Intelligence',
    items: [
      { path: '/market-intelligence', label: 'Market Intelligence', icon: TrendingUp, badge: 'New' },
      { path: '/compensation', label: 'Compensation', icon: Database, badge: 'Beta' },
    ]
  },
  {
    section: 'System',
    items: [
      { path: '/health', label: 'System Health', icon: Activity },
      { path: '/settings', label: 'Settings', icon: Settings },
    ]
  }
];

function Sidebar() {
  const navigate = useNavigate();
  const location = useLocation();
  const { state, actions } = useApp();
  const { sidebarCollapsed } = state;

  const handleNavigation = (path) => {
    navigate(path);
    actions.setCurrentPage(path);
  };

  const getBadgeComponent = (badge) => {
    switch (badge) {
      case 'New': return <FeatureBadge>NEW</FeatureBadge>;
      case 'Beta': return <BetaBadge>BETA</BetaBadge>;
      case 'Experimental': return <ExperimentalBadge>EXP</ExperimentalBadge>;
      default: return null;
    }
  };

  return (
    <SidebarContainer collapsed={sidebarCollapsed}>
      <SidebarHeader>
        <SidebarTitle>Navigation</SidebarTitle>
        <ToggleButton onClick={actions.toggleSidebar}>
          {sidebarCollapsed ? <ChevronRight size={20} /> : <ChevronLeft size={20} />}
        </ToggleButton>
      </SidebarHeader>

      <Navigation>
        {navigationItems.map((section, sectionIndex) => (
          <NavSection key={sectionIndex}>
            <SectionTitle>{section.section}</SectionTitle>
            {section.items.map((item, itemIndex) => {
              const Icon = item.icon;
              const isActive = location.pathname === item.path;
              
              return (
                <NavItem
                  key={itemIndex}
                  className={isActive ? 'active' : ''}
                  onClick={() => handleNavigation(item.path)}
                >
                  <NavIcon>
                    <Icon size={20} />
                  </NavIcon>
                  {item.label}
                  {item.badge && getBadgeComponent(item.badge)}
                </NavItem>
              );
            })}
          </NavSection>
        ))}
      </Navigation>

      <SidebarFooter>
        <FooterText>
          AdeptAI v1.0.0
          <br />
          AI-Powered Recruitment
        </FooterText>
      </SidebarFooter>
    </SidebarContainer>
  );
}

export default Sidebar;
