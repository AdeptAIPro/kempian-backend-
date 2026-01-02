import React, { useState } from 'react';
import styled from 'styled-components';
import { motion } from 'framer-motion';
import { 
  Settings, 
  Search, 
  Brain, 
  Shield, 
  Zap, 
  Save,
  RefreshCw,
  Download,
  Upload
} from 'lucide-react';
import { useApp } from '../context/AppContext';

const SettingsPageContainer = styled.div`
  display: flex;
  flex-direction: column;
  gap: 2rem;
`;

const PageHeader = styled.div`
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 1rem;
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

const SaveButton = styled.button`
  background: #3b82f6;
  color: white;
  border: none;
  border-radius: 0.5rem;
  padding: 0.75rem 1.5rem;
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

const SettingsGrid = styled.div`
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 2rem;

  @media (max-width: 1024px) {
    grid-template-columns: 1fr;
  }
`;

const SettingsSection = styled.div`
  background: white;
  border: 1px solid #e5e7eb;
  border-radius: 0.75rem;
  padding: 1.5rem;
  box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
`;

const SectionTitle = styled.h2`
  font-size: 1.25rem;
  font-weight: 600;
  color: #1e293b;
  margin: 0 0 1.5rem 0;
  display: flex;
  align-items: center;
  gap: 0.5rem;
`;

const SettingGroup = styled.div`
  margin-bottom: 1.5rem;

  &:last-child {
    margin-bottom: 0;
  }
`;

const SettingLabel = styled.label`
  display: block;
  font-size: 0.875rem;
  font-weight: 500;
  color: #374151;
  margin-bottom: 0.5rem;
`;

const SettingInput = styled.input`
  width: 100%;
  padding: 0.75rem;
  border: 1px solid #d1d5db;
  border-radius: 0.5rem;
  font-size: 0.875rem;
  color: #374151;
  transition: border-color 0.2s ease-in-out;

  &:focus {
    outline: none;
    border-color: #3b82f6;
    box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
  }
`;

const SettingSelect = styled.select`
  width: 100%;
  padding: 0.75rem;
  border: 1px solid #d1d5db;
  border-radius: 0.5rem;
  font-size: 0.875rem;
  color: #374151;
  background: white;
  cursor: pointer;
  transition: border-color 0.2s ease-in-out;

  &:focus {
    outline: none;
    border-color: #3b82f6;
    box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
  }
`;

const SettingCheckbox = styled.input`
  width: 1rem;
  height: 1rem;
  accent-color: #3b82f6;
  margin-right: 0.5rem;
`;

const SettingCheckboxLabel = styled.label`
  display: flex;
  align-items: center;
  font-size: 0.875rem;
  color: #374151;
  cursor: pointer;
  margin-bottom: 0.5rem;

  &:last-child {
    margin-bottom: 0;
  }
`;

const SettingDescription = styled.p`
  font-size: 0.75rem;
  color: #6b7280;
  margin: 0.25rem 0 0 0;
`;

const ActionButtons = styled.div`
  display: flex;
  gap: 1rem;
  margin-top: 2rem;
  padding-top: 2rem;
  border-top: 1px solid #e5e7eb;
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

const DangerZone = styled.div`
  background: #fef2f2;
  border: 1px solid #fecaca;
  border-radius: 0.75rem;
  padding: 1.5rem;
  margin-top: 2rem;
`;

const DangerTitle = styled.h3`
  font-size: 1rem;
  font-weight: 600;
  color: #dc2626;
  margin: 0 0 1rem 0;
`;

const DangerButton = styled.button`
  background: #dc2626;
  color: white;
  border: none;
  border-radius: 0.5rem;
  padding: 0.75rem 1rem;
  font-size: 0.875rem;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.2s ease-in-out;

  &:hover:not(:disabled) {
    background: #b91c1c;
    transform: translateY(-1px);
  }

  &:disabled {
    opacity: 0.6;
    cursor: not-allowed;
    transform: none;
  }
`;

function SettingsPage() {
  const { state, actions } = useApp();
  const { settings } = state;
  const [localSettings, setLocalSettings] = useState(settings);
  const [hasChanges, setHasChanges] = useState(false);

  const handleSettingChange = (key, value) => {
    setLocalSettings(prev => ({
      ...prev,
      [key]: value
    }));
    setHasChanges(true);
  };

  const handleSave = () => {
    actions.updateSettings(localSettings);
    setHasChanges(false);
  };

  const handleReset = () => {
    setLocalSettings(settings);
    setHasChanges(false);
  };

  const handleExportSettings = () => {
    const dataStr = JSON.stringify(localSettings, null, 2);
    const dataUri = 'data:application/json;charset=utf-8,'+ encodeURIComponent(dataStr);
    const exportFileDefaultName = 'adeptai-settings.json';
    
    const linkElement = document.createElement('a');
    linkElement.setAttribute('href', dataUri);
    linkElement.setAttribute('download', exportFileDefaultName);
    linkElement.click();
  };

  const handleImportSettings = (event) => {
    const file = event.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (e) => {
        try {
          const importedSettings = JSON.parse(e.target.result);
          setLocalSettings(importedSettings);
          setHasChanges(true);
        } catch (error) {
          console.error('Failed to import settings:', error);
        }
      };
      reader.readAsText(file);
    }
  };

  return (
    <SettingsPageContainer>
      <PageHeader>
        <PageTitle>
          <Settings size={32} />
          Settings
        </PageTitle>
        <SaveButton onClick={handleSave} disabled={!hasChanges}>
          <Save size={16} />
          Save Changes
        </SaveButton>
      </PageHeader>

      <SettingsGrid>
        <motion.div
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.6 }}
        >
          <SettingsSection>
            <SectionTitle>
              <Search size={20} />
              Search Settings
            </SectionTitle>
            
            <SettingGroup>
              <SettingLabel>Results per page</SettingLabel>
              <SettingSelect
                value={localSettings.resultsPerPage}
                onChange={(e) => handleSettingChange('resultsPerPage', parseInt(e.target.value))}
              >
                <option value={5}>5 results</option>
                <option value={10}>10 results</option>
                <option value={20}>20 results</option>
                <option value={50}>50 results</option>
              </SettingSelect>
              <SettingDescription>
                Number of candidates to display per search
              </SettingDescription>
            </SettingGroup>

            <SettingGroup>
              <SettingCheckboxLabel>
                <SettingCheckbox
                  type="checkbox"
                  checked={localSettings.domainFiltering}
                  onChange={(e) => handleSettingChange('domainFiltering', e.target.checked)}
                />
                Enable domain-aware filtering
              </SettingCheckboxLabel>
              <SettingDescription>
                Automatically filter candidates by domain (healthcare vs technology)
              </SettingDescription>
            </SettingGroup>

            <SettingGroup>
              <SettingCheckboxLabel>
                <SettingCheckbox
                  type="checkbox"
                  checked={localSettings.autoRefresh}
                  onChange={(e) => handleSettingChange('autoRefresh', e.target.checked)}
                />
                Auto-refresh data
              </SettingCheckboxLabel>
              <SettingDescription>
                Automatically refresh candidate data every 5 minutes
              </SettingDescription>
            </SettingGroup>
          </SettingsSection>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.6, delay: 0.2 }}
        >
          <SettingsSection>
            <SectionTitle>
              <Brain size={20} />
              AI Features
            </SectionTitle>
            
            <SettingGroup>
              <SettingCheckboxLabel>
                <SettingCheckbox
                  type="checkbox"
                  checked={localSettings.includeExplainableAI}
                  onChange={(e) => handleSettingChange('includeExplainableAI', e.target.checked)}
                />
                Include AI explanations
              </SettingCheckboxLabel>
              <SettingDescription>
                Show detailed explanations for why candidates match
              </SettingDescription>
            </SettingGroup>

            <SettingGroup>
              <SettingCheckboxLabel>
                <SettingCheckbox
                  type="checkbox"
                  checked={localSettings.includeBehavioralAnalysis}
                  onChange={(e) => handleSettingChange('includeBehavioralAnalysis', e.target.checked)}
                />
                Include behavioral analysis
              </SettingCheckboxLabel>
              <SettingDescription>
                Analyze candidate personality and work preferences
              </SettingDescription>
            </SettingGroup>

            <SettingGroup>
              <SettingLabel>AI Model Preference</SettingLabel>
              <SettingSelect
                value="balanced"
                onChange={(e) => console.log('Model changed:', e.target.value)}
              >
                <option value="fast">Fast (Lower accuracy)</option>
                <option value="balanced">Balanced (Recommended)</option>
                <option value="accurate">Accurate (Slower)</option>
              </SettingSelect>
              <SettingDescription>
                Choose between speed and accuracy for AI processing
              </SettingDescription>
            </SettingGroup>
          </SettingsSection>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.6, delay: 0.4 }}
        >
          <SettingsSection>
            <SectionTitle>
              <Shield size={20} />
              Privacy & Security
            </SectionTitle>
            
            <SettingGroup>
              <SettingCheckboxLabel>
                <SettingCheckbox
                  type="checkbox"
                  defaultChecked
                />
                Enable bias prevention
              </SettingCheckboxLabel>
              <SettingDescription>
                Automatically detect and prevent bias in search results
              </SettingDescription>
            </SettingGroup>

            <SettingGroup>
              <SettingCheckboxLabel>
                <SettingCheckbox
                  type="checkbox"
                  defaultChecked
                />
                Anonymize candidate data
              </SettingCheckboxLabel>
              <SettingDescription>
                Remove personally identifiable information from results
              </SettingDescription>
            </SettingGroup>

            <SettingGroup>
              <SettingLabel>Data Retention</SettingLabel>
              <SettingSelect
                defaultValue="90d"
                onChange={(e) => console.log('Retention changed:', e.target.value)}
              >
                <option value="30d">30 days</option>
                <option value="90d">90 days</option>
                <option value="1y">1 year</option>
                <option value="never">Never delete</option>
              </SettingSelect>
              <SettingDescription>
                How long to keep search history and analytics data
              </SettingDescription>
            </SettingGroup>
          </SettingsSection>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.6, delay: 0.6 }}
        >
          <SettingsSection>
            <SectionTitle>
              <Zap size={20} />
              Performance
            </SectionTitle>
            
            <SettingGroup>
              <SettingLabel>Search Timeout</SettingLabel>
              <SettingSelect
                defaultValue="30s"
                onChange={(e) => console.log('Timeout changed:', e.target.value)}
              >
                <option value="10s">10 seconds</option>
                <option value="30s">30 seconds</option>
                <option value="60s">60 seconds</option>
              </SettingSelect>
              <SettingDescription>
                Maximum time to wait for search results
              </SettingDescription>
            </SettingGroup>

            <SettingGroup>
              <SettingCheckboxLabel>
                <SettingCheckbox
                  type="checkbox"
                  defaultChecked
                />
                Enable caching
              </SettingCheckboxLabel>
              <SettingDescription>
                Cache search results for faster subsequent searches
              </SettingDescription>
            </SettingGroup>

            <SettingGroup>
              <SettingCheckboxLabel>
                <SettingCheckbox
                  type="checkbox"
                />
                Enable preloading
              </SettingCheckboxLabel>
              <SettingDescription>
                Preload common search results in the background
              </SettingDescription>
            </SettingGroup>
          </SettingsSection>
        </motion.div>
      </SettingsGrid>

      <ActionButtons>
        <ActionButton onClick={handleReset} disabled={!hasChanges}>
          <RefreshCw size={16} />
          Reset to Default
        </ActionButton>
        <ActionButton onClick={handleExportSettings}>
          <Download size={16} />
          Export Settings
        </ActionButton>
        <ActionButton>
          <Upload size={16} />
          <input
            type="file"
            accept=".json"
            onChange={handleImportSettings}
            style={{ display: 'none' }}
            id="import-settings"
          />
          <label htmlFor="import-settings" style={{ cursor: 'pointer', margin: 0 }}>
            Import Settings
          </label>
        </ActionButton>
      </ActionButtons>

      <DangerZone>
        <DangerTitle>Danger Zone</DangerTitle>
        <p style={{ color: '#6b7280', fontSize: '0.875rem', margin: '0 0 1rem 0' }}>
          These actions are irreversible. Please be careful.
        </p>
        <DangerButton>
          Clear All Data
        </DangerButton>
      </DangerZone>
    </SettingsPageContainer>
  );
}

export default SettingsPage;
