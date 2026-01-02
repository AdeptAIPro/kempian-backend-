import React, { createContext, useContext, useReducer, useEffect } from 'react';
import { toast } from 'react-hot-toast';
import { apiService, formatCandidateData } from '../services/api';

const AppContext = createContext();

const initialState = {
  // Search state
  searchQuery: '',
  searchResults: [],
  isSearching: false,
  searchError: null,
  
  // Candidates state
  candidates: [],
  selectedCandidate: null,
  isCandidatesLoading: false,
  
  // UI state
  sidebarCollapsed: false,
  currentPage: 'dashboard',
  theme: 'light',
  
  // System state
  systemHealth: {
    status: 'unknown',
    components: {},
    lastChecked: null
  },
  
  // Settings
  settings: {
    domainFiltering: true,
    includeBehavioralAnalysis: false,
    includeExplainableAI: true,
    resultsPerPage: 10,
    autoRefresh: false
  },
  
  // Performance metrics
  performance: {
    searchTime: 0,
    totalSearches: 0,
    cacheHitRate: 0
  }
};

function appReducer(state, action) {
  switch (action.type) {
    case 'SET_SEARCH_QUERY':
      return { ...state, searchQuery: action.payload };
    
    case 'SET_SEARCH_RESULTS':
      return { 
        ...state, 
        searchResults: action.payload.results,
        searchError: action.payload.error,
        isSearching: false
      };
    
    case 'SET_SEARCHING':
      return { ...state, isSearching: action.payload };
    
    case 'SET_CANDIDATES':
      return { 
        ...state, 
        candidates: action.payload,
        isCandidatesLoading: false
      };
    
    case 'SET_CANDIDATES_LOADING':
      return { ...state, isCandidatesLoading: action.payload };
    
    case 'SET_SELECTED_CANDIDATE':
      return { ...state, selectedCandidate: action.payload };
    
    case 'TOGGLE_SIDEBAR':
      return { ...state, sidebarCollapsed: !state.sidebarCollapsed };
    
    case 'SET_CURRENT_PAGE':
      return { ...state, currentPage: action.payload };
    
    case 'SET_SYSTEM_HEALTH':
      return { 
        ...state, 
        systemHealth: {
          ...action.payload,
          lastChecked: new Date().toISOString()
        }
      };
    
    case 'UPDATE_SETTINGS':
      return { 
        ...state, 
        settings: { ...state.settings, ...action.payload }
      };
    
    case 'UPDATE_PERFORMANCE':
      return { 
        ...state, 
        performance: { ...state.performance, ...action.payload }
      };
    
    case 'RESET_SEARCH':
      return { 
        ...state, 
        searchQuery: '',
        searchResults: [],
        searchError: null,
        isSearching: false
      };
    
    default:
      return state;
  }
}

export function AppProvider({ children }) {
  const [state, dispatch] = useReducer(appReducer, initialState);

  // Load settings from localStorage on mount
  useEffect(() => {
    const savedSettings = localStorage.getItem('adeptai-settings');
    if (savedSettings) {
      try {
        const settings = JSON.parse(savedSettings);
        dispatch({ type: 'UPDATE_SETTINGS', payload: settings });
      } catch (error) {
        console.error('Failed to load settings:', error);
      }
    }
  }, []);

  // Save settings to localStorage when they change
  useEffect(() => {
    localStorage.setItem('adeptai-settings', JSON.stringify(state.settings));
  }, [state.settings]);

  // Check system health on mount only (removed periodic checks to prevent spam)
  useEffect(() => {
    const checkHealth = async () => {
      try {
        const health = await apiService.checkHealth();
        dispatch({ type: 'SET_SYSTEM_HEALTH', payload: health });
      } catch (error) {
        console.error('Health check failed:', error);
        dispatch({ 
          type: 'SET_SYSTEM_HEALTH', 
          payload: { 
            status: 'unhealthy', 
            components: {}, 
            lastChecked: new Date().toISOString() 
          } 
        });
      }
    };

    checkHealth();
    // Removed the interval to prevent repeated health checks
  }, []);

  // Actions
  const actions = {
    setSearchQuery: (query) => {
      dispatch({ type: 'SET_SEARCH_QUERY', payload: query });
    },

    performSearch: async (query, options = {}) => {
      // Validate query before sending
      if (!query || !query.trim()) {
        toast.error('Please enter a search query');
        return;
      }
      
      dispatch({ type: 'SET_SEARCHING', payload: true });
      dispatch({ type: 'SET_SEARCH_QUERY', payload: query });
      
      try {
        const results = await apiService.search(query.trim(), {
          top_k: state.settings.resultsPerPage,
          enable_domain_filtering: state.settings.domainFiltering,
          include_behavioural_analysis: state.settings.includeBehavioralAnalysis,
          ...options
        });
        
        // Format the search results for frontend consumption
        const rawResults = results.data || results.results || [];
        const formattedResults = rawResults.map(formatCandidateData);
        
        dispatch({ 
          type: 'SET_SEARCH_RESULTS', 
          payload: { results: formattedResults, error: null }
        });
        
        // Update performance metrics
        if (results.performance) {
          dispatch({ 
            type: 'UPDATE_PERFORMANCE', 
            payload: {
              searchTime: results.performance.response_time_ms,
              totalSearches: state.performance.totalSearches + 1
            }
          });
        }
        
        toast.success(`Found ${formattedResults.length} candidates`);
        
      } catch (error) {
        console.error('Search failed:', error);
        dispatch({ 
          type: 'SET_SEARCH_RESULTS', 
          payload: { results: [], error: error.message }
        });
        toast.error(`Search failed: ${error.message}`);
      }
    },

    loadCandidates: async () => {
      dispatch({ type: 'SET_CANDIDATES_LOADING', payload: true });
      
      try {
        const candidates = await apiService.getAllCandidates();
        dispatch({ type: 'SET_CANDIDATES', payload: candidates });
        toast.success(`Loaded ${candidates.length} candidates`);
      } catch (error) {
        console.error('Failed to load candidates:', error);
        toast.error(`Failed to load candidates: ${error.message}`);
        dispatch({ type: 'SET_CANDIDATES_LOADING', payload: false });
      }
    },

    setSelectedCandidate: (candidate) => {
      dispatch({ type: 'SET_SELECTED_CANDIDATE', payload: candidate });
    },

    toggleSidebar: () => {
      dispatch({ type: 'TOGGLE_SIDEBAR' });
    },

    setCurrentPage: (page) => {
      dispatch({ type: 'SET_CURRENT_PAGE', payload: page });
    },

    updateSettings: (newSettings) => {
      dispatch({ type: 'UPDATE_SETTINGS', payload: newSettings });
      toast.success('Settings updated');
    },

    resetSearch: () => {
      dispatch({ type: 'RESET_SEARCH' });
    },

    refreshSystemHealth: async () => {
      try {
        const health = await apiService.checkHealth();
        dispatch({ type: 'SET_SYSTEM_HEALTH', payload: health });
        toast.success('System health refreshed');
      } catch (error) {
        console.error('Health check failed:', error);
        toast.error('Failed to refresh system health');
      }
    }
  };

  const value = {
    state,
    actions
  };

  return (
    <AppContext.Provider value={value}>
      {children}
    </AppContext.Provider>
  );
}

export function useApp() {
  const context = useContext(AppContext);
  if (!context) {
    throw new Error('useApp must be used within an AppProvider');
  }
  return context;
}
