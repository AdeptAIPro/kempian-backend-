import axios from 'axios';

// Create axios instance with default config
const api = axios.create({
  baseURL: process.env.REACT_APP_API_URL || 'http://localhost:5055',
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor
api.interceptors.request.use(
  (config) => {
    // Add any auth tokens here if needed
    console.log(`Making ${config.method?.toUpperCase()} request to ${config.url}`);
    return config;
  },
  (error) => {
    console.error('Request error:', error);
    return Promise.reject(error);
  }
);

// Response interceptor
api.interceptors.response.use(
  (response) => {
    console.log(`Response from ${response.config.url}:`, response.status);
    return response;
  },
  (error) => {
    console.error('Response error:', error);
    
    // Handle different error types
    if (error.response) {
      // Server responded with error status
      let message = 'Server error';
      
      if (error.response.data) {
        if (error.response.data.error) {
          // Handle validation errors
          if (error.response.data.error.message) {
            message = error.response.data.error.message;
          } else if (typeof error.response.data.error === 'string') {
            message = error.response.data.error;
          }
        } else if (error.response.data.message) {
          message = error.response.data.message;
        }
      }
      
      throw new Error(`${error.response.status}: ${message}`);
    } else if (error.request) {
      // Request was made but no response received
      throw new Error('No response from server. Please check your connection.');
    } else {
      // Something else happened
      throw new Error(error.message || 'An unexpected error occurred');
    }
  }
);

export const apiService = {
  // Search endpoints
  search: async (query, options = {}) => {
    const response = await api.post('/search', {
      query,
      top_k: options.top_k || 10,
      enable_domain_filtering: options.enable_domain_filtering !== false,
      include_behavioural_analysis: options.include_behavioural_analysis || false,
      ...options
    });
    return response.data;
  },

  // Health endpoints
  checkHealth: async () => {
    const response = await api.get('/api/health');
    return response.data;
  },

  getSystemStatus: async () => {
    const response = await api.get('/test');
    return response.data;
  },

  // Candidate endpoints
  getAllCandidates: async () => {
    const response = await api.get('/api/candidates/all');
    return response.data.candidates || [];
  },

  getDebugCandidates: async () => {
    const response = await api.get('/debug/candidates');
    return response.data;
  },

  // Performance endpoints
  getSearchPerformance: async () => {
    const response = await api.get('/api/search/performance');
    return response.data;
  },

  // Market Intelligence endpoints (if available)
  getMarketIntelligence: async () => {
    try {
      const response = await api.get('/market-intelligence');
      return response.data;
    } catch (error) {
      console.warn('Market intelligence not available:', error.message);
      return null;
    }
  },

  // Explainable AI endpoints (if available)
  getExplainableAI: async () => {
    try {
      const response = await api.get('/explainable-ai');
      return response.data;
    } catch (error) {
      console.warn('Explainable AI not available:', error.message);
      return null;
    }
  },

  // Compensation Intelligence endpoints (if available)
  getCompensationIntelligence: async () => {
    try {
      const response = await api.get('/compensation-intelligence');
      return response.data;
    } catch (error) {
      console.warn('Compensation intelligence not available:', error.message);
      return null;
    }
  }
};

// Utility functions
export const formatCandidateData = (candidate) => {
  return {
    id: candidate.email || `candidate-${Math.random()}`,
    email: candidate.email || 'N/A',
    fullName: candidate.full_name || 'Unknown',
    skills: Array.isArray(candidate.skills) ? candidate.skills : [],
    experienceYears: candidate.total_experience_years || 0,
    resumeText: candidate.resume_text || '',
    phone: candidate.phone || 'N/A',
    sourceURL: candidate.sourceURL || '',
    score: candidate.final_score || candidate.similarity_score || 0,
    grade: candidate.grade || 'N/A',
    domain: candidate.domain || 'general',
    aiExplanation: candidate.ai_explanation || null,
    confidenceLevel: candidate.confidence_level || null,
    recommendation: candidate.recommendation || null,
    riskFactors: candidate.risk_factors || [],
    strengthAreas: candidate.strength_areas_ai || [],
    behavioralAnalysis: candidate.behavioural_analysis || null,
    matchingAlgorithm: candidate.matching_algorithm || 'unknown',
    featureContributions: candidate.feature_contributions || {}
  };
};

export const formatSearchResults = (results) => {
  if (!Array.isArray(results)) return [];
  return results.map(formatCandidateData);
};

export const calculateScoreColor = (score) => {
  if (score >= 0.8) return '#10b981'; // Green
  if (score >= 0.6) return '#f59e0b'; // Yellow
  if (score >= 0.4) return '#f97316'; // Orange
  return '#ef4444'; // Red
};

export const getGradeColor = (grade) => {
  switch (grade?.toUpperCase()) {
    case 'A': return '#10b981';
    case 'B': return '#3b82f6';
    case 'C': return '#f59e0b';
    case 'D': return '#ef4444';
    default: return '#6b7280';
  }
};

export const formatSearchTime = (timeMs) => {
  if (timeMs < 1000) return `${timeMs.toFixed(0)}ms`;
  return `${(timeMs / 1000).toFixed(2)}s`;
};

export const formatFileSize = (bytes) => {
  if (bytes === 0) return '0 Bytes';
  const k = 1024;
  const sizes = ['Bytes', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
};

export default api;
