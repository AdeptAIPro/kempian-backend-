"""
Skill Demand Forecasting

Predicts future skill demand using time-series forecasting models.
"""

import os
import pickle
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

# Try to import Prophet
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    logger.warning("Prophet not available. Install with: pip install prophet")

# Try to import statsmodels
try:
    from statsmodels.tsa.arima.model import ARIMA
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    logger.warning("statsmodels not available. Install with: pip install statsmodels")


class SkillDemandForecaster:
    """
    Skill Demand Forecaster
    
    Predicts future skill demand using time-series forecasting
    """
    
    def __init__(self):
        """Initialize Skill Demand Forecaster"""
        self.models = {}  # One model per skill
        self.historical_data = {}  # Historical data per skill
        
        # Model type preference
        self.use_prophet = PROPHET_AVAILABLE
        self.use_arima = STATSMODELS_AVAILABLE and not PROPHET_AVAILABLE
    
    def add_historical_data(self, skill: str, data: List[Dict[str, Any]]):
        """
        Add historical demand data for a skill
        
        Args:
            skill: Skill name
            data: List of dicts with 'date' and 'demand' (job posting count)
        """
        if skill not in self.historical_data:
            self.historical_data[skill] = []
        
        self.historical_data[skill].extend(data)
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        df['ds'] = pd.to_datetime(df['date'])
        df['y'] = df['demand']
        
        # Store
        self.historical_data[skill] = df[['ds', 'y']]
    
    def train_model(self, skill: str):
        """
        Train forecasting model for a skill
        
        Args:
            skill: Skill name
        """
        if skill not in self.historical_data:
            logger.warning(f"No historical data for skill: {skill}")
            return
        
        df = self.historical_data[skill]
        
        if len(df) < 10:
            logger.warning(f"Insufficient data for skill {skill}. Need at least 10 data points.")
            return
        
        try:
            if self.use_prophet:
                model = Prophet()
                model.fit(df)
                self.models[skill] = model
                logger.info(f"Trained Prophet model for skill: {skill}")
            elif self.use_arima:
                # Simple ARIMA model
                model = ARIMA(df['y'], order=(1, 1, 1))
                fitted_model = model.fit()
                self.models[skill] = fitted_model
                logger.info(f"Trained ARIMA model for skill: {skill}")
            else:
                logger.warning("No forecasting library available")
        except Exception as e:
            logger.error(f"Failed to train model for skill {skill}: {e}")
    
    def forecast_skill_demand(self, skill: str, months: int = 6) -> Dict[str, Any]:
        """
        Forecast skill demand for future months
        
        Args:
            skill: Skill name
            months: Number of months to forecast
            
        Returns:
            Dictionary with forecast information
        """
        if skill not in self.models:
            logger.warning(f"No model trained for skill: {skill}")
            return {
                'skill': skill,
                'error': 'No model trained'
            }
        
        model = self.models[skill]
        
        try:
            if self.use_prophet:
                # Create future dataframe
                future = model.make_future_dataframe(periods=months * 30)  # Approximate months to days
                forecast = model.predict(future)
                
                # Get last forecast point
                last_idx = len(forecast) - 1
                
                return {
                    'skill': skill,
                    'current_demand': float(self.historical_data[skill]['y'].iloc[-1]),
                    'predicted_demand': float(forecast['yhat'].iloc[last_idx]),
                    'trend': 'increasing' if forecast['trend'].iloc[last_idx] > forecast['trend'].iloc[last_idx - 1] else 'decreasing',
                    'confidence_interval': (
                        float(forecast['yhat_lower'].iloc[last_idx]),
                        float(forecast['yhat_upper'].iloc[last_idx])
                    ),
                    'forecast_date': forecast['ds'].iloc[last_idx].strftime('%Y-%m-%d')
                }
            elif self.use_arima:
                # ARIMA forecast
                forecast = model.forecast(steps=months)
                conf_int = model.get_forecast(steps=months).conf_int()
                
                return {
                    'skill': skill,
                    'current_demand': float(self.historical_data[skill]['y'].iloc[-1]),
                    'predicted_demand': float(forecast.iloc[-1]),
                    'trend': 'increasing' if forecast.iloc[-1] > forecast.iloc[0] else 'decreasing',
                    'confidence_interval': (
                        float(conf_int.iloc[-1, 0]),
                        float(conf_int.iloc[-1, 1])
                    ),
                    'forecast_months': months
                }
        except Exception as e:
            logger.error(f"Forecast failed for skill {skill}: {e}")
            return {
                'skill': skill,
                'error': str(e)
            }
    
    def get_trending_skills(self, top_n: int = 10) -> List[Dict[str, Any]]:
        """
        Get trending skills (increasing demand)
        
        Args:
            top_n: Number of top trending skills to return
            
        Returns:
            List of trending skills with forecast data
        """
        trending = []
        
        for skill in self.models.keys():
            forecast = self.forecast_skill_demand(skill, months=1)
            if 'error' not in forecast and forecast.get('trend') == 'increasing':
                trending.append(forecast)
        
        # Sort by predicted demand increase
        trending.sort(key=lambda x: x.get('predicted_demand', 0), reverse=True)
        
        return trending[:top_n]


# Global instance
_forecaster = None


def get_forecaster() -> SkillDemandForecaster:
    """Get or create global forecaster instance"""
    global _forecaster
    if _forecaster is None:
        _forecaster = SkillDemandForecaster()
    return _forecaster

