"""
Feedback Collection System
Collects recruiter actions and converts to training labels for continuous learning.
"""

import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)

class FeedbackAction(Enum):
    """Recruiter action types"""
    CANDIDATE_VIEWED = 'candidate_viewed'
    CANDIDATE_SHORTLISTED = 'candidate_shortlisted'
    INTERVIEW_SCHEDULED = 'interview_scheduled'
    INTERVIEW_COMPLETED = 'interview_completed'
    OFFER_EXTENDED = 'offer_extended'
    OFFER_ACCEPTED = 'offer_accepted'
    HIRED = 'hired'
    CANDIDATE_REJECTED = 'candidate_rejected'
    INTERVIEW_CANCELLED = 'interview_cancelled'
    OFFER_DECLINED = 'offer_declined'
    NOT_SELECTED = 'not_selected'
    CANDIDATE_CONTACTED = 'candidate_contacted'
    NO_RESPONSE = 'no_response'


@dataclass
class FeedbackRecord:
    """Feedback record"""
    feedback_id: str
    job_id: str
    candidate_id: str
    action: FeedbackAction
    label: float  # 1.0 for positive, 0.0 for negative, 0.5 for neutral
    timestamp: datetime
    recruiter_id: Optional[str] = None
    reason: Optional[str] = None
    metadata: Dict[str, Any] = None


class FeedbackCollector:
    """Collect and process feedback for training"""
    
    def __init__(self, db_connection=None):
        self.db = db_connection
        self.feedback_cache: List[FeedbackRecord] = []
        
        # Action to label mapping
        self.action_labels = {
            FeedbackAction.CANDIDATE_VIEWED: 0.3,
            FeedbackAction.CANDIDATE_SHORTLISTED: 0.7,
            FeedbackAction.INTERVIEW_SCHEDULED: 0.8,
            FeedbackAction.INTERVIEW_COMPLETED: 0.85,
            FeedbackAction.OFFER_EXTENDED: 0.9,
            FeedbackAction.OFFER_ACCEPTED: 0.95,
            FeedbackAction.HIRED: 1.0,
            FeedbackAction.CANDIDATE_REJECTED: 0.1,
            FeedbackAction.INTERVIEW_CANCELLED: 0.2,
            FeedbackAction.OFFER_DECLINED: 0.15,
            FeedbackAction.NOT_SELECTED: 0.0,
            FeedbackAction.CANDIDATE_CONTACTED: 0.4,
            FeedbackAction.NO_RESPONSE: 0.25
        }
    
    def record_feedback(
        self,
        job_id: str,
        candidate_id: str,
        action: str,
        recruiter_id: Optional[str] = None,
        reason: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> FeedbackRecord:
        """Record feedback action"""
        try:
            # Parse action
            if isinstance(action, str):
                action_enum = FeedbackAction(action.lower())
            else:
                action_enum = action
            
            # Get label
            label = self.action_labels.get(action_enum, 0.5)
            
            # Create feedback record
            feedback = FeedbackRecord(
                feedback_id=f"{job_id}_{candidate_id}_{datetime.now().timestamp()}",
                job_id=job_id,
                candidate_id=candidate_id,
                action=action_enum,
                label=label,
                timestamp=datetime.now(),
                recruiter_id=recruiter_id,
                reason=reason,
                metadata=metadata or {}
            )
            
            # Store in cache
            self.feedback_cache.append(feedback)
            
            # Store in database if available
            if self.db:
                self._store_in_db(feedback)
            
            logger.info(f"Feedback recorded: {action_enum.value} for job {job_id}, candidate {candidate_id}")
            
            return feedback
            
        except Exception as e:
            logger.error(f"Error recording feedback: {e}")
            raise
    
    def get_training_labels(
        self,
        job_id: str,
        candidate_ids: List[str],
        min_label_threshold: float = 0.5
    ) -> Dict[str, float]:
        """Get training labels for candidates"""
        labels = {}
        
        for candidate_id in candidate_ids:
            # Get most recent feedback for this job-candidate pair
            feedbacks = [
                f for f in self.feedback_cache
                if f.job_id == job_id and f.candidate_id == candidate_id
            ]
            
            if feedbacks:
                # Use most recent feedback
                latest = max(feedbacks, key=lambda x: x.timestamp)
                if latest.label >= min_label_threshold:
                    labels[candidate_id] = latest.label
            else:
                # No feedback, use default
                labels[candidate_id] = 0.5
        
        return labels
    
    def get_feedback_batch(
        self,
        limit: int = 1000,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[FeedbackRecord]:
        """Get batch of feedback for training"""
        feedbacks = self.feedback_cache.copy()
        
        # Filter by date
        if start_date:
            feedbacks = [f for f in feedbacks if f.timestamp >= start_date]
        if end_date:
            feedbacks = [f for f in feedbacks if f.timestamp <= end_date]
        
        # Sort by timestamp
        feedbacks.sort(key=lambda x: x.timestamp, reverse=True)
        
        return feedbacks[:limit]
    
    def _store_in_db(self, feedback: FeedbackRecord):
        """Store feedback in database"""
        if not self.db:
            return
        
        try:
            # Insert into feedback table
            # This would be your actual DB insert logic
            pass
        except Exception as e:
            logger.error(f"Error storing feedback in DB: {e}")


# Global instance
_feedback_collector = None

def get_feedback_collector(db_connection=None) -> FeedbackCollector:
    """Get or create global feedback collector instance"""
    global _feedback_collector
    if _feedback_collector is None:
        _feedback_collector = FeedbackCollector(db_connection)
    return _feedback_collector

