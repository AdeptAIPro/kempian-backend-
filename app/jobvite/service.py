"""
Unified Jobvite Integration Service
Facade that exposes a clean API for all Jobvite operations.
"""

from typing import Dict, List, Optional, Any
from app.jobvite.client_v2 import JobviteV2Client
from app.jobvite.client_onboarding import JobviteOnboardingClient
from app.simple_logger import get_logger

logger = get_logger("jobvite_service")


class JobviteIntegrationService:
    """
    Unified service facade for Jobvite integration.
    
    Provides a clean, consistent API for all Jobvite operations:
    - v2 API endpoints (jobs, candidates, applications, etc.)
    - Onboarding API endpoints (processes, tasks, milestones)
    """
    
    def __init__(self, v2_client: JobviteV2Client, onboarding_client: JobviteOnboardingClient):
        """
        Initialize the service with v2 and onboarding clients.
        
        Args:
            v2_client: JobviteV2Client instance
            onboarding_client: JobviteOnboardingClient instance
        """
        self.v2 = v2_client
        self.ob = onboarding_client
    
    # ==================== v2 API Methods ====================
    
    def getJobs(self, job_id: Optional[str] = None, 
                requisition_id: Optional[str] = None,
                filters: Optional[Dict] = None,
                start: int = 0,
                count: int = 50) -> Dict[str, Any]:
        """Get job(s) from Jobvite."""
        return self.v2.get_job(job_id=job_id, requisition_id=requisition_id, 
                              filters=filters, start=start, count=count)
    
    def getCandidates(self, candidate_id: Optional[str] = None,
                     application_id: Optional[str] = None,
                     filters: Optional[Dict] = None,
                     start: int = 0,
                     count: int = 50) -> Dict[str, Any]:
        """Get candidate(s) from Jobvite."""
        return self.v2.get_candidate(candidate_id=candidate_id, 
                                    application_id=application_id,
                                    filters=filters, start=start, count=count)
    
    def getApplications(self, application_id: Optional[str] = None,
                       filters: Optional[Dict] = None,
                       start: int = 0,
                       count: int = 50) -> Dict[str, Any]:
        """Get application(s) from Jobvite."""
        return self.v2.get_application(application_id=application_id,
                                       filters=filters, start=start, count=count)
    
    def getApplicationHistory(self, application_id: str,
                             start: int = 0,
                             count: int = 50) -> Dict[str, Any]:
        """Get application history."""
        return self.v2.get_application_history(application_id=application_id,
                                               start=start, count=count)
    
    def getEmployee(self, employee_id: Optional[str] = None,
                   filters: Optional[Dict] = None,
                   start: int = 0,
                   count: int = 50) -> Dict[str, Any]:
        """Get employee(s) from Jobvite."""
        return self.v2.get_employee(employee_id=employee_id,
                                    filters=filters, start=start, count=count)
    
    def getContact(self, contact_id: Optional[str] = None,
                  filters: Optional[Dict] = None,
                  start: int = 0,
                  count: int = 50) -> Dict[str, Any]:
        """Get contact(s) from Jobvite."""
        return self.v2.get_contact(contact_id=contact_id,
                                   filters=filters, start=start, count=count)
    
    def getOffer(self, offer_id: Optional[str] = None,
                filters: Optional[Dict] = None,
                start: int = 0,
                count: int = 50) -> Dict[str, Any]:
        """Get offer(s) from Jobvite."""
        return self.v2.get_offer(offer_id=offer_id,
                                 filters=filters, start=start, count=count)
    
    def getInterview(self, interview_id: Optional[str] = None,
                    filters: Optional[Dict] = None,
                    start: int = 0,
                    count: int = 50) -> Dict[str, Any]:
        """Get interview(s) from Jobvite."""
        return self.v2.get_interview(interview_id=interview_id,
                                     filters=filters, start=start, count=count)
    
    def getCustomField(self, filters: Optional[Dict] = None) -> Dict[str, Any]:
        """Get custom fields from Jobvite."""
        return self.v2.get_customfield(filters=filters)
    
    def getCandidateWithArtifacts(self, candidate_id: str) -> Dict[str, Any]:
        """Get candidate with encoded artifacts (documents)."""
        return self.v2.get_candidate_with_artifacts(candidate_id)
    
    # ==================== Webhook Management ====================
    
    def listWebhooks(self, filters: Optional[Dict] = None) -> Dict[str, Any]:
        """List webhooks configured in Jobvite."""
        return self.v2.list_webhooks(filters=filters)
    
    def createWebhook(self, webhook_config: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new webhook in Jobvite."""
        return self.v2.create_webhook(webhook_config)
    
    def deleteWebhook(self, webhook_id: str) -> Dict[str, Any]:
        """Delete a webhook from Jobvite."""
        return self.v2.delete_webhook(webhook_id)
    
    # ==================== Onboarding API Methods ====================
    
    def postOnboardingProcess(self, filters: Optional[Dict] = None,
                             start: int = 0,
                             count: int = 50,
                             default_sort_type: str = "ASC") -> Dict[str, Any]:
        """
        Get onboarding processes.
        
        Args:
            filters: Filter criteria using operators (eq, in, nin, lt, lte, gt, gte)
            start: Pagination start index
            count: Page size
            default_sort_type: "ASC" or "DESC"
        
        Returns:
            Decrypted process data
        """
        return self.ob.get_processes(filters=filters, start=start, count=count,
                                     default_sort_type=default_sort_type)
    
    def postOnboardingTask(self, filters: Optional[Dict] = None,
                          start: int = 0,
                          count: int = 50,
                          return_file_info: bool = False) -> Dict[str, Any]:
        """
        Get onboarding tasks.
        
        Args:
            filters: Filter criteria using operators
            start: Pagination start index
            count: Page size
            return_file_info: If True, include file data (base64 encoded)
        
        Returns:
            Decrypted task data
        """
        return self.ob.get_tasks(filters=filters, start=start, count=count,
                                return_file_info=return_file_info)
    
    def postOnboardingMilestone(self, process_ids: List[str],
                               milestones: List[str],
                               operation: str = "add",
                               milestone_type: str = "api_retrieved") -> Dict[str, Any]:
        """
        Mark milestone(s) for process(es) as API retrieved.
        
        Args:
            process_ids: List of Jobvite process IDs
            milestones: List of milestone names to mark
            operation: "set" (replace all) or "add" (append)
            milestone_type: Type of milestone operation (default: "api_retrieved")
        
        Returns:
            Response from milestone API
        """
        return self.ob.set_milestone(process_ids=process_ids, milestones=milestones,
                                    operation=operation, milestone_type=milestone_type)

