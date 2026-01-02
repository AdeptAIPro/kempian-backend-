"""
Document retrieval endpoints for Jobvite integration.
Provides secure access to candidate documents stored in S3.
"""

from flask import Blueprint, jsonify, send_file, g
from app.models import JobviteCandidateDocument, db
from app.utils import get_current_user
from app.jobvite.storage import get_document_from_s3, get_document_url
from app.simple_logger import get_logger
from io import BytesIO

logger = get_logger("jobvite_documents")

documents_bp = Blueprint('jobvite_documents', __name__)

@documents_bp.route('/api/integrations/jobvite/candidates/<int:candidate_id>/documents', methods=['GET'])
def get_candidate_documents(candidate_id: int):
    """Get list of documents for a candidate"""
    user_jwt = get_current_user()
    if not user_jwt:
        return jsonify({'error': 'Unauthorized'}), 401
    
    from app.models import User, JobviteCandidate
    user = User.query.filter_by(email=user_jwt['email']).first()
    if not user:
        return jsonify({'error': 'User not found'}), 404
    
    tenant_id = g.tenant_id or user.tenant_id
    
    # Verify candidate belongs to tenant
    candidate = JobviteCandidate.query.filter_by(
        id=candidate_id,
        tenant_id=tenant_id
    ).first()
    
    if not candidate:
        return jsonify({'error': 'Candidate not found'}), 404
    
    documents = JobviteCandidateDocument.query.filter_by(candidate_id=candidate_id).all()
    
    return jsonify({
        'documents': [{
            'id': doc.id,
            'docType': doc.doc_type,
            'filename': doc.filename,
            'mimeType': doc.mime_type,
            'sizeBytes': doc.size_bytes,
            'hasStorage': bool(doc.storage_path),
            'externalUrl': doc.external_url,
            'storagePath': doc.storage_path,
            'createdAt': doc.created_at.isoformat() if doc.created_at else None
        } for doc in documents]
    }), 200

@documents_bp.route('/api/integrations/jobvite/documents/<int:document_id>/download', methods=['GET'])
def download_document(document_id: int):
    """Download a document (returns presigned URL or file)"""
    user_jwt = get_current_user()
    if not user_jwt:
        return jsonify({'error': 'Unauthorized'}), 401
    
    from app.models import User
    user = User.query.filter_by(email=user_jwt['email']).first()
    if not user:
        return jsonify({'error': 'User not found'}), 404
    
    tenant_id = g.tenant_id or user.tenant_id
    
    # Get document
    document = JobviteCandidateDocument.query.get(document_id)
    if not document:
        return jsonify({'error': 'Document not found'}), 404
    
    # Verify candidate belongs to tenant
    candidate = document.candidate
    if candidate.tenant_id != tenant_id:
        return jsonify({'error': 'Unauthorized'}), 403
    
    # If document is stored in S3, return presigned URL
    if document.storage_path and document.storage_path.startswith('jobvite/documents/'):
        presigned_url = get_document_url(document.storage_path, expires_in=3600)
        if presigned_url:
            return jsonify({
                'downloadUrl': presigned_url,
                'filename': document.filename,
                'mimeType': document.mime_type
            }), 200
        else:
            return jsonify({'error': 'Failed to generate download URL'}), 500
    
    # Fallback: try to retrieve from S3
    if document.storage_path:
        file_content = get_document_from_s3(document.storage_path)
        if file_content:
            return send_file(
                BytesIO(file_content),
                mimetype=document.mime_type or 'application/octet-stream',
                as_attachment=True,
                download_name=document.filename
            )
    
    return jsonify({'error': 'Document not available'}), 404

