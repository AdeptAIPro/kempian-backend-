import io
import json
from datetime import datetime

from flask import Blueprint, jsonify, request, send_file, url_for
from werkzeug.utils import secure_filename

from app.auth_utils import get_current_user_flexible
from app.models import (
    db,
    User,
    UserBankAccount,
    UserBankDocument,
)
from app.simple_logger import get_logger

logger = get_logger("user_bank_account")

user_bp = Blueprint("user_api", __name__)

# Register activity logs routes
from app.user.activity_logs_routes import activity_logs_bp
user_bp.register_blueprint(activity_logs_bp)

DOCUMENT_FIELD_MAP = {
    "voidedCheckOrBankLetter": "voided_check",
    "signedServiceAgreement": "signed_service_agreement",
    "powerOfAttorney": "power_of_attorney",
    "businessRegistrationCertificate": "business_registration_certificate",
}

DOCUMENT_URL_KEY_MAP = {
    "voided_check": "voidedCheckOrBankLetterUrl",
    "signed_service_agreement": "signedServiceAgreementUrl",
    "power_of_attorney": "powerOfAttorneyUrl",
    "business_registration_certificate": "businessRegistrationCertificateUrl",
}

DOCUMENT_MODEL_URL_ATTR = {
    "voided_check": "voided_check_or_bank_letter_url",
    "signed_service_agreement": "signed_service_agreement_url",
    "power_of_attorney": "power_of_attorney_url",
    "business_registration_certificate": "business_registration_certificate_url",
}


def _resolve_authenticated_user():
    """Resolve the authenticated user from the request headers."""
    auth_payload = get_current_user_flexible()
    if not auth_payload or not auth_payload.get("email"):
        return None, jsonify({"error": "Unauthorized"}), 401

    user = User.query.filter_by(email=auth_payload["email"]).first()
    if not user:
        return None, jsonify({"error": "User not found"}), 404

    return user, None, None


def _to_bool(value):
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return value != 0
    return str(value).strip().lower() in {"true", "1", "yes", "on"}


def _parse_company_states(raw_value):
    if raw_value is None:
        return []

    if isinstance(raw_value, list):
        return raw_value

    try:
        parsed = json.loads(raw_value)
        if isinstance(parsed, list):
            return parsed
    except (TypeError, ValueError):
        pass

    if isinstance(raw_value, str):
        return [state.strip() for state in raw_value.split(",") if state.strip()]

    return []


def _get_or_create_bank_account(user):
    account = UserBankAccount.query.filter_by(user_id=user.id).first()
    if account:
        return account

    account = UserBankAccount(user_id=user.id)
    db.session.add(account)
    return account


def _serialize_account(account):
    document_urls = {}
    for document in account.documents:
        if document.doc_type in DOCUMENT_URL_KEY_MAP:
            try:
                document_urls[document.doc_type] = url_for(
                    "user_api.download_bank_document",
                    doc_type=document.doc_type,
                    _external=False,
                )
            except RuntimeError:
                # url_for requires an active request context; skip if unavailable
                document_urls[document.doc_type] = None

    payload = account.to_dict(document_urls=document_urls)
    for doc_type, url_key in DOCUMENT_URL_KEY_MAP.items():
        if doc_type in document_urls and document_urls[doc_type]:
            payload[url_key] = document_urls[doc_type]
    return payload


def _handle_document_upload(account, form_data):
    for field_name, doc_type in DOCUMENT_FIELD_MAP.items():
        remove_flag = form_data.get(f"{field_name}Remove")
        if remove_flag is not None and _to_bool(remove_flag):
            existing = next(
                (doc for doc in account.documents if doc.doc_type == doc_type), None
            )
            if existing:
                db.session.delete(existing)
            continue

        upload = request.files.get(field_name)
        if not upload or not upload.filename:
            continue

        file_name = secure_filename(upload.filename)
        file_bytes = upload.read()
        if not file_bytes:
            continue

        existing_doc = next(
            (doc for doc in account.documents if doc.doc_type == doc_type), None
        )
        if existing_doc:
            target_doc = existing_doc
        else:
            target_doc = UserBankDocument(doc_type=doc_type)
            account.documents.append(target_doc)

        target_doc.file_name = file_name
        target_doc.content_type = upload.mimetype or "application/octet-stream"
        target_doc.file_size = len(file_bytes)
        target_doc.data = file_bytes

        # Clear any externally stored URL when storing binary data internally
        url_attr = DOCUMENT_MODEL_URL_ATTR.get(doc_type)
        if url_attr:
            setattr(account, url_attr, None)


@user_bp.route("/bank-account", methods=["GET"])
def get_bank_account():
    user, error_response, status = _resolve_authenticated_user()
    if error_response:
        return error_response, status

    account = UserBankAccount.query.filter_by(user_id=user.id).first()
    if not account:
        return jsonify({"data": None}), 200

    return jsonify({"data": _serialize_account(account)}), 200


@user_bp.route("/bank-account", methods=["POST"])
def upsert_bank_account():
    user, error_response, status = _resolve_authenticated_user()
    if error_response:
        return error_response, status

    if request.content_type and request.content_type.startswith("application/json"):
        form_data = request.get_json() or {}
    else:
        form_data = request.form.to_dict()

    account = _get_or_create_bank_account(user)

    account.owner_or_authorized_rep_name = (
        form_data.get("ownerOrAuthorizedRepName")
        or form_data.get("accountHolderName")
        or ""
    )
    account.title_or_role = form_data.get("titleOrRole")
    account.contact_email = form_data.get("contactEmail")
    account.contact_phone = form_data.get("contactPhone")
    account.payroll_contact_person = form_data.get("payrollContactPerson")
    account.is_payroll_contact_different = _to_bool(
        form_data.get("isPayrollContactDifferent")
    )
    account.account_holder_name = (
        form_data.get("accountHolderName")
        or form_data.get("ownerOrAuthorizedRepName")
        or ""
    )

    account.bank_name = form_data.get("bankName")
    account.routing_number = form_data.get("routingNumber")
    account.account_number = form_data.get("accountNumber")

    account.pay_frequency = form_data.get("payFrequency")

    first_intended_pay_date = form_data.get("firstIntendedPayDate")
    if first_intended_pay_date:
        try:
            account.first_intended_pay_date = datetime.strptime(
                first_intended_pay_date, "%Y-%m-%d"
            ).date()
        except ValueError:
            logger.warning(
                "Invalid firstIntendedPayDate provided by user_id=%s: %s",
                user.id,
                first_intended_pay_date,
            )
    else:
        account.first_intended_pay_date = None

    account.direct_deposit_or_check_preference = form_data.get(
        "directDepositOrCheckPreference"
    )
    account.third_party_integrations = form_data.get("thirdPartyIntegrations")

    account.tax_filing_responsibility = form_data.get("taxFilingResponsibility")
    account.state_unemployment_account_info = form_data.get(
        "stateUnemploymentAccountInfo"
    )
    account.workers_comp_carrier = form_data.get("workersCompCarrier")
    account.workers_comp_policy_number = form_data.get("workersCompPolicyNumber")
    account.company_registration_states = _parse_company_states(
        form_data.get("companyRegistrationStates")
    )

    # Handle optional external URLs (if provided)
    if "voidedCheckOrBankLetterUrl" in form_data:
        account.voided_check_or_bank_letter_url = form_data.get(
            "voidedCheckOrBankLetterUrl"
        )
    if "signedServiceAgreementUrl" in form_data:
        account.signed_service_agreement_url = form_data.get(
            "signedServiceAgreementUrl"
        )
    if "powerOfAttorneyUrl" in form_data:
        account.power_of_attorney_url = form_data.get("powerOfAttorneyUrl")
    if "businessRegistrationCertificateUrl" in form_data:
        account.business_registration_certificate_url = form_data.get(
            "businessRegistrationCertificateUrl"
        )

    _handle_document_upload(account, form_data)

    try:
        db.session.commit()
    except Exception as exc:
        logger.error("Failed to save bank account for user_id=%s: %s", user.id, exc)
        db.session.rollback()
        return jsonify({"error": "Failed to save bank account data"}), 500

    return jsonify({"data": _serialize_account(account)}), 200


@user_bp.route("/bank-account", methods=["DELETE"])
def delete_bank_account():
    user, error_response, status = _resolve_authenticated_user()
    if error_response:
        return error_response, status

    account = UserBankAccount.query.filter_by(user_id=user.id).first()
    if not account:
        return jsonify({"message": "No bank account data to delete"}), 200

    try:
        db.session.delete(account)
        db.session.commit()
        return jsonify({"message": "Bank account data deleted"}), 200
    except Exception as exc:
        logger.error("Failed to delete bank account for user_id=%s: %s", user.id, exc)
        db.session.rollback()
        return jsonify({"error": "Failed to delete bank account data"}), 500


@user_bp.route("/bank-account/documents/<string:doc_type>", methods=["GET"])
def download_bank_document(doc_type):
    user, error_response, status = _resolve_authenticated_user()
    if error_response:
        return error_response, status

    if doc_type not in DOCUMENT_URL_KEY_MAP:
        return jsonify({"error": "Invalid document type"}), 400

    account = UserBankAccount.query.filter_by(user_id=user.id).first()
    if not account:
        return jsonify({"error": "Bank account data not found"}), 404

    document = next(
        (doc for doc in account.documents if doc.doc_type == doc_type),
        None,
    )
    if not document:
        return jsonify({"error": "Document not found"}), 404

    return send_file(
        io.BytesIO(document.data),
        mimetype=document.content_type or "application/octet-stream",
        as_attachment=True,
        download_name=document.file_name or f"{doc_type}.bin",
    )

