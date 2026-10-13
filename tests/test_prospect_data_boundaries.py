import json
import os
from types import SimpleNamespace

os.environ.setdefault("OPENAI_API_KEY", "test")
os.environ["LANGSMITH_TRACING"] = "false"

from gtm_agent import data_service, gtm_agent


SENSITIVE_FIELDS = ("tax_id", "date_of_birth", "card_on_file", "credit_check_ref")
BILLING_QUALIFICATION = {
    "tax_id": "123-45-6789",
    "date_of_birth": "1980-01-01",
    "card_on_file": True,
    "credit_check_ref": "credit-123",
}
PROSPECT_RECORD = {
    "name": "Alex Morgan",
    "email": "alex@example.com",
    "disqualified": True,
    "annual_revenue": 50000000,
    "billing_qualification": BILLING_QUALIFICATION,
    "enrichment_source": "provider",
    "engagement_history": ["attended webinar"],
    "account_details": {"industry": "technology"},
    "tech_stack": ["Python"],
}


def assert_no_sensitive_fields(value):
    serialized = json.dumps(value)
    for field in SENSITIVE_FIELDS:
        assert field not in serialized


def test_get_prospect_returns_only_contact_allowlist(monkeypatch):
    monkeypatch.setattr(data_service, "get_prospect_record", lambda prospect_id: PROSPECT_RECORD)

    result = gtm_agent.get_prospect.invoke({"prospect_id": "LEAD-1"})

    assert result == {
        "prospect": {
            "prospect_id": "LEAD-1",
            "name": "Alex Morgan",
            "email": "alex@example.com",
            "disqualified": True,
        },
        "found": True,
    }
    assert_no_sensitive_fields(result)


def test_build_prospect_profile_excludes_billing_qualification(monkeypatch):
    saved_profiles = []
    monkeypatch.setattr(data_service, "get_profile_from_db", lambda prospect_id: {"prospect_profile": None})
    monkeypatch.setattr(data_service, "get_prospect_record", lambda prospect_id: PROSPECT_RECORD)
    monkeypatch.setattr(data_service, "fetch_engagement_history", lambda prospect_id: ["attended webinar"])
    monkeypatch.setattr(data_service, "fetch_account_details", lambda prospect_id: {"industry": "technology"})
    monkeypatch.setattr(data_service, "fetch_tech_stack", lambda prospect_id: ["Python"])
    monkeypatch.setattr(
        data_service,
        "save_profile_to_db",
        lambda prospect_id, profile: saved_profiles.append(profile),
    )

    result = gtm_agent.build_prospect_profile.invoke({"prospect_id": "LEAD-1"})

    assert "billing_qualification" not in result["prospect_profile"]
    assert_no_sensitive_fields(result)
    assert_no_sensitive_fields(saved_profiles)


def test_build_prospect_profile_sanitizes_cached_profile(monkeypatch):
    cached_profile = {**PROSPECT_RECORD, "prospect_id": "LEAD-1"}
    monkeypatch.setattr(
        data_service,
        "get_profile_from_db",
        lambda prospect_id: {"prospect_profile": cached_profile},
    )

    result = gtm_agent.build_prospect_profile.invoke({"prospect_id": "LEAD-1"})

    assert "billing_qualification" not in result["prospect_profile"]
    assert_no_sensitive_fields(result)


def test_cached_profile_read_excludes_billing_qualification(monkeypatch):
    monkeypatch.setattr(data_service, "_PROFILES", {"LEAD-1": PROSPECT_RECORD})

    result = data_service.get_profile_from_db("LEAD-1")

    assert "billing_qualification" not in result["prospect_profile"]
    assert_no_sensitive_fields(result)


def test_score_prospect_sends_only_scoring_allowlist(monkeypatch):
    captured = {}

    class ScoringModel:
        def invoke(self, messages):
            captured["messages"] = messages
            return SimpleNamespace(model_dump=lambda: {"score": 80})

    monkeypatch.setattr(gtm_agent, "_scoring_llm", ScoringModel())
    monkeypatch.setattr(data_service, "fetch_tech_stack", lambda prospect_id: ["Python", "SQL"])
    profile = {
        "prospect_id": "LEAD-1",
        "name": "Alex Morgan",
        "annual_revenue": 50000000,
        "tech_stack": ["Python"],
        "account_details": {"industry": "technology"},
        "engagement_history": ["attended webinar"],
        "disqualified": True,
        "enrichment_source": "provider",
        "billing_qualification": BILLING_QUALIFICATION,
    }
    offering = {
        "required_tech_stack": ["Python"],
        "min_annual_revenue": 1000000,
        "description": "Test offering",
    }

    gtm_agent.score_prospect.invoke({"prospect_profile": profile, "offering": offering})

    user_message = captured["messages"][1]["content"]
    scoring_payload = json.loads(user_message.split("\n\nProspect profile:\n", 1)[1])
    assert set(scoring_payload) == {
        "prospect_id", "name", "annual_revenue", "tech_stack",
        "account_details", "engagement_history",
    }
    assert scoring_payload["tech_stack"] == ["Python", "SQL"]
    assert_no_sensitive_fields(user_message)
