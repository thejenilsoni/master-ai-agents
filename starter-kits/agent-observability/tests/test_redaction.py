"""Redaction must actually remove the secret, not merely mask part of it."""

from __future__ import annotations

import re

from obs.redaction import Redactor, redact

# Obviously synthetic values. They are shaped like credentials so the patterns fire,
# but they are not valid for any service.
FAKE_PROVIDER_KEY = "sk-notarealkey000111222333444555666777888"
FAKE_AWS_KEY_ID = "AKIAEXAMPLEEXAMPLE00"
FAKE_JWT = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIwMDAwIn0.c2lnbmF0dXJlLXBsYWNlaG9sZGVy"


def test_provider_style_key_is_removed_from_free_text() -> None:
    text = f"calling the API with {FAKE_PROVIDER_KEY} now"
    cleaned = Redactor().redact_text(text)
    assert FAKE_PROVIDER_KEY not in cleaned
    assert "notarealkey" not in cleaned
    assert "[REDACTED]" in cleaned


def test_keyed_secret_keeps_the_label_but_drops_the_value() -> None:
    cleaned = Redactor().redact_text('{"api_key": "hunter2-super-secret"}')
    assert "hunter2" not in cleaned
    assert "api_key" in cleaned


def test_authorization_bearer_header_is_removed() -> None:
    cleaned = Redactor().redact_text(f"Authorization: Bearer {FAKE_JWT}")
    assert FAKE_JWT not in cleaned
    assert "signature" not in cleaned.lower()


def test_aws_access_key_id_is_removed() -> None:
    cleaned = Redactor().redact_text(f"aws id {FAKE_AWS_KEY_ID} in the log line")
    assert FAKE_AWS_KEY_ID not in cleaned


def test_email_and_card_and_ssn_are_removed() -> None:
    text = "contact ada@example.com card 4111 1111 1111 1111 ssn 123-45-6789"
    cleaned = Redactor().redact_text(text)
    assert "ada@example.com" not in cleaned
    assert "4111" not in cleaned
    assert "123-45-6789" not in cleaned


def test_sensitive_mapping_keys_are_dropped_whatever_the_value_looks_like() -> None:
    payload = {
        "prompt": "summarise the ticket",
        "authorization": "opaque-value-with-no-recognisable-shape",
        "nested": {"session_id": "abc123", "safe": "keep me"},
    }
    result = Redactor().scrub(payload)
    assert result.value["authorization"] == "[REDACTED]"
    assert result.value["nested"]["session_id"] == "[REDACTED]"
    assert result.value["nested"]["safe"] == "keep me"
    assert result.value["prompt"] == "summarise the ticket"
    assert "sensitive-key" in result.categories


def test_secret_inside_a_nested_list_is_removed() -> None:
    payload = {"messages": [{"role": "user", "content": f"key={FAKE_PROVIDER_KEY}"}]}
    cleaned = redact(payload)
    assert FAKE_PROVIDER_KEY not in str(cleaned)


def test_recursion_is_bounded() -> None:
    deep: dict[str, object] = {"level": 0}
    node = deep
    for i in range(1, 40):
        child: dict[str, object] = {"level": i}
        node["child"] = child
        node = child
    cleaned = redact(deep)
    assert "[REDACTED]" in str(cleaned)


def test_bytes_are_never_logged_verbatim() -> None:
    cleaned = redact({"blob": b"raw-bytes-that-might-be-anything"})
    assert cleaned["blob"] == "[REDACTED]"


def test_extra_domain_pattern_can_be_added() -> None:
    redactor = Redactor(extra_patterns=(("cust-ref", re.compile(r"\bCUST-\d{6}\b")),))
    result = redactor.scrub("ticket for CUST-004821")
    assert "CUST-004821" not in str(result.value)
    assert "cust-ref" in result.categories


def test_oversized_text_is_truncated_before_matching() -> None:
    redactor = Redactor(max_text_length=100)
    cleaned = redactor.redact_text("a" * 500)
    assert cleaned.endswith("...[truncated]")
    assert len(cleaned) < 500


def test_clean_payload_reports_no_categories() -> None:
    result = Redactor().scrub({"question": "how do refunds work", "count": 3})
    assert result.redacted is False
    assert result.categories == ()
