import base64
import hashlib
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from urllib.parse import parse_qs, urlencode, urlsplit

import jwt
import pytest
import requests
from cryptography.hazmat.primitives.asymmetric import rsa

from infra.chatgpt_auth import AUTH_ORIGIN, SCOPES, ChatGPTAccounts, ChatGPTAuthError, _credentials, _metadata


def profile(expires=0):
    return {
        "client_id": "oaiapp_test",
        "subject": "user-test",
        "email": "test@example.com",
        "access_token": "old-access",
        "refresh_token": "old-refresh",
        "id_token": "old-id",
        "expires_at": expires,
        "scopes": SCOPES.split(),
    }


def seed(accounts, expires=0):
    accounts._write(
        {"host_id": "urn:uuid:test", "active": "oaiapp_test", "profiles": {"oaiapp_test": profile(expires)}}
    )


def metadata():
    return {
        "issuer": AUTH_ORIGIN,
        "authorization_endpoint": AUTH_ORIGIN + "/api/accounts/authorize",
        "token_endpoint": AUTH_ORIGIN + "/api/accounts/oauth/token",
        "jwks_uri": AUTH_ORIGIN + "/jwks",
        "revocation_endpoint": AUTH_ORIGIN + "/revoke",
    }


def test_file_permissions_and_valid_access_need_no_network(tmp_path, monkeypatch):
    accounts = ChatGPTAccounts(tmp_path / "private" / "accounts.json")
    seed(accounts, time.time() + 3600)
    monkeypatch.setattr("infra.chatgpt_auth._metadata", lambda: pytest.fail("No refresh needed"))
    assert accounts.access_token() == "old-access"
    if os.name != "nt":
        assert accounts.path.stat().st_mode & 0o777 == 0o600
    assert not list(accounts.path.parent.glob(".chatgpt-*"))


def test_concurrent_refresh_rotates_once(tmp_path, monkeypatch):
    accounts = ChatGPTAccounts(tmp_path / "accounts.json")
    seed(accounts)
    calls = []
    monkeypatch.setattr("infra.chatgpt_auth._metadata", metadata)

    def token_request(endpoint, data):
        calls.append(data)
        return {"access_token": "new-access", "refresh_token": "new-refresh", "expires_in": 3600}

    monkeypatch.setattr("infra.chatgpt_auth._token_request", token_request)
    with ThreadPoolExecutor(max_workers=4) as pool:
        assert list(pool.map(lambda _: ChatGPTAccounts(accounts.path).access_token(), range(4))) == ["new-access"] * 4
    assert len(calls) == 1
    assert calls[0]["client_id"] == "oaiapp_test"
    assert accounts.profile()["refresh_token"] == "new-refresh"
    assert accounts.read()["host_id"] == "urn:uuid:test"


def test_missing_credentials_and_denied_permission(tmp_path):
    accounts = ChatGPTAccounts(tmp_path / "accounts.json")
    with pytest.raises(ChatGPTAuthError, match="login"):
        accounts.profile()
    with pytest.raises(ChatGPTAuthError, match="permission"):
        _credentials({"scope": "openid"}, "oaiapp_test", {})
    with pytest.raises(ChatGPTAuthError, match="Incomplete"):
        _credentials({"scope": SCOPES}, "oaiapp_test", {})
    accounts.path.write_text("not-json")
    with pytest.raises(ChatGPTAuthError, match="Invalid"):
        accounts.read()


@pytest.mark.parametrize("bad_nonce,bad_audience", [(False, False), (True, False), (False, True)])
def test_oauth_real_loopback_pkce_and_signed_identity(tmp_path, monkeypatch, bad_nonce, bad_audience):
    accounts = ChatGPTAccounts(tmp_path / "accounts.json")
    private = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    monkeypatch.setattr("infra.chatgpt_auth._metadata", metadata)
    monkeypatch.setattr(
        "jwt.PyJWKClient.get_signing_key_from_jwt", lambda self, token: SimpleNamespace(key=private.public_key())
    )
    captured, callbacks = {}, []

    def browser(url):
        captured.update({k: v[0] for k, v in parse_qs(urlsplit(url).query).items()})

        def complete():
            # A foreign callback must not complete or replace this sign-in attempt.
            bad = requests.get(captured["redirect_uri"] + "?" + urlencode({"state": "wrong", "code": "bad"}), timeout=3)
            assert bad.status_code == 400
            good = requests.get(
                captured["redirect_uri"]
                + "?"
                + urlencode({"state": captured["state"], "client_id": "oaiapp_new", "code": "fake-code"}),
                timeout=3,
            )
            assert good.status_code == 200

        thread = threading.Thread(target=complete)
        callbacks.append(thread)
        thread.start()
        return True

    def exchange(endpoint, data):
        assert data["client_id"] == "oaiapp_new" and data["client_id"] != "dynamic_agent_client"
        assert data["redirect_uri"] == captured["redirect_uri"]
        assert (
            captured["code_challenge"]
            == base64.urlsafe_b64encode(hashlib.sha256(data["code_verifier"].encode()).digest()).rstrip(b"=").decode()
        )
        token = jwt.encode(
            {
                "sub": "user-test",
                "email": "test@example.com",
                "iss": AUTH_ORIGIN,
                "aud": "wrong" if bad_audience else "oaiapp_new",
                "exp": time.time() + 3600,
                "nonce": "wrong" if bad_nonce else captured["nonce"],
            },
            private,
            algorithm="RS256",
        )
        return {
            "access_token": "new-access",
            "refresh_token": "new-refresh",
            "id_token": token,
            "scope": SCOPES,
            "expires_in": 3600,
        }

    monkeypatch.setattr("infra.chatgpt_auth.webbrowser.open", browser)
    monkeypatch.setattr("infra.chatgpt_auth._token_request", exchange)
    if bad_nonce or bad_audience:
        with pytest.raises(ChatGPTAuthError, match="nonce|validation"):
            accounts.login(timeout=5)
        assert accounts.read()["profiles"] == {}
    else:
        result = accounts.login(timeout=5)
        assert result["subject"] == "user-test"
        assert accounts.profile()["client_id"] == "oaiapp_new"
        assert captured["client_id"] == "dynamic_agent_client"
        assert captured["agent_name_hint"] == "CodeBoarding"
        assert captured["ext_agent_host_id"].startswith("urn:uuid:")
    for thread in callbacks:
        thread.join(timeout=3)


def test_refresh_failure_preserves_saved_session(tmp_path, monkeypatch):
    accounts = ChatGPTAccounts(tmp_path / "accounts.json")
    seed(accounts)
    before = accounts.path.read_bytes()
    monkeypatch.setattr("infra.chatgpt_auth._metadata", metadata)
    monkeypatch.setattr("infra.chatgpt_auth.requests.post", lambda *a, **kw: SimpleNamespace(status_code=503))
    with pytest.raises(ChatGPTAuthError, match="503"):
        accounts.access_token()
    assert accounts.path.read_bytes() == before


def test_logout_revokes_then_clears_tokens_but_retains_registration(tmp_path, monkeypatch):
    accounts = ChatGPTAccounts(tmp_path / "accounts.json")
    seed(accounts)
    monkeypatch.setattr("infra.chatgpt_auth._metadata", metadata)
    calls = []

    def revoke(*args, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(status_code=200)

    monkeypatch.setattr("infra.chatgpt_auth.requests.post", revoke)
    accounts.logout()
    saved = accounts.read()["profiles"]["oaiapp_test"]
    assert "access_token" not in saved and "refresh_token" not in saved and "id_token" not in saved
    assert saved["subject"] == "user-test"
    assert calls[0]["data"]["token_type_hint"] == "refresh_token"


def test_discovery_rejects_cross_origin_endpoints(monkeypatch):
    data = metadata() | {"token_endpoint": "https://attacker.example/token"}
    monkeypatch.setattr(
        "infra.chatgpt_auth.requests.get",
        lambda *a, **kw: SimpleNamespace(raise_for_status=lambda: None, json=lambda: data),
    )
    with pytest.raises(ChatGPTAuthError, match="endpoint"):
        _metadata()
