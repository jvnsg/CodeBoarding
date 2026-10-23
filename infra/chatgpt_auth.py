"""ChatGPT plan OAuth registration and protected, renewable account profiles."""

import base64
import hashlib
import hmac
import json
import os
import secrets
import tempfile
import time
import uuid
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlsplit

import jwt
import requests
from filelock import FileLock

AUTH_ORIGIN = "https://auth.openai.com"
RESOURCE = "https://api.openai.com/v1"
SCOPES = "openid profile email offline_access resource.invoke chatgpt.tokens.use.direct"
AUTH_ENV = "CHATGPT_PLAN_AUTH_FILE"
DEFAULT_AUTH_PATH = Path.home() / ".codeboarding" / "chatgpt" / "accounts.json"
TIMEOUT = 30


class ChatGPTAuthError(RuntimeError):
    """A ChatGPT connection needs user attention."""


def auth_path() -> Path:
    return Path(os.environ.get(AUTH_ENV, str(DEFAULT_AUTH_PATH))).expanduser()


class ChatGPTAccounts:
    def __init__(self, path: Path | None = None):
        self.path = path if path is not None else auth_path()
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.lock = FileLock(str(self.path) + ".lock", timeout=TIMEOUT)

    def read(self) -> dict:
        if not self.path.exists():
            return {"host_id": "", "active": "", "profiles": {}}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(data.get("profiles"), dict):
                raise ValueError("Missing profiles")
            return data
        except (ValueError, TypeError, AttributeError) as exc:
            raise ChatGPTAuthError("Invalid ChatGPT credential file; restore it or choose a new --auth-file.") from exc

    def profile(self) -> dict:
        data = self.read()
        profile = data["profiles"].get(data["active"], {})
        if not profile.get("access_token"):
            raise ChatGPTAuthError("Run `codeboarding auth chatgpt login` first.")
        return profile

    def access_token(self) -> str:
        with self.lock:
            profile = self.profile()
            if "chatgpt.tokens.use.direct" not in profile.get("scopes", []):
                raise ChatGPTAuthError("ChatGPT plan permission is disabled; run `codeboarding auth chatgpt login`.")
            if float(profile["expires_at"]) > time.time() + 60:
                return str(profile["access_token"])
            metadata = _metadata()
            tokens = _token_request(
                metadata["token_endpoint"],
                {
                    "grant_type": "refresh_token",
                    "client_id": profile["client_id"],
                    "refresh_token": profile["refresh_token"],
                    "resource": RESOURCE,
                },
            )
            updated = _credentials(tokens, profile["client_id"], profile)
            data = self.read()
            data["profiles"][profile["client_id"]] = updated
            self._write(data)
            return str(updated["access_token"])

    def login(
        self,
        *,
        new_account: bool = False,
        account: str = "",
        port: int = 0,
        timeout: int = 300,
        open_browser: bool = True,
    ) -> dict:
        metadata = _metadata()
        with self.lock:
            data = self.read()
            if not data.get("host_id"):
                data["host_id"] = "urn:uuid:" + str(uuid.uuid4())
                self._write(data)
            selected = account or data.get("active", "")
            if account and account not in data["profiles"]:
                raise ChatGPTAuthError("Unknown account ID; use `codeboarding auth chatgpt status`.")
            saved = {} if new_account else data["profiles"].get(selected, {})
            host_id = data["host_id"]
        state, nonce, verifier = (secrets.token_urlsafe(32) for _ in range(3))
        callback: dict[str, list[str]] = {}

        class CallbackHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                parsed = urlsplit(self.path)
                values = parse_qs(parsed.query)
                if parsed.path != "/auth/callback":
                    self.send_error(404)
                    return
                if not hmac.compare_digest(values.get("state", [""])[0], state):
                    self.send_error(400, "Invalid OAuth state")
                    return
                callback.update(values)
                body = b"ChatGPT authorization received. You can close this tab and return to CodeBoarding."
                self.send_response(200)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format, *args):
                # The callback query contains a short-lived authorization code.
                return

        with HTTPServer(("127.0.0.1", port), CallbackHandler) as server:
            redirect = f"http://127.0.0.1:{server.server_port}/auth/callback"
            params = {
                "client_id": saved.get("client_id", "dynamic_agent_client"),
                "ext_agent_host_id": host_id,
                "response_type": "code",
                "redirect_uri": redirect,
                "scope": SCOPES,
                "resource": RESOURCE,
                "state": state,
                "nonce": nonce,
                "code_challenge_method": "S256",
                "code_challenge": base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest())
                .rstrip(b"=")
                .decode(),
            }
            if not saved:
                params["agent_name_hint"] = "CodeBoarding"
            # No id_token_hint: keep tokens out of console URLs and allow account selection.
            url = metadata["authorization_endpoint"] + "?" + urlencode(params)
            print("Continue with ChatGPT. Open this URL on this computer:\n" + url, flush=True)
            if open_browser:
                webbrowser.open(url)
            deadline = time.monotonic() + timeout
            while not callback and time.monotonic() < deadline:
                server.timeout = min(1.0, max(0.01, deadline - time.monotonic()))
                server.handle_request()
        if not callback:
            raise ChatGPTAuthError("ChatGPT login timed out; run login again.")
        if "error" in callback:
            raise ChatGPTAuthError("ChatGPT authorization was declined or failed; no account was changed.")
        issued = callback.get("client_id", [saved.get("client_id", "")])[0]
        if not issued or issued == "dynamic_agent_client" or (saved and issued != saved["client_id"]):
            raise ChatGPTAuthError("Invalid or missing issued OAuth client ID.")
        if not callback.get("code"):
            raise ChatGPTAuthError("OAuth callback did not include an authorization code.")
        tokens = _token_request(
            metadata["token_endpoint"],
            {
                "grant_type": "authorization_code",
                "client_id": issued,
                "code": callback["code"][0],
                "code_verifier": verifier,
                "redirect_uri": redirect,
                "resource": RESOURCE,
            },
        )
        try:
            key = jwt.PyJWKClient(metadata["jwks_uri"], timeout=TIMEOUT).get_signing_key_from_jwt(tokens["id_token"])
            identity = jwt.decode(
                tokens["id_token"],
                key.key,
                algorithms=["RS256"],
                audience=issued,
                issuer=AUTH_ORIGIN,
                options={"require": ["exp", "iss", "aud", "sub", "nonce"]},
            )
            if not hmac.compare_digest(identity["nonce"], nonce):
                raise ChatGPTAuthError("Invalid OAuth ID-token nonce.")
            if saved and identity["sub"] != saved["subject"]:
                raise ChatGPTAuthError("Different ChatGPT account selected; use login --new-account.")
        except (jwt.PyJWTError, KeyError, TypeError) as exc:
            raise ChatGPTAuthError("ChatGPT ID-token validation failed.") from exc
        profile = _credentials(tokens, issued, saved)
        profile.update(subject=identity["sub"], email=identity.get("email", ""))
        with self.lock:
            data = self.read()
            data["profiles"][issued] = profile
            data["active"] = issued
            self._write(data)
        return profile

    def logout(self) -> None:
        with self.lock:
            data = self.read()
            profile = data["profiles"].get(data["active"], {})
            if not profile.get("refresh_token"):
                return
            endpoint = _metadata().get("revocation_endpoint")
            if not endpoint:
                raise ChatGPTAuthError("OpenAI did not publish a revocation endpoint; disconnect in ChatGPT settings.")
            response = requests.post(
                endpoint,
                data={
                    "token": profile["refresh_token"],
                    "token_type_hint": "refresh_token",
                    "client_id": profile["client_id"],
                },
                timeout=TIMEOUT,
            )
            if response.status_code != 200:
                raise ChatGPTAuthError("Session revocation failed; credentials retained so you can retry logout.")
            data["profiles"][data["active"]] = {
                k: v for k, v in profile.items() if k not in {"access_token", "refresh_token", "id_token", "expires_at"}
            }
            self._write(data)

    def _write(self, data: dict) -> None:
        fd, name = tempfile.mkstemp(dir=self.path.parent, prefix=".chatgpt-")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(data, stream, indent=2)
            os.replace(name, self.path)
        finally:
            if os.path.exists(name):
                os.unlink(name)


def _metadata() -> dict:
    response = requests.get(AUTH_ORIGIN + "/.well-known/openid-configuration", timeout=TIMEOUT)
    response.raise_for_status()
    data = response.json()
    if data.get("issuer") != AUTH_ORIGIN:
        raise ChatGPTAuthError("Unexpected OpenAI OAuth issuer.")
    for key in ("authorization_endpoint", "token_endpoint", "jwks_uri", "revocation_endpoint"):
        endpoint = data.get(key)
        if endpoint and (urlsplit(endpoint).scheme != "https" or urlsplit(endpoint).netloc != "auth.openai.com"):
            raise ChatGPTAuthError("Unexpected OpenAI OAuth endpoint.")
    return data


def _token_request(endpoint: str, data: dict) -> dict:
    response = requests.post(endpoint, data=data, timeout=TIMEOUT)
    if response.status_code != 200:
        # Never include the request, token response, or authorization code in errors.
        raise ChatGPTAuthError(f"ChatGPT token exchange/refresh failed (HTTP {response.status_code}); retry login.")
    return response.json()


def _credentials(tokens: dict, client_id: str, saved: dict) -> dict:
    result = dict(saved)
    scope = tokens.get("scope")
    scopes = scope.split() if isinstance(scope, str) else saved.get("scopes", [])
    if "chatgpt.tokens.use.direct" not in scopes:
        raise ChatGPTAuthError("ChatGPT plan permission was not granted; run login and enable plan usage.")
    if not tokens.get("access_token") or not tokens.get("refresh_token"):
        raise ChatGPTAuthError("Incomplete ChatGPT token response; repeat login.")
    result.update(
        client_id=client_id,
        access_token=tokens["access_token"],
        refresh_token=tokens["refresh_token"],
        scopes=scopes,
        expires_at=time.time() + float(tokens["expires_in"]),
    )
    if tokens.get("id_token"):
        result["id_token"] = tokens["id_token"]
    return result
