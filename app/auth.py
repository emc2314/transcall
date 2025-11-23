import json
import logging
from typing import Optional, Dict, Any
from google.oauth2 import service_account
from google.auth.transport.requests import Request

logger = logging.getLogger("Auth")


class GoogleAuthManager:
    _SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]
    _creds_cache: Dict[str, service_account.Credentials] = {}

    @classmethod
    def get_access_token(cls, credentials_json: str) -> str:
        """
        Parses the credentials JSON (service account), creates or retrieves a
        Credentials object, refreshes it if necessary, and returns a valid access token.
        """
        if not credentials_json:
            raise ValueError("Credentials JSON content is empty")

        # Use a simple hash or just the content itself as a cache key might be too large.
        # For simplicity in this context (likely one SA per deployment), we can just parse it.
        # To be safe and efficient, let's just re-instantiate.
        # Google Auth library handles caching of the token inside the Credentials object efficiently
        # if we reuse the object.

        # We'll cache based on the first 64 chars (usually contains project_id/client_email)
        # or just parse it to get the client_email as key.
        try:
            info = json.loads(credentials_json)
            client_email = info.get("client_email")
            if not client_email:
                raise ValueError("Invalid Service Account JSON: missing 'client_email'")
        except json.JSONDecodeError:
            raise ValueError("Invalid Service Account JSON: Not valid JSON")

        creds = cls._creds_cache.get(client_email)
        if not creds:
            creds = service_account.Credentials.from_service_account_info(
                info, scopes=cls._SCOPES
            )
            cls._creds_cache[client_email] = creds

        if not creds.valid:
            logger.debug(f"Refreshing access token for {client_email}")
            creds.refresh(Request())

        return creds.token
