"""OpenAI-compatible HTTP upstream forwarding."""

import http.client
from urllib.parse import urlsplit
from typing import Dict, Optional, Tuple


class OpenAICompatibleUpstream:
    """Forwards API requests to a configured OpenAI-compatible service."""

    def __init__(self, base_url: str, api_key: Optional[str] = None, timeout: float = 600.0):
        parsed = urlsplit(base_url.strip())
        if parsed.scheme not in ("http", "https") or not parsed.netloc or not parsed.hostname:
            raise ValueError("upstream URL must be an absolute http:// or https:// URL")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("upstream URL must not contain credentials, a query, or a fragment")
        self.scheme = parsed.scheme
        self.host = parsed.hostname
        self.port = parsed.port
        self.base_path = parsed.path.rstrip("/")
        self.base_url = f"{self.scheme}://{parsed.netloc}{self.base_path}"
        self.api_key = api_key
        self.timeout = timeout

    def _target(self, request_path: str) -> str:
        parsed = urlsplit(request_path)
        path = parsed.path
        if self.base_path.endswith("/v1") and path.startswith("/v1/"):
            path = self.base_path + path[3:]
        else:
            path = self.base_path + path
        return path + (f"?{parsed.query}" if parsed.query else "")

    def open(
        self, method: str, request_path: str, body: bytes, headers: Dict[str, str]
    ) -> Tuple[http.client.HTTPConnection, http.client.HTTPResponse]:
        connection_type = (
            http.client.HTTPSConnection if self.scheme == "https" else http.client.HTTPConnection
        )
        connection = connection_type(self.host, self.port, timeout=self.timeout)
        hop_by_hop = {
            "connection",
            "content-length",
            "expect",
            "host",
            "keep-alive",
            "proxy-authenticate",
            "proxy-authorization",
            "te",
            "trailers",
            "transfer-encoding",
            "upgrade",
        }
        connection_header = next(
            (value for key, value in headers.items() if key.lower() == "connection"),
            "",
        )
        hop_by_hop.update(value.strip().lower() for value in connection_header.split(","))
        forwarded_headers = {
            key: value
            for key, value in headers.items()
            if key.lower() not in hop_by_hop and key.lower() not in {"cookie"}
        }
        if self.api_key:
            for key in tuple(forwarded_headers):
                if key.lower() == "authorization":
                    del forwarded_headers[key]
            forwarded_headers["Authorization"] = f"Bearer {self.api_key}"
        try:
            connection.request(
                method,
                self._target(request_path),
                body=body if method not in ("GET", "HEAD") else None,
                headers=forwarded_headers,
            )
            return connection, connection.getresponse()
        except (http.client.HTTPException, OSError, TimeoutError, ValueError):
            connection.close()
            raise
