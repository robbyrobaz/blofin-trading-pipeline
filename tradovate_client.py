#!/usr/bin/env python3
"""Small Tradovate REST helper focused on auth + market-data websocket bootstrap.

This module intentionally keeps dependencies minimal (stdlib only).
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, Optional


class TradovateError(RuntimeError):
    pass


@dataclass
class TradovateConfig:
    username: str
    password: str
    app_id: str = "OpenClaw"
    app_version: str = "1.0"
    cid: Optional[str] = None
    sec: Optional[str] = None
    auth_url: str = "https://demo.tradovateapi.com/v1/auth/accesstokenrequest"
    api_base_url: str = "https://demo.tradovateapi.com/v1"
    md_ws_url: str = "wss://md-demo.tradovateapi.com/v1/websocket"

    @classmethod
    def from_env(cls, prefix: str = "TRADOVATE_") -> "TradovateConfig":
        username = os.getenv(f"{prefix}USERNAME", "").strip()
        password = os.getenv(f"{prefix}PASSWORD", "").strip()
        if not username or not password:
            raise TradovateError(
                f"Missing {prefix}USERNAME/{prefix}PASSWORD in environment"
            )

        return cls(
            username=username,
            password=password,
            app_id=os.getenv(f"{prefix}APP_ID", "OpenClaw").strip() or "OpenClaw",
            app_version=os.getenv(f"{prefix}APP_VERSION", "1.0").strip() or "1.0",
            cid=(os.getenv(f"{prefix}CID") or "").strip() or None,
            sec=(os.getenv(f"{prefix}SEC") or "").strip() or None,
            auth_url=(
                os.getenv(
                    f"{prefix}AUTH_URL",
                    "https://demo.tradovateapi.com/v1/auth/accesstokenrequest",
                )
                .strip()
                or "https://demo.tradovateapi.com/v1/auth/accesstokenrequest"
            ),
            api_base_url=(
                os.getenv(f"{prefix}API_BASE_URL", "https://demo.tradovateapi.com/v1")
                .strip()
                or "https://demo.tradovateapi.com/v1"
            ),
            md_ws_url=(
                os.getenv(
                    f"{prefix}MD_WS_URL", "wss://md-demo.tradovateapi.com/v1/websocket"
                )
                .strip()
                or "wss://md-demo.tradovateapi.com/v1/websocket"
            ),
        )


class TradovateClient:
    def __init__(self, config: TradovateConfig):
        self.config = config
        self._token: Optional[str] = None
        self._token_expiry_ts = 0.0

    def _request_json(
        self,
        method: str,
        url: str,
        payload: Optional[Dict[str, Any]] = None,
        token: Optional[str] = None,
    ) -> Dict[str, Any]:
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url=url, method=method.upper(), data=body)
        req.add_header("Content-Type", "application/json")
        if token:
            req.add_header("Authorization", f"Bearer {token}")

        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")
            raise TradovateError(f"HTTP {e.code} for {url}: {detail}") from e
        except urllib.error.URLError as e:
            raise TradovateError(f"Network error for {url}: {e}") from e

    def access_token(self, force_refresh: bool = False) -> str:
        if self._token and not force_refresh and time.time() < self._token_expiry_ts:
            return self._token

        payload: Dict[str, Any] = {
            "name": self.config.username,
            "password": self.config.password,
            "appId": self.config.app_id,
            "appVersion": self.config.app_version,
        }
        if self.config.cid:
            payload["cid"] = self.config.cid
        if self.config.sec:
            payload["sec"] = self.config.sec

        data = self._request_json("POST", self.config.auth_url, payload=payload)
        token = data.get("accessToken")
        if not token:
            raise TradovateError(f"No accessToken returned: {data}")

        ttl_s = int(data.get("expirationTime", 3600))
        self._token = token
        self._token_expiry_ts = time.time() + max(60, ttl_s - 30)
        return token

    def contract_find(self, name: str) -> Dict[str, Any]:
        token = self.access_token()
        # Tradovate entity endpoint pattern:
        # GET /contract/find?name=<symbol>, ex: NQH6, NQM6, MNQH6
        url = f"{self.config.api_base_url}/contract/find?name={name}"
        return self._request_json("GET", url, token=token)

    def md_subscribe_quote_command(self, symbol: str) -> Dict[str, Any]:
        return {
            "url": "md/subscribeQuote",
            "body": {"symbol": symbol},
        }

    def md_subscribe_dom_command(self, symbol: str) -> Dict[str, Any]:
        return {
            "url": "md/subscribeDOM",
            "body": {"symbol": symbol},
        }
