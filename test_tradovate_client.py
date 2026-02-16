#!/usr/bin/env python3

import os
import unittest
from unittest.mock import patch

from tradovate_client import TradovateClient, TradovateConfig, TradovateError


class TradovateClientTests(unittest.TestCase):
    def test_from_env_requires_creds(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(TradovateError):
                TradovateConfig.from_env()

    def test_from_env_reads_overrides(self):
        with patch.dict(
            os.environ,
            {
                "TRADOVATE_USERNAME": "alice",
                "TRADOVATE_PASSWORD": "secret",
                "TRADOVATE_AUTH_URL": "https://auth.example",
                "TRADOVATE_API_BASE_URL": "https://api.example",
                "TRADOVATE_MD_WS_URL": "wss://md.example",
            },
            clear=True,
        ):
            cfg = TradovateConfig.from_env()

        self.assertEqual(cfg.username, "alice")
        self.assertEqual(cfg.password, "secret")
        self.assertEqual(cfg.auth_url, "https://auth.example")
        self.assertEqual(cfg.api_base_url, "https://api.example")
        self.assertEqual(cfg.md_ws_url, "wss://md.example")

    def test_auth_payload_and_token_cache(self):
        cfg = TradovateConfig(username="u", password="p")
        c = TradovateClient(cfg)

        calls = []

        def fake_request(method, url, payload=None, token=None):
            calls.append((method, url, payload, token))
            return {"accessToken": "abc", "expirationTime": 300}

        c._request_json = fake_request  # type: ignore[method-assign]

        t1 = c.access_token()
        t2 = c.access_token()

        self.assertEqual(t1, "abc")
        self.assertEqual(t2, "abc")
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0], "POST")
        self.assertIn("name", calls[0][2])

    def test_subscribe_payload_shape(self):
        cfg = TradovateConfig(username="u", password="p")
        c = TradovateClient(cfg)
        quote = c.md_subscribe_quote_command("NQH6")
        dom = c.md_subscribe_dom_command("NQH6")
        self.assertEqual(quote["url"], "md/subscribeQuote")
        self.assertEqual(dom["url"], "md/subscribeDOM")
        self.assertEqual(quote["body"]["symbol"], "NQH6")


if __name__ == "__main__":
    unittest.main()
