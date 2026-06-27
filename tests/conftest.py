"""Shared offline test doubles.

These let the whole suite run with no network and no ML deps. They are exposed
as fixtures that return the *class*, so each test can queue its own responses.
"""

from __future__ import annotations

import pytest


class FakeResponse:
    def __init__(self, json_data, status_code: int = 200, headers: dict | None = None):
        self._json = json_data
        self.status_code = status_code
        self.headers = headers or {}

    def json(self):
        return self._json

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class FakeSession:
    """Pops queued FakeResponses in order; records calls."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.headers: dict = {}
        self.calls: list = []

    def get(self, url, params=None, timeout=None):
        self.calls.append(("GET", url, params))
        return self._responses.pop(0)

    def post(self, url, params=None, json=None, timeout=None):
        self.calls.append(("POST", url, json))
        return self._responses.pop(0)


class FakeClient:
    """Stands in for WikipediaClient (get_json / get_external_json)."""

    def __init__(self, json_responses=None, external_responses=None):
        self._json = list(json_responses or [])
        self._external = list(external_responses or [])
        self.json_calls: list = []
        self.external_calls: list = []

    def get_json(self, params, use_cache=True):
        self.json_calls.append(params)
        return self._json.pop(0)

    def get_external_json(self, url, params=None, method="GET", json_body=None, use_cache=True):
        self.external_calls.append((url, method, json_body))
        return self._external.pop(0)


@pytest.fixture
def response_cls():
    return FakeResponse


@pytest.fixture
def session_cls():
    return FakeSession


@pytest.fixture
def client_cls():
    return FakeClient
