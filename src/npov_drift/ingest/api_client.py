"""A polite, cached MediaWiki / Wikimedia HTTP client.

Politeness features (per Wikimedia etiquette and the build spec):
  * descriptive User-Agent with contact info,
  * ``maxlag`` parameter so we voluntarily back off under server load,
  * exponential backoff honouring ``Retry-After`` on 429/503,
  * a minimum interval between live requests,
  * on-disk caching of every response.

The HTTP session is injectable so unit tests can run fully offline.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Optional

from .. import config
from .cache import JsonCache, canonical_key


class ApiError(RuntimeError):
    """Raised when the API returns an error we cannot recover from."""


class WikipediaClient:
    def __init__(
        self,
        endpoint: str = config.API_ENDPOINT,
        user_agent: str = config.USER_AGENT,
        cache: Optional[JsonCache] = None,
        session: Any = None,
        min_interval: float = config.DEFAULT_MIN_INTERVAL,
        max_retries: int = config.DEFAULT_MAX_RETRIES,
        maxlag: int = config.DEFAULT_MAXLAG,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.endpoint = endpoint
        self.cache = cache
        self.min_interval = min_interval
        self.max_retries = max_retries
        self.maxlag = maxlag
        self._sleep = sleep
        self._clock = clock
        self._last_request_at: Optional[float] = None

        if session is None:
            # Imported lazily so importing this module never requires requests
            # to be installed in environments that only run offline tests with
            # an injected session.
            import requests

            session = requests.Session()
        self.session = session
        # Set UA on the session if it exposes a headers mapping.
        headers = getattr(self.session, "headers", None)
        if headers is not None:
            headers["User-Agent"] = user_agent

    # -- public API --------------------------------------------------------
    def get_json(self, params: dict[str, Any], use_cache: bool = True) -> dict:
        """GET the MediaWiki Action API with the given params (cached)."""
        full = {
            **params,
            "format": "json",
            "formatversion": 2,
            "maxlag": self.maxlag,
        }
        key = canonical_key({"url": self.endpoint, "params": full})
        if use_cache and self.cache is not None and self.cache.has(key):
            return self.cache.get(key)
        data = self._get_with_retries(self.endpoint, full)
        if use_cache and self.cache is not None:
            self.cache.set(key, data)
        return data

    def get_external_json(
        self,
        url: str,
        params: Optional[dict[str, Any]] = None,
        method: str = "GET",
        json_body: Optional[dict] = None,
        use_cache: bool = True,
    ) -> dict:
        """GET/POST an arbitrary Wikimedia JSON endpoint (e.g. ORES / Lift Wing)."""
        key = canonical_key(
            {"url": url, "params": params or {}, "method": method, "body": json_body or {}}
        )
        if use_cache and self.cache is not None and self.cache.has(key):
            return self.cache.get(key)
        data = self._get_with_retries(url, params, method=method, json_body=json_body)
        if use_cache and self.cache is not None:
            self.cache.set(key, data)
        return data

    # -- internals ---------------------------------------------------------
    def _respect_interval(self) -> None:
        if self._last_request_at is None:
            self._last_request_at = self._clock()
            return
        elapsed = self._clock() - self._last_request_at
        if elapsed < self.min_interval:
            self._sleep(self.min_interval - elapsed)
        self._last_request_at = self._clock()

    def _get_with_retries(
        self,
        url: str,
        params: Optional[dict[str, Any]],
        method: str = "GET",
        json_body: Optional[dict] = None,
    ) -> dict:
        delay = 1.0
        for _ in range(self.max_retries):
            self._respect_interval()
            if method == "POST":
                resp = self.session.post(url, params=params, json=json_body, timeout=30)
            else:
                resp = self.session.get(url, params=params, timeout=30)

            status = resp.status_code
            if status in (429, 502, 503, 504):
                retry_after = resp.headers.get("Retry-After")
                wait = float(retry_after) if retry_after else delay
                self._sleep(wait)
                delay = min(delay * 2, 60.0)
                continue

            resp.raise_for_status()
            data = resp.json()

            # maxlag errors arrive as HTTP 200 with an "error" block.
            if isinstance(data, dict) and data.get("error", {}).get("code") == "maxlag":
                self._sleep(delay)
                delay = min(delay * 2, 60.0)
                continue

            return data

        raise ApiError(f"Exceeded {self.max_retries} retries for {url}")
