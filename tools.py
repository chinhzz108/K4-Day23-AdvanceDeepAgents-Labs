"""Host-side source tools for the deep research agents.

API calls stay on the host so no credentials need to enter the sandbox. Each
LangChain tool returns JSON text, ``NO RESULTS``, or a sanitized ``ERROR:``.
"""
from __future__ import annotations

import json
import os
import random
import re
import threading
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import quote

import httpx
from langchain_core.tools import tool
from dotenv import load_dotenv

load_dotenv()

ARXIV_URL = "https://export.arxiv.org/api/query"
HF_DAILY_URL = "https://huggingface.co/api/daily_papers"
HF_SEARCH_URL = "https://huggingface.co/api/papers/search"
EXA_URL = "https://mcp.exa.ai/mcp"

_RETRY_STATUSES = {429, 500, 502, 503, 504}
_ARXIV_NAMESPACE = "{http://www.w3.org/2005/Atom}"
_ARXIV_LOCK = threading.Lock()
_ARXIV_LAST_REQUEST = 0.0


class RetryableError(Exception):
    """An upstream error that is safe to retry, optionally with Retry-After seconds."""

    def __init__(self, message: str, retry_after: float | None = None):
        super().__init__(message)
        self.retry_after = retry_after


def with_retry(fn, *, attempts=5, base=1.0, cap=30.0):
    """Retry only RetryableError, using capped exponential backoff and jitter."""
    attempts = int(attempts)
    if attempts < 1:
        raise ValueError("attempts must be at least 1")
    base = max(0.0, float(base))
    cap = max(0.0, float(cap))

    for attempt in range(attempts):
        try:
            return fn()
        except RetryableError as exc:
            if attempt == attempts - 1:
                raise
            if exc.retry_after is not None:
                delay = min(cap, max(0.0, float(exc.retry_after)))
            else:
                exponential = base * (2 ** attempt)
                jitter = random.uniform(0.0, min(base, cap)) if cap else 0.0
                delay = min(cap, exponential + jitter)
            time.sleep(delay)


def _retry_after(response: httpx.Response) -> float | None:
    value = response.headers.get("Retry-After")
    if not value:
        return None
    try:
        return max(0.0, float(value.strip()))
    except ValueError:
        try:
            when = parsedate_to_datetime(value)
            if when.tzinfo is None:
                when = when.replace(tzinfo=timezone.utc)
            return max(0.0, (when - datetime.now(timezone.utc)).total_seconds())
        except (TypeError, ValueError, OverflowError):
            return None


def _check_response(response: httpx.Response) -> httpx.Response:
    if response.status_code in _RETRY_STATUSES:
        raise RetryableError(
            f"HTTP {response.status_code}", retry_after=_retry_after(response)
        )
    response.raise_for_status()
    return response


def _get_json(url: str, *, params: dict, timeout: float = 30.0) -> object:
    def request():
        try:
            response = httpx.get(url, params=params, timeout=timeout)
        except httpx.TransportError as exc:
            raise RetryableError(f"network error: {type(exc).__name__}: {exc}") from exc
        return _check_response(response).json()

    return with_retry(request)


def _clean_text(value) -> str:
    return " ".join(str(value or "").split())


def _clamp_int(value, low: int, high: int) -> int:
    return max(low, min(high, int(value)))


def _error_text(exc: Exception, *, secret: str | None = None) -> str:
    message = f"{type(exc).__name__}: {exc}"
    if secret:
        for candidate in {secret, quote(secret, safe="")}:
            if candidate:
                message = message.replace(candidate, "[REDACTED]")
    return f"ERROR: {message}"


def _arxiv_terms(query: str) -> list[str]:
    # Keep only Unicode letters/digits and internal hyphens. Drop field names and
    # Boolean operators so LLM-generated syntax cannot break the API query.
    terms = re.findall(r"[^\W_]+(?:-[^\W_]+)*", str(query), flags=re.UNICODE)
    return [term for term in terms if term.lower() not in {"all", "and", "or"}]


def _wait_for_arxiv_slot() -> None:
    global _ARXIV_LAST_REQUEST
    with _ARXIV_LOCK:
        elapsed = time.monotonic() - _ARXIV_LAST_REQUEST
        if _ARXIV_LAST_REQUEST and elapsed < 3.0:
            time.sleep(3.0 - elapsed)
        _ARXIV_LAST_REQUEST = time.monotonic()


@tool
def arxiv_search(query: str, max_results: int = 3) -> str:
    """Search arXiv papers by keywords, newest first; returns JSON records with id, URL, date, title, and summary."""
    terms = _arxiv_terms(query)
    if not terms:
        return "NO RESULTS"
    try:
        limit = _clamp_int(max_results, 1, 30)
        search_query = " AND ".join(f"all:{term}" for term in terms)

        def request():
            _wait_for_arxiv_slot()
            try:
                response = httpx.get(
                    ARXIV_URL,
                    params={
                        "search_query": search_query,
                        "sortBy": "submittedDate",
                        "sortOrder": "descending",
                        "max_results": limit,
                    },
                    timeout=40.0,
                )
            except httpx.TransportError as exc:
                raise RetryableError(
                    f"network error: {type(exc).__name__}: {exc}"
                ) from exc
            return _check_response(response).text

        xml_text = with_retry(request, attempts=6, base=2.0, cap=60.0)
        root = ET.fromstring(xml_text)
        records = []
        for entry in root.findall(f"{_ARXIV_NAMESPACE}entry"):
            entry_id = _clean_text(entry.findtext(f"{_ARXIV_NAMESPACE}id"))
            raw_id = entry_id.rsplit("/abs/", 1)[-1].rsplit("/", 1)[-1]
            paper_id = re.sub(r"v\d+$", "", raw_id)
            if not paper_id:
                continue
            records.append(
                {
                    "id": paper_id,
                    "url": f"https://arxiv.org/abs/{paper_id}",
                    "published": _clean_text(
                        entry.findtext(f"{_ARXIV_NAMESPACE}published")
                    )[:10],
                    "title": _clean_text(entry.findtext(f"{_ARXIV_NAMESPACE}title")),
                    "summary": _clean_text(
                        entry.findtext(f"{_ARXIV_NAMESPACE}summary")
                    )[:600],
                }
            )
        return json.dumps(records, ensure_ascii=False) if records else "NO RESULTS"
    except Exception as exc:  # tools must report errors to the agent, not raise
        return _error_text(exc)


def _hf_items(payload) -> list:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        # Be tolerant of API envelopes while keeping malformed responses explicit.
        for key in ("papers", "results", "data"):
            value = payload.get(key)
            if isinstance(value, list):
                return value
    return []


def _hf_record(item, *, prefer_ai_summary=False) -> dict | None:
    if not isinstance(item, dict):
        return None
    paper = item.get("paper") if isinstance(item.get("paper"), dict) else item
    paper_id = _clean_text(paper.get("id"))
    if not paper_id:
        return None
    title = _clean_text(paper.get("title") or item.get("title"))
    summary_candidates = []
    if prefer_ai_summary:
        summary_candidates.extend([paper.get("ai_summary"), item.get("ai_summary")])
    summary_candidates.extend(
        [paper.get("summary"), item.get("summary"), item.get("ai_summary")]
    )
    summary = next((_clean_text(value) for value in summary_candidates if value), "")
    try:
        upvotes = int(paper.get("upvotes") or item.get("upvotes") or 0)
    except (TypeError, ValueError):
        upvotes = 0
    try:
        stars = int(paper.get("githubStars") or item.get("githubStars") or 0)
    except (TypeError, ValueError):
        stars = 0
    return {
        "id": paper_id,
        "url": f"https://huggingface.co/papers/{paper_id}",
        "published": _clean_text(
            paper.get("publishedAt") or item.get("publishedAt") or ""
        )[:10],
        "title": title,
        "summary": summary[:600],
        "upvotes": upvotes,
        "github": paper.get("githubRepo") or item.get("githubRepo") or "",
        "stars": stars,
    }


@tool
def hf_daily_papers(limit: int = 5, date: str = "", keyword: str = "") -> str:
    """Return trending Hugging Face Daily Papers, optionally filtered by date and title/summary keyword."""
    try:
        params = {"limit": _clamp_int(limit, 1, 100)}
        if date:
            if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(date).strip()):
                raise ValueError("date must use YYYY-MM-DD")
            params["date"] = str(date).strip()
        payload = _get_json(HF_DAILY_URL, params=params)
        records = [record for item in _hf_items(payload) if (record := _hf_record(item))]
        if keyword:
            needle = _clean_text(keyword).casefold()
            records = [
                record
                for record in records
                if needle in f"{record['title']} {record['summary']}".casefold()
            ]
        records.sort(key=lambda record: record["upvotes"], reverse=True)
        return json.dumps(records, ensure_ascii=False) if records else "NO RESULTS"
    except Exception as exc:
        return _error_text(exc)


@tool
def hf_search_papers(query: str, limit: int = 3) -> str:
    """Search Hugging Face Papers by topic; returns paper metadata, summaries, votes, and repository links."""
    if not _clean_text(query):
        return "NO RESULTS"
    try:
        payload = _get_json(
            HF_SEARCH_URL,
            params={"q": _clean_text(query), "limit": _clamp_int(limit, 1, 50)},
        )
        records = [
            record
            for item in _hf_items(payload)
            if (record := _hf_record(item, prefer_ai_summary=True))
        ]
        return json.dumps(records, ensure_ascii=False) if records else "NO RESULTS"
    except Exception as exc:
        return _error_text(exc)


def _rate_limited(payload) -> bool:
    """Detect Exa's HTTP-200 rate-limit signal in content or MCP metadata."""
    def message_is_limited(value):
        normalized = re.sub(r"[^a-z0-9]+", " ", str(value).casefold())
        return bool(
            re.search(r"\btoo many requests\b", normalized)
            or re.search(r"\bhttp 429\b", normalized)
            or re.search(r"\bquota\b.{0,50}\b(exceed|reach|exhaust)\w*", normalized)
            or re.search(r"\brate limited\b", normalized)
            or re.search(
                r"\brate limit(?:ed)?\b.{0,60}\b(exceed|reach|hit|retry|wait|temporar)\w*",
                normalized,
            )
            or re.search(
                r"\b(exceed|reach|hit|retry|wait|temporar)\w*\b.{0,60}\brate limit(?:ed)?\b",
                normalized,
            )
        )

    def inspect(value):
        if isinstance(value, dict):
            for key, child in value.items():
                compact_key = re.sub(r"[^a-z0-9]", "", str(key).casefold())
                if "ratelimit" in compact_key and child not in (None, False, 0, "", "false"):
                    return True
                if inspect(child):
                    return True
            return False
        if isinstance(value, (list, tuple)):
            return any(inspect(child) for child in value)
        return isinstance(value, str) and message_is_limited(value)

    return inspect(payload)


def _mcp_payload(response: httpx.Response):
    text = response.text
    data_lines = [
        line[5:].strip()
        for line in text.splitlines()
        if line.startswith("data:") and line[5:].strip() not in {"[DONE]", ""}
    ]
    if data_lines:
        parsed = [json.loads(line) for line in data_lines]
        # A single request normally has one JSON-RPC response. If the server emits
        # progress events too, use the last response carrying result/error.
        for payload in reversed(parsed):
            if isinstance(payload, dict) and ("result" in payload or "error" in payload):
                return payload
        return parsed[-1]
    return response.json()


def _exa_call(name: str, arguments: dict):
    secret = (os.getenv("EXA_API_KEY") or "").strip()
    endpoint_params = {"exaApiKey": secret} if secret else None
    request_body = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {"name": name, "arguments": arguments},
    }

    def request():
        try:
            response = httpx.post(
                EXA_URL,
                params=endpoint_params,
                json=request_body,
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json, text/event-stream",
                },
                timeout=45.0,
            )
        except httpx.TransportError as exc:
            raise RetryableError(
                _error_text(exc, secret=secret).removeprefix("ERROR: ")
            ) from exc
        _check_response(response)
        try:
            payload = _mcp_payload(response)
        except (ValueError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"invalid Exa MCP response: {exc}") from exc
        if _rate_limited(payload):
            raise RetryableError("Exa rate limit signaled in HTTP 200 response")
        if isinstance(payload, dict) and payload.get("error"):
            raise RuntimeError(f"Exa MCP error: {payload['error']}")
        result = payload.get("result", payload) if isinstance(payload, dict) else payload
        if isinstance(result, dict) and result.get("isError"):
            raise RuntimeError("Exa MCP tool returned an error")
        return result

    try:
        return with_retry(request, attempts=6, base=2.0, cap=60.0), secret
    except Exception as exc:
        raise RuntimeError(_error_text(exc, secret=secret).removeprefix("ERROR: ")) from exc


def _exa_text(result) -> str:
    if isinstance(result, str):
        return result.strip()
    if isinstance(result, dict):
        content = result.get("content")
        if isinstance(content, list):
            chunks = [
                str(part.get("text", ""))
                for part in content
                if isinstance(part, dict) and part.get("type") == "text"
            ]
            text = "\n".join(chunk.strip() for chunk in chunks if chunk.strip())
            if text:
                return text
        structured = result.get("structuredContent")
        if structured is not None:
            return json.dumps(structured, ensure_ascii=False)
    return ""


def _exa_error(exc: Exception) -> str:
    secret = (os.getenv("EXA_API_KEY") or "").strip()
    return _error_text(exc, secret=secret)


@tool
def web_search(query: str, objective: str = "", num_results: int = 2) -> str:
    """Search the web through Exa and return result text with URLs; describe the desired sources in natural language."""
    query = _clean_text(query)
    if not query:
        return "NO RESULTS"
    objective = _clean_text(objective) or f"Find authoritative research sources about {query}."
    try:
        result, _secret = _exa_call(
            "web_search_exa",
            {
                "query": query,
                "objective": objective,
                "numResults": _clamp_int(num_results, 1, 10),
            },
        )
        text = _exa_text(result)
        if _rate_limited(text):
            # Defensive fallback for servers that put the limit notice only in text.
            raise RuntimeError("Exa returned a rate-limit message after retries")
        if not text or re.search(r"\bno results? found\b", text, re.I):
            return "NO RESULTS"
        return text
    except Exception as exc:
        return _exa_error(exc)


@tool
def web_fetch(url: str) -> str:
    """Fetch one HTTP(S) page through Exa as readable text; long results are truncated to about 6,000 characters."""
    url = str(url or "").strip()
    if not re.match(r"^https?://", url, flags=re.I):
        return "NO RESULTS" if not url else "ERROR: ValueError: url must start with http:// or https://"
    try:
        result, _secret = _exa_call("web_fetch_exa", {"urls": [url]})
        text = _exa_text(result)
        if _rate_limited(text):
            raise RuntimeError("Exa returned a rate-limit message after retries")
        if not text or re.search(r"\bno results? found\b", text, re.I):
            return "NO RESULTS"
        return text[:6000]
    except Exception as exc:
        return _exa_error(exc)


SOURCE_TOOLS = [arxiv_search, hf_daily_papers, hf_search_papers, web_search, web_fetch]


if __name__ == "__main__":
    for name, fn, args in [
        ("arxiv_search", arxiv_search, {"query": "world model", "max_results": 3}),
        ("hf_daily_papers", hf_daily_papers, {"limit": 20}),
        ("hf_search_papers", hf_search_papers, {"query": "world model", "limit": 3}),
        ("web_search", web_search, {"query": "survey paper on world models", "num_results": 2}),
        ("web_fetch", web_fetch, {"url": "https://arxiv.org/abs/1803.10122"}),
    ]:
        print(f"== {name}\n{fn.invoke(args)[:400]}\n")
