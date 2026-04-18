from __future__ import annotations

import ipaddress
import json
import os
import socket
from urllib.parse import urlparse

import httpx
import trafilatura

OUTPUT_CAP = 8 * 1024
FETCH_TIMEOUT = 15
FETCH_MAX_SIZE = 2 * 1024 * 1024
DEFAULT_MAX_RESULTS = 5
USER_AGENT = "LeoBot/0.1 (+https://github.com/yuanlin2004/leo)"


def _truncate(text: str) -> str:
    if len(text) <= OUTPUT_CAP:
        return text
    omitted = len(text) - OUTPUT_CAP
    return text[:OUTPUT_CAP] + f"\n...(truncated, {omitted} bytes omitted)"


def _net_check(ctx) -> str | None:
    if not ctx.net_on:
        return "error: network is disabled (use /net-on to enable)"
    return None


def _url_safe(url: str) -> str | None:
    try:
        parsed = urlparse(url)
    except Exception as e:
        return f"error: invalid URL: {e}"
    if parsed.scheme not in ("http", "https"):
        return f"error: unsupported URL scheme {parsed.scheme!r} (only http/https allowed)"
    host = parsed.hostname
    if not host:
        return "error: URL has no hostname"
    try:
        addrs = socket.getaddrinfo(host, None)
    except socket.gaierror as e:
        return f"error: DNS resolution failed: {e}"
    for info in addrs:
        ip = ipaddress.ip_address(info[4][0])
        if ip.is_loopback or ip.is_private or ip.is_link_local or ip.is_reserved:
            return f"error: refusing to connect to non-public address {ip}"
    return None


def web_fetch(ctx, url: str) -> str:
    if err := _net_check(ctx):
        return err
    if err := _url_safe(url):
        return err
    try:
        with httpx.Client(
            timeout=FETCH_TIMEOUT,
            follow_redirects=True,
            headers={"User-Agent": USER_AGENT},
        ) as client:
            resp = client.get(url)
    except httpx.HTTPError as e:
        return f"error: fetch failed: {type(e).__name__}: {e}"
    if resp.status_code >= 400:
        return f"error: HTTP {resp.status_code}"
    ctype = resp.headers.get("content-type", "").lower()
    if "html" not in ctype and "text" not in ctype:
        return f"error: unsupported content-type {ctype!r} (only html/text handled)"
    if len(resp.content) > FETCH_MAX_SIZE:
        return f"error: response too large ({len(resp.content)} bytes, limit {FETCH_MAX_SIZE})"
    html = resp.text
    extracted = trafilatura.extract(
        html,
        include_links=True,
        include_formatting=True,
        output_format="markdown",
    ) or ""
    metadata = trafilatura.extract_metadata(html)
    title = (metadata.title if metadata else "") or ""
    return json.dumps(
        {"url": str(resp.url), "title": title, "content": _truncate(extracted)},
        indent=2,
        ensure_ascii=False,
    )


def web_search(ctx, query: str, max_results: int = DEFAULT_MAX_RESULTS) -> str:
    if err := _net_check(ctx):
        return err
    key = os.environ.get("TAVILY_API_KEY")
    if not key:
        return "error: TAVILY_API_KEY not set"
    try:
        from tavily import TavilyClient
    except ImportError:
        return "error: tavily-python is not installed"
    try:
        client = TavilyClient(api_key=key)
        resp = client.search(query=query, max_results=max_results, search_depth="basic")
    except Exception as e:
        return f"error: search failed: {type(e).__name__}: {e}"
    results = [
        {
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "snippet": r.get("content", ""),
        }
        for r in resp.get("results", [])
    ]
    return json.dumps(results, indent=2, ensure_ascii=False)


SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the web via Tavily. Returns a JSON list of results, each with "
                "title, url, and snippet (a content-extracted excerpt, not just the "
                "page's meta description). Requires the TAVILY_API_KEY environment "
                "variable. Disabled when the network toggle is off (/net-off)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query."},
                    "max_results": {
                        "type": "integer",
                        "description": f"Max results to return (default {DEFAULT_MAX_RESULTS}).",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_fetch",
            "description": (
                "Fetch a URL and return the main article content extracted as "
                "markdown, along with the final URL and page title. Only http(s) "
                "schemes allowed; refuses non-public addresses (localhost, private "
                "IPs, link-local). HTML/text content-types only. Content truncated "
                "to 8 KB. Disabled when the network toggle is off."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "http(s) URL to fetch."},
                },
                "required": ["url"],
            },
        },
    },
]

FUNCTIONS = {
    "web_search": web_search,
    "web_fetch": web_fetch,
}
