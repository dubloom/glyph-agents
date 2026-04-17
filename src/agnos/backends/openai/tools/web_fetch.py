"""Simple WebFetch-like tool for the OpenAI backend."""

from html.parser import HTMLParser
import json
from typing import Annotated
from urllib.error import HTTPError
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import Request
from urllib.request import urlopen

from agents import function_tool

from agnos.approvals import request_tool_approval
from agnos.options import ApprovalHandler


_WEB_FETCH_MAX_BYTES = 2 * 1024 * 1024
_WEB_FETCH_MAX_TEXT_CHARS = 20_000
_WEB_FETCH_USER_AGENT = "agnos-web-fetch/1.0"
_TEXTUAL_MEDIA_TYPES = {
    "application/json",
    "application/ld+json",
    "application/rss+xml",
    "application/xml",
    "application/xhtml+xml",
}


def _charset_from_content_type(content_type: str) -> str | None:
    for segment in content_type.split(";")[1:]:
        key, _, value = segment.partition("=")
        if key.strip().lower() == "charset" and value:
            return value.strip().strip("\"'")
    return None


def _looks_textual(content_type: str) -> bool:
    media_type = content_type.split(";", 1)[0].strip().lower()
    return media_type.startswith("text/") or media_type in _TEXTUAL_MEDIA_TYPES


def _truncate(text: str, max_chars: int) -> tuple[str, bool]:
    if len(text) <= max_chars:
        return text, False
    return text[:max_chars] + "...", True


class _HTMLTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._skip_depth = 0
        self._in_title = False
        self._title_parts: list[str] = []
        self._parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        lowered = tag.lower()
        if lowered in {"script", "style", "noscript"}:
            self._skip_depth += 1
            return
        if lowered == "title":
            self._in_title = True
        elif lowered in {"p", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6"}:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        lowered = tag.lower()
        if lowered in {"script", "style", "noscript"} and self._skip_depth > 0:
            self._skip_depth -= 1
            return
        if lowered == "title":
            self._in_title = False
        elif lowered in {"p", "br", "li", "tr"}:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth > 0:
            return
        text = " ".join(data.split())
        if not text:
            return
        if self._in_title:
            self._title_parts.append(text)
        self._parts.append(text)

    def title(self) -> str | None:
        title = " ".join(self._title_parts).strip()
        return title or None

    def text(self) -> str:
        joined = " ".join(self._parts)
        compact = "\n".join(line.strip() for line in joined.splitlines() if line.strip())
        return compact.strip()


def make_web_fetch_tool(
    *,
    confirm_fetch: bool = False,
    approval_handler: ApprovalHandler | None = None,
):
    @function_tool
    def web_fetch(
        url: Annotated[str, "Absolute URL to fetch (http or https)."],
        max_chars: Annotated[int, "Maximum number of characters to return from the fetched content."] = 6000,
        timeout_seconds: Annotated[int, "HTTP request timeout in seconds."] = 15,
    ) -> str:
        """Fetch a specific URL and return normalized text."""
        parsed = urlparse(url.strip())
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            return "Error: url must be an absolute http(s) URL."

        bounded_max_chars = min(max(max_chars, 1), _WEB_FETCH_MAX_TEXT_CHARS)
        bounded_timeout_seconds = min(max(timeout_seconds, 1), 30)

        if confirm_fetch:
            approved, denied_reason = request_tool_approval(
                handler=approval_handler,
                capability="web",
                tool_name="web_fetch",
                payload={
                    "url": url,
                    "max_chars": bounded_max_chars,
                    "timeout_seconds": bounded_timeout_seconds,
                },
            )
            if not approved:
                return json.dumps(
                    {
                        "ok": False,
                        "requested_url": url,
                        "error": denied_reason or "web_fetch declined.",
                    },
                    ensure_ascii=False,
                )

        request = Request(
            url=url,
            headers={
                "User-Agent": _WEB_FETCH_USER_AGENT,
                "Accept": "text/html,application/json,text/plain,*/*;q=0.8",
            },
        )

        try:
            with urlopen(request, timeout=bounded_timeout_seconds) as response:
                final_url = response.geturl()
                status_code = response.getcode()
                content_type = response.headers.get("Content-Type", "")
                raw = response.read(_WEB_FETCH_MAX_BYTES + 1)
        except HTTPError as exc:
            return json.dumps(
                {
                    "ok": False,
                    "requested_url": url,
                    "final_url": exc.geturl(),
                    "status_code": exc.code,
                    "error": str(exc),
                },
                ensure_ascii=False,
            )
        except URLError as exc:
            return json.dumps(
                {
                    "ok": False,
                    "requested_url": url,
                    "error": str(exc.reason),
                },
                ensure_ascii=False,
            )
        except Exception as exc:
            return json.dumps(
                {
                    "ok": False,
                    "requested_url": url,
                    "error": str(exc),
                },
                ensure_ascii=False,
            )

        if len(raw) > _WEB_FETCH_MAX_BYTES:
            return json.dumps(
                {
                    "ok": False,
                    "requested_url": url,
                    "final_url": final_url,
                    "status_code": status_code,
                    "content_type": content_type,
                    "error": f"Response too large (>{_WEB_FETCH_MAX_BYTES} bytes).",
                },
                ensure_ascii=False,
            )

        if not _looks_textual(content_type):
            return json.dumps(
                {
                    "ok": False,
                    "requested_url": url,
                    "final_url": final_url,
                    "status_code": status_code,
                    "content_type": content_type,
                    "error": "Unsupported non-text content type.",
                },
                ensure_ascii=False,
            )

        charset = _charset_from_content_type(content_type) or "utf-8"
        try:
            decoded = raw.decode(charset, errors="replace")
        except LookupError:
            decoded = raw.decode("utf-8", errors="replace")

        media_type = content_type.split(";", 1)[0].strip().lower()
        title: str | None = None
        text = decoded
        if media_type in {"text/html", "application/xhtml+xml"}:
            parser = _HTMLTextExtractor()
            parser.feed(decoded)
            parser.close()
            title = parser.title()
            extracted = parser.text()
            if extracted:
                text = extracted

        truncated_text, truncated = _truncate(text.strip() or "(empty response body)", bounded_max_chars)
        return json.dumps(
            {
                "ok": True,
                "requested_url": url,
                "final_url": final_url,
                "status_code": status_code,
                "content_type": content_type,
                "title": title,
                "content": truncated_text,
                "truncated": truncated,
            },
            ensure_ascii=False,
        )

    return web_fetch
