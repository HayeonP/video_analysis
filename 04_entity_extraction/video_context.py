"""Optional, source-backed media context. Text is untrusted data, never identity truth.

Fetching is an explicit preparation command; pipeline loading is entirely offline.
Only title, media type and short source excerpts can enter the inference context.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from html.parser import HTMLParser
import http.client
import ipaddress
import json
from pathlib import Path
import queue
import re
import socket
import ssl
import threading
import time
from urllib.parse import urljoin, urlsplit, urlunsplit

VERSION = 1
MAX_SOURCES = 3
MAX_EXCERPT_CHARS = 600
MAX_DOWNLOAD_BYTES = 512 * 1024
MAX_CONTEXT_BYTES = 32 * 1024
FETCH_TIMEOUT_SEC = 15


def _public_ip(value):
    address = ipaddress.ip_address(value)
    return address.is_global and not address.is_multicast and not address.is_reserved


def _text(value, limit, field):
    if not isinstance(value, str):
        raise ValueError(f'{field} must be a string')
    value = ' '.join(value.split())
    if len(value) > limit:
        raise ValueError(f'{field} exceeds {limit} characters')
    return value


def _url(value):
    value = _text(value, 2048, 'source URL')
    if any(c.isspace() or ord(c) < 32 for c in value):
        raise ValueError('Whitespace is not allowed in source URLs')
    try:
        parts = urlsplit(value)
        host = (parts.hostname or '').rstrip('.').lower()
        port = parts.port
    except ValueError as exc:
        raise ValueError('Invalid source URL') from exc
    if parts.scheme not in ('http', 'https') or not host or parts.username is not None or parts.password is not None:
        raise ValueError('Sources must be public HTTP(S) URLs without credentials')
    if port not in (None, 80, 443):
        raise ValueError('Sources must use a standard HTTP(S) port')
    if host == 'localhost' or host.endswith(('.localhost', '.local', '.internal')) or '%' in host:
        raise ValueError('Private network URLs are not allowed')
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        if '.' not in host:
            raise ValueError('Sources must use a public hostname')
    else:
        if not _public_ip(address):
            raise ValueError('Private network URLs are not allowed')
    return urlunsplit((parts.scheme, parts.netloc, parts.path or '/', parts.query, ''))


def normalize_video_context(data):
    """Bound the context and reject annotations/merge rules rather than importing them."""
    if not isinstance(data, dict):
        raise ValueError('Video context must be a JSON object')
    allowed = {'schema_version', 'title', 'media_type', 'sources', 'usage'}
    if set(data) - allowed:
        raise ValueError('Unsupported video context fields: ' + ', '.join(sorted(set(data) - allowed)))
    if type(data.get('schema_version', VERSION)) is not int or data.get('schema_version', VERSION) != VERSION:
        raise ValueError('Unsupported video context schema version')
    if data.get('usage', 'weak_prior_only') != 'weak_prior_only':
        raise ValueError('Video context can only be a weak prior')
    sources = data.get('sources', [])
    if not isinstance(sources, list) or len(sources) > MAX_SOURCES:
        raise ValueError(f'Video context accepts at most {MAX_SOURCES} sources')
    result = dict(schema_version=VERSION,
                  title=_text(data.get('title', ''), 160, 'title'),
                  media_type=_text(data.get('media_type', ''), 80, 'media_type'),
                  usage='weak_prior_only', sources=[])
    for source in sources:
        fields = {'url', 'final_url', 'title', 'excerpt', 'text_kind', 'retrieved_at', 'content_sha256'}
        if not isinstance(source, dict) or set(source) - fields:
            raise ValueError('Unsupported source fields')
        if not source.get('url') or not source.get('excerpt'):
            raise ValueError('Each context source requires a URL and a short excerpt')
        row = dict(url=_url(source['url']),
                   final_url=_url(source.get('final_url', source['url'])),
                   title=_text(source.get('title', ''), 200, 'source title'),
                   excerpt=_text(source['excerpt'], MAX_EXCERPT_CHARS, 'source excerpt'))
        if not row['excerpt']:
            raise ValueError('Source excerpt cannot be blank')
        if 'text_kind' in source:
            if source['text_kind'] not in ('excerpt', 'paraphrase'):
                raise ValueError('text_kind must be excerpt or paraphrase')
            row['text_kind'] = source['text_kind']
        if 'retrieved_at' in source:
            row['retrieved_at'] = _text(source['retrieved_at'], 40, 'retrieved_at')
            try:
                datetime.fromisoformat(row['retrieved_at'].replace('Z', '+00:00'))
            except ValueError as exc:
                raise ValueError('retrieved_at must be an ISO-8601 timestamp') from exc
        if 'content_sha256' in source:
            digest = source['content_sha256']
            if not isinstance(digest, str) or not re.fullmatch(r'[0-9a-f]{64}', digest):
                raise ValueError('content_sha256 must be a SHA-256 hex digest')
            row['content_sha256'] = digest
        result['sources'].append(row)
    return result if result['title'] or result['media_type'] or result['sources'] else {}


def load_video_context(path: Path | None) -> dict:
    """Load bounded JSON without network access. No supplied file means no context."""
    if path is None:
        return {}
    with Path(path).open('rb') as handle:
        body = handle.read(MAX_CONTEXT_BYTES + 1)
    if len(body) > MAX_CONTEXT_BYTES:
        raise ValueError('Video context file is too large')
    return normalize_video_context(json.loads(body.decode('utf-8')))


class _IntroParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.title = []
        self.descriptions = {}
        self.paragraphs = []
        self._paragraph = None
        self._in_title = False
        self._hidden = []

    def handle_starttag(self, tag, attrs):
        if tag in ('script', 'style', 'noscript', 'svg', 'nav', 'footer', 'header'):
            self._hidden.append(tag)
        if self._hidden:
            return
        attrs = dict(attrs)
        if tag == 'meta':
            key = (attrs.get('name') or attrs.get('property') or '').lower()
            if key in ('description', 'og:description', 'twitter:description') and attrs.get('content'):
                self.descriptions.setdefault(key, attrs['content'])
        if tag == 'title':
            self._in_title = True
        if tag == 'p':
            self._paragraph = []

    def handle_endtag(self, tag):
        if self._hidden:
            if tag == self._hidden[-1]:
                self._hidden.pop()
            return
        if tag == 'title':
            self._in_title = False
        if tag == 'p' and self._paragraph is not None:
            text = ' '.join(''.join(self._paragraph).split())
            if len(text) >= 60:
                self.paragraphs.append(text)
            self._paragraph = None

    def handle_data(self, data):
        if not self._hidden:
            if self._in_title:
                self.title.append(data)
            if self._paragraph is not None:
                self._paragraph.append(data)


def _public_addresses(host, port, timeout):
    """Resolve once within the deadline; pin the validated IP when connecting."""
    answers = queue.Queue(maxsize=1)

    def resolve():
        try:
            answers.put(socket.getaddrinfo(host, port, type=socket.SOCK_STREAM))
        except OSError as exc:
            answers.put(exc)

    threading.Thread(target=resolve, daemon=True).start()
    try:
        addresses = answers.get(timeout=timeout)
    except queue.Empty as exc:
        raise TimeoutError('Source DNS lookup timed out') from exc
    if isinstance(addresses, Exception):
        raise addresses
    if not addresses or any(not _public_ip(row[4][0]) for row in addresses):
        raise ValueError('Source hostname resolves to a private or reserved address')
    return addresses


def _remaining(deadline):
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError('Source download timed out')
    return remaining


def _download(url, timeout):
    deadline = time.monotonic() + timeout
    current = _url(url)
    for _ in range(5):
        parsed = urlsplit(current)
        host = parsed.hostname.rstrip('.').encode('idna').decode('ascii')
        port = parsed.port or (443 if parsed.scheme == 'https' else 80)
        addresses = _public_addresses(host, port, _remaining(deadline))
        family, kind, proto, _, address = addresses[0]
        connection = http.client.HTTPConnection(host, port, timeout=_remaining(deadline))
        sock = socket.socket(family, kind, proto)
        connection.sock = sock
        response = None
        try:
            sock.settimeout(_remaining(deadline))
            sock.connect(address)
            if parsed.scheme == 'https':
                sock.settimeout(_remaining(deadline))
                connection.sock = ssl.create_default_context().wrap_socket(sock, server_hostname=host)
            transport = connection.sock
            transport.settimeout(_remaining(deadline))
            target = urlunsplit(('', '', parsed.path or '/', parsed.query, ''))
            connection.request('GET', target, headers={
                'User-Agent': 'VideoContext/1.0', 'Accept': 'text/html, text/plain',
                'Accept-Encoding': 'identity'})
            response = connection.getresponse()
            if response.status in (301, 302, 303, 307, 308):
                location = response.getheader('Location')
                if not location:
                    raise ValueError('Source redirect has no destination')
                current = _url(urljoin(current, location))
                continue
            if response.status != 200:
                raise ValueError(f'Source returned HTTP {response.status}')
            content_type = response.getheader('Content-Type', '').split(';')[0].strip().lower()
            if content_type not in ('text/html', 'application/xhtml+xml', 'text/plain'):
                raise ValueError('Source must be HTML or plain text')
            if response.getheader('Content-Encoding', 'identity').lower() != 'identity':
                raise ValueError('Compressed source responses are not accepted')
            length = response.getheader('Content-Length')
            if length and int(length) > MAX_DOWNLOAD_BYTES:
                raise ValueError('Source exceeds download limit')
            chunks, size = [], 0
            while True:
                # read1 avoids unbounded socket reads from a slowly streaming peer.
                transport.settimeout(_remaining(deadline))
                chunk = response.read1(min(16384, MAX_DOWNLOAD_BYTES + 1 - size))
                _remaining(deadline)
                if not chunk:
                    break
                chunks.append(chunk)
                size += len(chunk)
                if size > MAX_DOWNLOAD_BYTES:
                    raise ValueError('Source exceeds download limit')
            charset = response.headers.get_content_charset() or 'utf-8'
            return b''.join(chunks), charset, content_type, current
        finally:
            if response is not None:
                response.close()
            connection.close()
    raise ValueError('Source redirected too many times')


def fetch_source(url: str, *, timeout: float = FETCH_TIMEOUT_SEC) -> dict:
    """Fetch one explicitly provided public page, preserving compact provenance."""
    if not 0 < timeout <= 60:
        raise ValueError('Source timeout must be between 0 and 60 seconds')
    original = _url(url)
    body, charset, content_type, final_url = _download(original, timeout)
    try:
        text = body.decode(charset, errors='replace')
    except LookupError:
        text = body.decode('utf-8', errors='replace')
    title = ''
    if content_type != 'text/plain':
        parser = _IntroParser()
        parser.feed(text)
        title = ' '.join(''.join(parser.title).split())[:200]
        text = next((parser.descriptions[key] for key in ('description', 'og:description', 'twitter:description')
                     if key in parser.descriptions), ' '.join(parser.paragraphs[:2]))
    excerpt = ' '.join(text.split())[:MAX_EXCERPT_CHARS]
    if not excerpt:
        raise ValueError('Source has no readable introduction')
    return dict(url=original, final_url=final_url, title=title, excerpt=excerpt,
                retrieved_at=datetime.now(timezone.utc).isoformat(timespec='seconds'),
                content_sha256=hashlib.sha256(body).hexdigest())


def main(argv=None):
    parser = argparse.ArgumentParser(description='Collect optional source-backed video context; source text is a weak prior only.')
    parser.add_argument('--title', required=True)
    parser.add_argument('--media-type', required=True, help='For example: film trailer, documentary, animation')
    parser.add_argument('--url', action='append', required=True, help=f'Public introduction page; repeat up to {MAX_SOURCES} times')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if len(args.url) > MAX_SOURCES:
        parser.error(f'At most {MAX_SOURCES} URLs are supported')
    try:
        context = normalize_video_context(dict(title=args.title, media_type=args.media_type))
        if not context.get('title') or not context.get('media_type'):
            raise ValueError('Title and media type cannot be blank')
        context['sources'] = [fetch_source(url) for url in dict.fromkeys(args.url)]
        context = normalize_video_context(context)
    except (ValueError, OSError, http.client.HTTPException) as exc:
        parser.error(str(exc))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(context, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'Saved {len(context["sources"])} context source(s) to {args.output}')


if __name__ == '__main__':
    main()
