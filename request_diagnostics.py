"""Small, credential-free request traces; never log prompts or response bodies."""
import hashlib
import json
import re
import time
import uuid


class RequestTrace:
    def __init__(self, callback, label, model, contents, config):
        self.callback = callback
        self.request_id = uuid.uuid4().hex[:12]
        self.label = label
        self.model = model
        self.attempt = 0
        self.started = None
        self.sent = False
        self.chunks = 0
        self.output_chars = 0
        self.meta = {}
        if callback:
            try:
                data = config.model_dump(mode='json', exclude_none=True) if hasattr(config, 'model_dump') else {}
                schema = data.get('response_json_schema') or data.get('response_schema')
                content = json.dumps(contents, ensure_ascii=False, default=str)
                self.meta = dict(content_chars=len(content), schema_bytes=len(json.dumps(schema)) if schema else 0,
                    max_output_tokens=data.get('max_output_tokens'), thinking_config=data.get('thinking_config'),
                    fingerprint=hashlib.sha256((str(model) + content + json.dumps(data, sort_keys=True, default=str)).encode()).hexdigest()[:16])
            except Exception as error:
                self.meta = {'metadata_unavailable': type(error).__name__}

    def emit(self, event, **fields):
        if not self.callback:
            return
        row = dict(event=event, request_id=self.request_id, label=self.label,
                   model=self.model, attempt=self.attempt, **fields)
        # Diagnostics must not interrupt generation, including disk/log failures.
        try:
            self.callback(row)
        except Exception:
            pass

    def begin(self):
        self.attempt += 1
        self.started = time.monotonic()
        self.sent = False
        self.chunks = 0
        self.output_chars = 0
        self.emit('prepare', **self.meta)

    def send(self):
        self.sent = True
        self.sent_at = time.monotonic()
        self.emit('send', queue_seconds=round(self.sent_at - self.started, 3))

    def chunk(self, text):
        self.chunks += 1
        self.output_chars += len(text)
        if self.chunks == 1:
            self.emit('first_chunk', elapsed_seconds=round(time.monotonic() - self.sent_at, 3))

    def success(self, text):
        self.emit('success', elapsed_seconds=round(time.monotonic() - self.sent_at, 3),
                  chunks=self.chunks, output_chars=len(text))

    def failure(self, error):
        code = getattr(error, 'code', getattr(error, 'status_code', None))
        try:
            code = int(code)
        except (TypeError, ValueError):
            code = None
        kind = {400: 'invalid_request', 401: 'authentication', 403: 'permission',
                429: 'quota', 503: 'provider_unavailable', 504: 'provider_gateway_timeout'}.get(code)
        if not kind:
            name = type(error).__name__.lower()
            kind = ('provider_server_error' if code and code >= 500 else
                    'transport_timeout' if 'timeout' in name else
                    'transport_protocol' if 'protocol' in name else
                    'transport_connection' if 'connect' in name else
                    'local_validation' if isinstance(error, ValueError) else 'transport_or_local_error')
        headers = getattr(getattr(error, 'response', None), 'headers', {}) or {}
        request_id = headers.get('x-request-id') or headers.get('x-goog-request-id')
        # Only opaque request IDs, never error text/URLs/keys or arbitrary headers.
        if request_id and not re.fullmatch(r'[A-Za-z0-9._:/-]{1,180}', str(request_id)):
            request_id = None
        self.emit('failure', category=kind, http_code=code, error_type=type(error).__name__,
            sent=self.sent, elapsed_seconds=round(time.monotonic() - self.started, 3),
            chunks=self.chunks, output_chars=self.output_chars, provider_request_id=request_id)

    def wait(self, seconds, reason):
        self.emit('retry_wait', wait_seconds=seconds, reason=reason)
