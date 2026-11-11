"""Bounded recovery for provider-side JSON-schema rejection."""


def generate_with_schema_recovery(generate, config, report=None):
    """Retry a rejected schema once in JSON mode; keep all local validators.

    A bare 400 cannot establish which field is invalid. Removing only the schema
    once isolates that possibility; failure of the second request propagates.
    Auth, quota, capacity and transport failures retain their own retry policy.
    """
    try:
        return generate(config)
    except Exception as error:
        code = getattr(error, 'code', getattr(error, 'status_code', None))
        message = str(error).lower()
        schema = getattr(config, 'response_json_schema', None) or getattr(config, 'response_schema', None)
        if str(code) != '400' or not schema or not any(
                term in message for term in ('invalid_argument', 'invalid argument', 'schema')):
            raise
        if report:
            report('Gemini tu choi request co schema (400); thu lai mot lan o JSON mode, '
                   'van kiem tra ID, so cau va bang chung tai backend.')
        fallback = config.model_copy(update={'response_schema': None,
            'response_json_schema': None, 'response_mime_type': 'application/json'})
        return generate(fallback)
