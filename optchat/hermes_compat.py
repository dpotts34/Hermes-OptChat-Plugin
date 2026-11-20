"""Compatibility for Hermes APIs moved between core generations.

Runtime content reads use the current public decoder when available. Older
cores expose only SessionDB's private codec; the writer is used only to seed
test rows and delegates to the core rather than duplicating its storage format.
"""

try:
    from agent.context_engine import ContextEngineError
except ImportError:
    class ContextEngineError(RuntimeError):
        """Fallback base for core versions without the legacy fail-turn exception."""


def decode_content(value):
    try:
        from session_rules.turns import decode_content
    except (ImportError, AttributeError):
        from hermes_state import SessionDB
        decode_content = SessionDB._decode_content
    return decode_content(value)


def encode_content(value):
    """Encode fixture content with the active core's writer; not used at runtime."""
    try:
        from persistence.sqlite.conversation._store.messages import _encode_content
    except (ImportError, AttributeError):
        from hermes_state import SessionDB
        _encode_content = SessionDB._encode_content
    return _encode_content(value)


def tool_call_id_apis():
    try:
        from session_rules.turns import coalesce_tool_call_id, tool_call_id_variants
    except (ImportError, AttributeError):
        from agent.message_sanitization import coalesce_tool_call_id, tool_call_id_variants
    return coalesce_tool_call_id, tool_call_id_variants
