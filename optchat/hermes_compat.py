"""Compatibility for Hermes APIs and message forms that differ between core generations and copies.

Runtime content reads use the current public decoder when available. Older
cores expose only SessionDB's private codec; the writer is used only to seed
test rows and delegates to the core rather than duplicating its storage format.

Image turns have two copies of the same ask: the admitted form (persisted and given to
``pre_llm_call``) ends with ``@image:<path>`` lines; the model-facing form carries the images
as parts with ``[Image attached at: <path>]`` hints (native input) or leading vision notes
(text input). ``image_turn_cut`` matches the two through Hermes's own note parser.
"""

import re

_TEXT_TYPES = ("text", "input_text")
_TEXT_IMAGE_NOTE = re.compile(r"\[The user (?:sent|attached) an image")

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


def _unwrap_reference(value):
    # Mirrors Hermes's reference quoting: a value with spaces is wrapped in ` " or '.
    return value[1:-1] if len(value) >= 2 and value[0] == value[-1] and value[0] in "`\"'" else value


def _ask_and_image_refs(text):
    """Split an admitted ask into its typed text and trailing ``@image:`` references."""
    lines, refs = text.split("\n"), []
    while lines and lines[-1].startswith("@image:"):
        refs.insert(0, _unwrap_reference(lines.pop()[len("@image:"):]))
    return "\n".join(lines), refs


def _same_ask(model_text, ask, refs, split_image_notes):
    typed, paths = split_image_notes(model_text)
    return paths == refs and typed.strip() == ask.strip()


def image_turn_cut(live, admitted):
    """Where the admitted image ask begins in Hermes's model-facing copy, or None.

    Returns ``(block, offset)`` like the other cut forms: ``block`` indexes a content list
    (None for a string). Any difference beyond Hermes's image notes (typed text, image
    paths, or image bytes) returns None, so the caller still stops the turn.
    """
    try:
        from agent.image_routing import split_image_notes
    except ImportError:
        return None
    if isinstance(admitted, str) and isinstance(live, str):
        ask, refs = _ask_and_image_refs(admitted)
        if not refs:
            return None
        starts = [m.start() for m in _TEXT_IMAGE_NOTE.finditer(live)
                  if _same_ask(live[m.start():], ask, refs, split_image_notes)]
        return (None, starts[0]) if len(starts) == 1 else None
    if not (isinstance(admitted, list) and isinstance(live, list) and admitted):
        return None
    head, images = admitted[0], admitted[1:]
    if (not isinstance(head, dict) or head.get("type") not in _TEXT_TYPES or not isinstance(head.get("text"), str)
            or not images or any(not isinstance(p, dict) or p.get("type") in _TEXT_TYPES for p in images)):
        return None
    block = len(live) - len(images) - 1
    if block < 0 or live[block + 1:] != images:
        return None
    part = live[block]
    if not isinstance(part, dict) or part.get("type") not in _TEXT_TYPES or not isinstance(part.get("text"), str):
        return None
    ask, refs = _ask_and_image_refs(head["text"])
    if len(refs) != len(images) or not _same_ask(part["text"], ask, refs, split_image_notes):
        return None
    return (block, 0)
