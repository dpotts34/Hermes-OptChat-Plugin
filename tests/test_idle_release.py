"""Two Hermes apps (Desktop's backend, the gateway Hermex talks to) take turns on one chat.

Neither ever ends the session, so the archive writer must be released once its app is idle and reopened
on that app's next use; a turn in progress keeps it.
"""
import fcntl
import time
from dataclasses import replace

import pytest

import optchat
from optchat.store import Store
from test_identity_recovery import SESSION, SETTINGS, engine_at, message, tool_result, workspace

IDLE = replace(SETTINGS, idle_release_seconds=0.2, idle_release_max_seconds=30)


def lock_free(engine):
    with open(engine.home + "/optchat/" + engine_archive(engine) + "/writer.lock", "rb") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        fcntl.flock(handle, fcntl.LOCK_UN)
        return True


def engine_archive(engine):
    import hashlib
    return hashlib.sha256(SESSION.encode()).hexdigest()


def wait_until(condition, seconds=5.0):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.05)
    return False


def other_app(home):
    """What another Hermes process opens: its own Store on the same archive."""
    return Store(home / "optchat", SESSION, 128000, IDLE)


def test_an_idle_app_releases_the_memory_for_another_app(workspace, engine_at):
    _, home, conn = workspace
    message(conn, 1, "first", "Remember the code ORCHID-17.")
    engine = engine_at(IDLE)
    assert engine.problem is None
    assert wait_until(lambda: lock_free(engine)), "the idle app kept the archive"
    other = other_app(home)  # Desktop now opens the chat
    other.close()


def test_the_released_app_reopens_on_its_next_use_and_sees_the_other_apps_messages(workspace, engine_at):
    _, home, conn = workspace
    message(conn, 1, "first", "Remember the code ORCHID-17.")
    engine = engine_at(IDLE)
    assert wait_until(lambda: lock_free(engine))
    other = other_app(home)
    message(conn, 2, "second", "From the other app: TULIP-9.")
    other.sync_hermes(str(home), SESSION)
    other.close()
    hits = tool_result(engine, "optchat_search", {"query": "TULIP-9"})["matches"]
    assert [hit["id"] for hit in hits] == [1]
    assert tool_result(engine, "optchat_zoom", {"id": 0, "n": 1})["text"] == "Remember the code ORCHID-17."


def test_hooks_reopen_released_memory(workspace, engine_at):
    _, home, conn = workspace
    message(conn, 1, "first", "Remember the code ORCHID-17.")
    engine = engine_at(IDLE)
    assert wait_until(lambda: lock_free(engine))
    optchat.pre_tool(session_id=SESSION, tool_call_id="call-1", tool_name="read", args={})
    assert not lock_free(engine)


def test_a_turn_in_progress_keeps_the_memory(workspace, engine_at):
    _, home, conn = workspace
    message(conn, 1, "first", "Remember the code ORCHID-17.")
    engine = engine_at(IDLE)
    engine.begin_turn("turn-1", [{"role": "user", "content": "Remember the code ORCHID-17."}],
                      user_message="Remember the code ORCHID-17.")
    time.sleep(1.0)
    assert not lock_free(engine), "a turn in progress lost its archive"
    engine.on_turn_complete([])
    assert wait_until(lambda: lock_free(engine))


def test_use_while_another_app_holds_the_memory_names_that_app(workspace, engine_at):
    _, home, conn = workspace
    message(conn, 1, "first", "Remember the code ORCHID-17.")
    engine = engine_at(IDLE)
    assert wait_until(lambda: lock_free(engine))
    other = other_app(home)
    try:
        result = tool_result(engine, "optchat_search", {"query": "ORCHID-17"})
        assert "another Hermes app" in result["error"]
    finally:
        other.close()
    assert tool_result(engine, "optchat_search", {"query": "ORCHID-17"})["matches"][0]["id"] == 0


def test_idle_release_can_be_disabled(workspace, engine_at):
    _, home, conn = workspace
    message(conn, 1, "first", "Remember the code ORCHID-17.")
    engine = engine_at(replace(IDLE, idle_release_seconds=0))
    time.sleep(0.8)
    assert not lock_free(engine)


@pytest.mark.parametrize("field,value", [("idle_release_seconds", -1), ("idle_release_max_seconds", 0),
                                         ("idle_release_seconds", float("inf"))])
def test_idle_release_settings_are_validated(field, value):
    with pytest.raises(ValueError):
        replace(SETTINGS, **{field: value})
