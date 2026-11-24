"""Hermes may build a new agent, and so a new engine, for a session whose old agent it only soft-released.

The gateway's cache eviction releases an agent's clients without ending its session, so the old engine
never sees on_session_end. A replacement engine in the same process must still open the session's memory.
"""
import fcntl

import pytest

from optchat.store import Store
from test_identity_recovery import SESSION, SETTINGS, engine_at, message, tool_result, workspace


def test_a_replacement_engine_uses_the_memory_its_soft_released_predecessor_still_holds(workspace, engine_at):
    _, _, conn = workspace
    message(conn, 1, "first", "Remember the code ORCHID-17.")
    retired = engine_at()
    assert retired.problem is None
    replacement = engine_at()  # the retired engine never ended its session
    assert replacement.problem is None, replacement.problem
    assert [m["id"] for m in tool_result(replacement, "optchat_search", {"query": "ORCHID-17"})["matches"]] == [0]
    message(conn, 2, "second", "And the code TULIP-9.")
    replacement.worker.store.sync_hermes(replacement.home, SESSION)
    assert tool_result(replacement, "optchat_zoom", {"id": 1, "n": 1})["text"] == "And the code TULIP-9."


def test_ending_one_engine_leaves_the_memory_open_for_the_engine_still_using_it(workspace, engine_at):
    _, _, conn = workspace
    message(conn, 1, "first", "Remember the code ORCHID-17.")
    first = engine_at()
    second = engine_at()
    first.on_session_end(SESSION, [])
    assert tool_result(second, "optchat_zoom", {"id": 0, "n": 1})["text"] == "Remember the code ORCHID-17."
    message(conn, 2, "second", "And the code TULIP-9.")
    second.worker.store.sync_hermes(second.home, SESSION)
    assert tool_result(second, "optchat_search", {"query": "TULIP-9"})["matches"][0]["id"] == 1


def test_the_memory_is_released_once_every_engine_has_ended(workspace, engine_at):
    root, home, conn = workspace
    message(conn, 1, "first", "Remember the code ORCHID-17.")
    first, second = engine_at(), engine_at()
    lock = first.worker.store.root / "writer.lock"
    first.on_session_end(SESSION, [])
    second.on_session_end(SESSION, [])
    with open(lock, "rb") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)  # raises if an engine still owns the archive


def test_another_process_is_still_refused_while_an_engine_owns_the_memory(workspace, engine_at):
    _, home, conn = workspace
    message(conn, 1, "first", "Remember the code ORCHID-17.")
    owner = engine_at()
    assert owner.problem is None
    # A second Store is what another Hermes process (Desktop's backend) opens; it must not share a writer.
    with pytest.raises(RuntimeError, match="already has a writer"):
        Store(home / "optchat", SESSION, 128000, SETTINGS)
