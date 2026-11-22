"""Image turns: Hermes's model copy of the ask differs from its admitted copy only in image notes."""
import copy

import pytest

from optchat.hermes_compat import ContextEngineError
from test_identity_recovery import PNG_BYTES, engine_at, message, workspace

image_routing = pytest.importorskip("agent.image_routing")

PROMPT = "make another prototype that feels like this image"
ROW = 2


def image_at(tmp_path, name="upload.png", data=PNG_BYTES):
    path = tmp_path / name
    path.write_bytes(data)
    return str(path)


def admitted_refs(paths):
    # Hermes persists the ask (and hands pre_llm_call) caption first, then one directive per image.
    from agent.context_references import format_reference_value
    return "\n".join(f"@image:{format_reference_value(p)}" for p in paths)


def run_turn(engine, database, admitted, live, durable):
    message(database, 1, "earlier", "Earlier context the view must still carry")
    message(database, ROW, "image-ask", durable)
    current = {"role": "user", "content": copy.deepcopy(live), "_row_id": ROW}
    engine.begin_turn("image-turn", [current], user_message=copy.deepcopy(admitted))
    return engine.select_context([{"role": "system", "content": "host"}, copy.deepcopy(current)],
                                 conversation_messages=[current], incoming_message=current)


def native_turn(paths, prompt=PROMPT):
    live, skipped = image_routing.build_native_content_parts(prompt, paths)
    assert not skipped and len(live) == len(paths) + 1
    refs = admitted_refs(paths)
    admitted = [{"type": "text", "text": f"{prompt}\n{refs}" if prompt else refs}, *copy.deepcopy(live[1:])]
    return admitted, live


@pytest.mark.parametrize("prompt,names", [(PROMPT, ["upload.png"]), ("", ["upload.png"]),
                                          (PROMPT, ["first.png", "second shot.png"])])
def test_native_image_turn_keeps_images_and_hints_in_the_request(workspace, engine_at, tmp_path, prompt, names):
    _, _, database = workspace
    paths = [image_at(tmp_path, name) for name in names]
    admitted, live = native_turn(paths, prompt)
    selected = run_turn(engine_at(), database, admitted, live, prompt + "\n[screenshot]" * len(paths))
    assert selected[-1]["content"] == live
    assert "Earlier context the view must still carry" in str(selected)


def test_native_image_turn_after_a_wire_only_note(workspace, engine_at, tmp_path):
    _, _, database = workspace
    admitted, live = native_turn([image_at(tmp_path)])
    noted = [{"type": "text", "text": "[Desktop surface note]"}, *live]
    selected = run_turn(engine_at(), database, admitted, noted, PROMPT + "\n[screenshot]")
    assert selected[-1]["content"] == live


def test_text_mode_image_turn_keeps_the_vision_note(workspace, engine_at, tmp_path):
    _, _, database = workspace
    path = image_at(tmp_path)
    note = (f"[The user attached an image: upload.png]\n"
            f"[Examine it with the vision_analyze tool using image_url: {path}]")
    live = f"{note}\n\n{PROMPT}"
    selected = run_turn(engine_at(), database, f"{PROMPT}\n@image:{path}", live, f"{PROMPT}\n@image:{path}")
    assert selected[-1]["content"] == live


@pytest.mark.parametrize("change", ["typed text", "image bytes", "image path"])
def test_a_real_change_to_an_image_ask_still_stops_the_turn(workspace, engine_at, tmp_path, change):
    _, _, database = workspace
    path = image_at(tmp_path)
    admitted, live = native_turn([path])
    if change == "typed text":
        live, _ = image_routing.build_native_content_parts(PROMPT + " and also delete the old one", [path])
    elif change == "image bytes":
        live, _ = image_routing.build_native_content_parts(PROMPT, [image_at(tmp_path, "other.png", PNG_BYTES + b"x")])
        live[0]["text"] = live[0]["text"].replace("other.png", "upload.png")
    else:
        admitted[0]["text"] = admitted[0]["text"].replace("upload.png", "elsewhere.png")
    with pytest.raises(ContextEngineError, match="changed the admitted user content"):
        run_turn(engine_at(), database, admitted, live, PROMPT + "\n[screenshot]")
