import copy
import pytest
from app.mappers import RequestMapper
from app.schemas import UnifiedImageRequest


@pytest.mark.parametrize(
    "style,expected",
    [
        ("vivid", "vivid"),
        ("natural", "natural"),
        (None, None),
    ],
)
def test_openai_style_lift(openai_config, style, expected):
    body = {"model": "gpt-image-1", "prompt": "test prompt"}
    if style:
        body["style"] = style
    unified = RequestMapper.openai_gen_to_unified(body, openai_config, "openai")
    assert unified.style == expected


@pytest.mark.parametrize(
    "body,expected_n",
    [
        ({"model": "gpt-image-1", "prompt": "t"}, 1),
        ({"model": "gpt-image-1", "prompt": "t", "n": 4}, 4),
    ],
)
def test_openai_generation_n_defaults(openai_config, body, expected_n):
    unified = RequestMapper.openai_gen_to_unified(body, openai_config, "openai")
    assert unified.n == expected_n


def test_openai_generation_lift_snapshot(
    openai_generation_request, openai_config, snapshot
):
    unified = RequestMapper.openai_gen_to_unified(
        openai_generation_request, openai_config, "openai"
    )
    assert unified.model_dump() == snapshot


def test_openai_lowering_preserves_semantics(snapshot):
    unified = UnifiedImageRequest(
        target_model="gpt-image-1",
        provider="openai",
        prompt="A lighthouse at night",
        n=1,
        size=None,
        response_format="b64_json",
        style="vivid",
        background="transparent",
        quality="high",
        output_format="png",
        user="tester",
    )
    lowered = RequestMapper.unified_to_openai_payload(unified)

    # Copy to ensure snapshots stay stable even if mapper mutates later
    lowered_copy = copy.deepcopy(lowered)
    assert lowered_copy == snapshot
