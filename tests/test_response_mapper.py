from app.mappers import ResponseMapper


def test_openai_response_lift_snapshot(openai_generation_response, snapshot):
    unified = ResponseMapper.openai_to_unified(openai_generation_response)
    assert unified.model_dump() == snapshot


def test_openai_roundtrip_preserves_core_fields(openai_generation_response):
    unified = ResponseMapper.openai_to_unified(openai_generation_response)
    lowered = ResponseMapper.unified_to_openai_format(unified)

    assert lowered["output_format"] == openai_generation_response["output_format"]
    assert (
        lowered["data"][0]["b64_json"]
        == openai_generation_response["data"][0]["b64_json"]
    )
    assert lowered["data"][0]["revised_prompt"].startswith("A red panda")


def test_gemini_response_lift_snapshot(
    gemini_generate_content_response, snapshot
):
    unified = ResponseMapper.gemini_to_unified(gemini_generate_content_response)
    assert unified.model_dump() == snapshot