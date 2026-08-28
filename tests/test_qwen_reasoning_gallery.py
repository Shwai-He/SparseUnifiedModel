import pytest

from qwen_geneval_gallery import parse_reasoning_output


def test_parse_reasoning_output() -> None:
    reasoning, prompt = parse_reasoning_output(
        "<reasoning>It is a newly emerged dragonfly.</reasoning>"
        "<generation_prompt>A dragonfly beside its split nymph shell.</generation_prompt>"
    )
    assert reasoning == "It is a newly emerged dragonfly."
    assert prompt == "A dragonfly beside its split nymph shell."


@pytest.mark.parametrize(
    "value",
    [
        "<reasoning>Only reasoning.</reasoning>",
        "<generation_prompt>Only a prompt.</generation_prompt>",
        "<reasoning></reasoning><generation_prompt>Prompt.</generation_prompt>",
    ],
)
def test_parse_reasoning_output_fails_closed(value: str) -> None:
    with pytest.raises(ValueError):
        parse_reasoning_output(value)
