import importlib.util
import hashlib
from pathlib import Path
import sys

import pytest
import torch


MODULE_PATH = Path(__file__).parents[1] / "eval" / "gen" / "qwen_latent_merge.py"
sys.path.insert(0, str(MODULE_PATH.parent))
SPEC = importlib.util.spec_from_file_location("qwen_latent_merge", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_file_sha256(tmp_path):
    artifact = tmp_path / "metadata.jsonl"
    artifact.write_bytes(b"fixed metadata\n")

    assert MODULE.file_sha256(artifact) == hashlib.sha256(b"fixed metadata\n").hexdigest()


def test_lerp_midpoint():
    left = torch.tensor([[0.0, 2.0]])
    right = torch.tensor([[2.0, 4.0]])
    assert torch.equal(MODULE.merge_latents(left, right, "lerp", 0.5), torch.tensor([[1.0, 3.0]]))


def test_slerp_preserves_shape_dtype_and_finite_values():
    left = torch.tensor([[1.0, 0.0]], dtype=torch.float16)
    right = torch.tensor([[0.0, 1.0]], dtype=torch.float16)
    merged = MODULE.merge_latents(left, right, "slerp", 0.5)
    assert merged.shape == left.shape
    assert merged.dtype == left.dtype
    assert torch.isfinite(merged).all()


def test_slerp_preserves_unit_norm_between_orthogonal_endpoints():
    left = torch.tensor([[1.0, 0.0]])
    right = torch.tensor([[0.0, 1.0]])

    merged = MODULE.merge_latents(left, right, "slerp", 0.5)

    assert torch.linalg.vector_norm(merged).item() == pytest.approx(1.0)
    assert torch.linalg.vector_norm(torch.lerp(left, right, 0.5)).item() < 1.0


@pytest.mark.parametrize("alpha", [-0.1, 1.1])
def test_merge_rejects_invalid_alpha(alpha):
    with pytest.raises(ValueError, match="alpha"):
        MODULE.merge_latents(torch.zeros(1), torch.ones(1), "lerp", alpha)


def test_merge_rejects_unknown_method():
    with pytest.raises(ValueError, match="unsupported"):
        MODULE.merge_latents(torch.zeros(1), torch.ones(1), "mean", 0.5)


def test_qwen_generate_uses_explicit_consecutive_seeds():
    from models.qwen import QwenGenModel

    class FakePipeline:
        def __init__(self):
            self.seeds = []

        def __call__(self, **kwargs):
            self.seeds.append(kwargs["generator"].initial_seed())
            return type("Output", (), {"images": [object()]})()

    pipe = FakePipeline()
    model = QwenGenModel(pipe, "cpu")
    images = model.generate("prompt", num_images=3, seed=17, device="cpu")
    assert len(images) == 3
    assert pipe.seeds == [17, 18, 19]


def test_geneval_prompt_seed_ranges_do_not_overlap():
    base_seed = 42
    num_images = 4
    first = [base_seed + 80 * num_images + i for i in range(num_images)]
    second = [base_seed + 81 * num_images + i for i in range(num_images)]
    assert first == [362, 363, 364, 365]
    assert set(first).isdisjoint(second)


def test_bagel_seeded_latent_is_replayable_and_isolated():
    from models.bagel import prepare_seeded_vae_latent

    class FakeBagel:
        @staticmethod
        def prepare_vae_latent(**_kwargs):
            return torch.randn(8)

    torch.manual_seed(999)
    expected_next = torch.randn(1)
    torch.manual_seed(999)
    first = prepare_seeded_vae_latent(FakeBagel(), 17)
    actual_next = torch.randn(1)
    replay = prepare_seeded_vae_latent(FakeBagel(), 17)
    other = prepare_seeded_vae_latent(FakeBagel(), 18)

    assert torch.equal(first, replay)
    assert not torch.equal(first, other)
    assert torch.equal(actual_next, expected_next)
