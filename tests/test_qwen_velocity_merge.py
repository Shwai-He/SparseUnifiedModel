import hashlib
import importlib.util
import math
from pathlib import Path
import sys

import pytest
import torch

MODULE_PATH = Path(__file__).parents[1] / "eval" / "gen" / "qwen_velocity_merge.py"
sys.path.insert(0, str(MODULE_PATH.parent))
SPEC = importlib.util.spec_from_file_location("qwen_velocity_merge", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_file_sha256(tmp_path):
    artifact = tmp_path / "sample.jsonl"
    artifact.write_bytes(b"velocity merge metadata\n")
    expected = hashlib.sha256(b"velocity merge metadata\n").hexdigest()
    assert MODULE.file_sha256(artifact) == expected


def test_tensor_sha256_reproducibility():
    t1 = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)
    t2 = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)
    t3 = torch.tensor([1.0, 2.0, 3.0, 4.0001], dtype=torch.float32)

    assert MODULE.tensor_sha256(t1) == MODULE.tensor_sha256(t2)
    assert MODULE.tensor_sha256(t1) != MODULE.tensor_sha256(t3)


@pytest.mark.parametrize("alpha_0", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_compute_alpha_constant(alpha_0):
    total_steps = 30
    for step in range(total_steps):
        alpha = MODULE.compute_alpha("constant", step, total_steps, alpha_0=alpha_0)
        assert alpha == pytest.approx(alpha_0)


def test_compute_alpha_linear_decay():
    total_steps = 11  # steps 0 to 10
    alpha_0 = 1.0
    alphas = [MODULE.compute_alpha("linear_decay", i, total_steps, alpha_0=alpha_0) for i in range(total_steps)]

    # Boundary conditions: step 0 -> alpha_0, step 10 -> 0.0
    assert alphas[0] == pytest.approx(1.0)
    assert alphas[-1] == pytest.approx(0.0)
    # Midpoint step 5 -> 0.5
    assert alphas[5] == pytest.approx(0.5)

    # Monotonicity check
    for i in range(len(alphas) - 1):
        assert alphas[i] >= alphas[i + 1]


def test_compute_alpha_cosine_decay():
    total_steps = 21  # steps 0 to 20
    alpha_0 = 0.8
    alphas = [MODULE.compute_alpha("cosine_decay", i, total_steps, alpha_0=alpha_0) for i in range(total_steps)]

    # Boundary conditions: step 0 -> alpha_0, step 20 -> 0.0
    assert alphas[0] == pytest.approx(alpha_0)
    assert alphas[-1] == pytest.approx(0.0)
    # Midpoint step 10 -> alpha_0 / 2
    assert alphas[10] == pytest.approx(alpha_0 * 0.5)

    # Monotonicity and bounds
    for a in alphas:
        assert 0.0 <= a <= alpha_0
    for i in range(len(alphas) - 1):
        assert alphas[i] >= alphas[i + 1]


def test_compute_alpha_early_half():
    total_steps = 10
    alpha_0 = 1.0
    alphas = [MODULE.compute_alpha("early_half", i, total_steps, alpha_0=alpha_0) for i in range(total_steps)]

    for i in range(5):
        assert alphas[i] == pytest.approx(1.0)
    for i in range(5, 10):
        assert alphas[i] == pytest.approx(0.0)


def test_compute_alpha_quadratic_decay():
    total_steps = 5
    alpha_0 = 1.0
    # tau at steps 0, 1, 2, 3, 4 -> 1.0, 0.75, 0.5, 0.25, 0.0
    # tau^2 -> 1.0, 0.5625, 0.25, 0.0625, 0.0
    alphas = [MODULE.compute_alpha("quadratic_decay", i, total_steps, alpha_0=alpha_0) for i in range(total_steps)]
    assert alphas[0] == pytest.approx(1.0)
    assert alphas[2] == pytest.approx(0.25)
    assert alphas[4] == pytest.approx(0.0)


@pytest.mark.parametrize("invalid_alpha_0", [-0.1, 1.1, 2.0])
def test_compute_alpha_rejects_invalid_alpha_0(invalid_alpha_0):
    with pytest.raises(ValueError, match="alpha_0"):
        MODULE.compute_alpha("constant", 0, 10, alpha_0=invalid_alpha_0)


def test_compute_alpha_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="total_steps"):
        MODULE.compute_alpha("constant", 0, 0, alpha_0=0.5)
    with pytest.raises(ValueError, match="step_index"):
        MODULE.compute_alpha("constant", 10, 10, alpha_0=0.5)
    with pytest.raises(ValueError, match="Unsupported schedule_type"):
        MODULE.compute_alpha("exponential_weird", 0, 10, alpha_0=0.5)


def test_merge_velocities_endpoints():
    v_direct = torch.randn(2, 16, 32, 32)
    v_reasoning = torch.randn(2, 16, 32, 32)

    # alpha = 0.0 -> exact identity to v_direct
    merged_0 = MODULE.merge_velocities(v_direct, v_reasoning, 0.0, method="linear")
    assert torch.equal(merged_0, v_direct)

    # alpha = 1.0 -> exact identity to v_reasoning
    merged_1 = MODULE.merge_velocities(v_direct, v_reasoning, 1.0, method="linear")
    assert torch.equal(merged_1, v_reasoning)


def test_merge_velocities_linear_midpoint():
    v_direct = torch.tensor([[1.0, 3.0], [5.0, 7.0]])
    v_reasoning = torch.tensor([[3.0, 7.0], [1.0, -1.0]])

    merged = MODULE.merge_velocities(v_direct, v_reasoning, 0.5, method="linear")
    expected = torch.tensor([[2.0, 5.0], [3.0, 3.0]])
    assert torch.allclose(merged, expected)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_merge_velocities_preserves_shape_and_dtype(dtype):
    shape = (1, 16, 32, 32)
    v_d = torch.randn(shape, dtype=dtype)
    v_r = torch.randn(shape, dtype=dtype)

    for method in ["linear", "norm_preserving", "slerp"]:
        merged = MODULE.merge_velocities(v_d, v_r, 0.35, method=method)
        assert merged.shape == shape
        assert merged.dtype == dtype
        assert torch.isfinite(merged).all()


def test_merge_velocities_norm_preserving():
    # Orthogonal vectors with unit norms
    v_d = torch.tensor([[1.0, 0.0]])
    v_r = torch.tensor([[0.0, 1.0]])

    # Linear merge has norm sqrt(0.5^2 + 0.5^2) = 1/sqrt(2) ~ 0.7071
    v_lin = MODULE.merge_velocities(v_d, v_r, 0.5, method="linear")
    assert torch.linalg.vector_norm(v_lin).item() < 1.0

    # Norm-preserving merge preserves target norm = 0.5 * 1.0 + 0.5 * 1.0 = 1.0
    v_norm = MODULE.merge_velocities(v_d, v_r, 0.5, method="norm_preserving")
    assert torch.linalg.vector_norm(v_norm).item() == pytest.approx(1.0, rel=1e-5)


def test_merge_velocities_rejects_mismatch_or_invalid_alpha():
    v1 = torch.ones(2, 4)
    v2 = torch.ones(3, 4)
    with pytest.raises(ValueError, match="Shape mismatch"):
        MODULE.merge_velocities(v1, v2, 0.5)

    with pytest.raises(ValueError, match="alpha"):
        MODULE.merge_velocities(v1, v1, -0.1)

    with pytest.raises(ValueError, match="Unsupported merge method"):
        MODULE.merge_velocities(v1, v1, 0.5, method="invalid_method")


def test_cosine_similarity_tensor():
    v1 = torch.tensor([1.0, 0.0, 0.0])
    v2 = torch.tensor([0.0, 1.0, 0.0])
    v3 = torch.tensor([-2.0, 0.0, 0.0])
    v4 = torch.tensor([3.0, 0.0, 0.0])

    # Orthogonal
    assert MODULE.cosine_similarity_tensor(v1, v2) == pytest.approx(0.0)
    # Anti-parallel
    assert MODULE.cosine_similarity_tensor(v1, v3) == pytest.approx(-1.0)
    # Parallel with scale difference
    assert MODULE.cosine_similarity_tensor(v1, v4) == pytest.approx(1.0)


def test_apply_qwen_cfg_invariants():
    cond = torch.randn(2, 8, 16)
    uncond = torch.randn(2, 8, 16)

    # cfg_scale == 1.0 returns cond
    out_1 = MODULE.apply_qwen_cfg(cond, uncond, cfg_scale=1.0)
    assert torch.equal(out_1, cond)

    # uncond is None returns cond
    out_none = MODULE.apply_qwen_cfg(cond, None, cfg_scale=4.0)
    assert torch.equal(out_none, cond)

    # cfg_scale > 1.0 scales norm to cond_norm
    out_cfg = MODULE.apply_qwen_cfg(cond, uncond, cfg_scale=4.0)
    norm_cond = torch.norm(cond, dim=-1, keepdim=True)
    norm_out = torch.norm(out_cfg, dim=-1, keepdim=True)
    assert torch.allclose(norm_cond, norm_out, atol=1e-5)


def test_euler_step_flow_match_invariants():
    z_t = torch.tensor([[10.0, 20.0]])
    v = torch.tensor([[2.0, -4.0]])
    dt = -0.5  # flow matching moving towards 0

    z_next = MODULE.euler_step_flow_match(z_t, v, dt)
    expected = torch.tensor([[9.0, 22.0]])
    assert torch.allclose(z_next, expected)


def test_build_schedule_specs():
    alphas = [0.0, 0.5, 1.0]
    schedules = ["linear_decay", "cosine_decay"]
    specs = MODULE.build_schedule_specs(alphas, schedules, schedule_alpha_0=1.0)

    assert len(specs) == 5
    assert specs[0].name == "const_alpha_0"
    assert "Direct" in specs[0].label
    assert specs[1].name == "const_alpha_0.5"
    assert specs[2].name == "const_alpha_1"
    assert "Reasoning" in specs[2].label
    assert specs[3].name == "linear_decay"
    assert specs[4].name == "cosine_decay"


def test_multi_step_trajectory_simulation():
    """Simulate a multi-step Euler flow matching integration with synthetic velocity fields."""
    num_steps = 6
    shape = (1, 4, 8, 8)

    # Fixed starting latent z_T
    generator = torch.Generator().manual_seed(101)
    initial_z = torch.randn(shape, generator=generator)

    # Deterministic synthetic velocity fields
    v_direct = torch.ones(shape) * 1.5
    v_reasoning = torch.ones(shape) * -0.5

    trajectories = {}
    specs = [
        MODULE.ScheduleSpec("direct", "constant", 0.0, "Direct"),
        MODULE.ScheduleSpec("reasoning", "constant", 1.0, "Reasoning"),
        MODULE.ScheduleSpec("midpoint", "constant", 0.5, "Midpoint"),
        MODULE.ScheduleSpec("decay", "linear_decay", 1.0, "Linear Decay"),
    ]

    dt = -1.0 / num_steps

    for spec in specs:
        z = initial_z.clone()
        step_records = []
        for step_idx in range(num_steps):
            alpha = MODULE.compute_alpha(spec.schedule_type, step_idx, num_steps, alpha_0=spec.alpha_0)
            v_merged = MODULE.merge_velocities(v_direct, v_reasoning, alpha, method="linear")
            z = MODULE.euler_step_flow_match(z, v_merged, dt)
            cos_sim = MODULE.cosine_similarity_tensor(v_direct, v_reasoning)
            step_records.append({"step": step_idx, "alpha": alpha, "cos_sim": cos_sim, "z_norm": z.norm().item()})

        trajectories[spec.name] = (z, step_records)

    # All trajectories start from the identical initial_z, but arrive at distinct destinations
    z_direct, _ = trajectories["direct"]
    z_reasoning, _ = trajectories["reasoning"]
    z_midpoint, _ = trajectories["midpoint"]
    z_decay, records_decay = trajectories["decay"]

    assert not torch.allclose(z_direct, z_reasoning)
    assert not torch.allclose(z_direct, z_midpoint)
    assert not torch.allclose(z_reasoning, z_decay)

    # Direct stepped with 1.5, Reasoning stepped with -0.5, Midpoint stepped with 0.5
    # Total shift: num_steps * dt * v = 6 * (-1/6) * v = -1 * v
    expected_direct = initial_z - v_direct
    expected_reasoning = initial_z - v_reasoning
    expected_midpoint = initial_z - 0.5 * (v_direct + v_reasoning)

    assert torch.allclose(z_direct, expected_direct, atol=1e-5)
    assert torch.allclose(z_reasoning, expected_reasoning, atol=1e-5)
    assert torch.allclose(z_midpoint, expected_midpoint, atol=1e-5)

    # Linear decay alpha starts at 1.0 and ends at 0.0
    assert records_decay[0]["alpha"] == 1.0
    assert records_decay[-1]["alpha"] == 0.0


def test_parse_reasoning_output():
    text = (
        "<reasoning>The scene requires burning sulfur, which produces a blue flame in oxygen.</reasoning>\n"
        "<generation_prompt>A deflagrating spoon inside a glass gas jar with a vivid blue flame.</generation_prompt>"
    )
    reasoning, prompt = MODULE.parse_reasoning_output(text)
    assert "burning sulfur" in reasoning
    assert "blue flame" in prompt

    malformed = "Just some text without tags"
    with pytest.raises(ValueError, match="sections"):
        MODULE.parse_reasoning_output(malformed)
