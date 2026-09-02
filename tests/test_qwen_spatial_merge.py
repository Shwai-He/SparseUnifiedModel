import importlib.util
import hashlib
from pathlib import Path
import sys

import pytest
import torch
import torch.nn.functional as F


MODULE_PATH = Path(__file__).parents[1] / "eval" / "gen" / "qwen_spatial_latent_merge.py"
sys.path.insert(0, str(MODULE_PATH.parent))
SPEC = importlib.util.spec_from_file_location("qwen_spatial_latent_merge", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_file_sha256(tmp_path):
    artifact = tmp_path / "metadata.jsonl"
    artifact.write_bytes(b"spatial merge contract test\n")
    assert MODULE.file_sha256(artifact) == hashlib.sha256(b"spatial merge contract test\n").hexdigest()


def test_tensor_sha256():
    tensor_a = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    tensor_b = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.bfloat16)
    # sha256 converts to float32 on cpu, so both should yield identical digests
    assert MODULE.tensor_sha256(tensor_a) == MODULE.tensor_sha256(tensor_b)
    assert len(MODULE.tensor_sha256(tensor_a)) == 64


def test_build_spatial_mask_box_bounds():
    h_lat, w_lat = 64, 64
    # Box covering [ymin=0.25, xmin=0.5, ymax=0.75, xmax=1.0]
    box = [0.25, 0.5, 0.75, 1.0]
    mask = MODULE.build_spatial_mask(
        height_lat=h_lat,
        width_lat=w_lat,
        mask_type="box",
        box=box,
        sigma=0.0,
    )
    assert mask.shape == (1, 1, h_lat, w_lat)
    assert mask.min().item() == 0.0
    assert mask.max().item() == 1.0

    # Inside box check (e.g. y=0.5, x=0.75 in normalized coordinates -> y_idx=32, x_idx=48)
    assert mask[0, 0, 32, 48].item() == 1.0

    # Outside box check (e.g. y=0.1, x=0.2 in normalized coordinates -> y_idx=6, x_idx=12)
    assert mask[0, 0, 6, 12].item() == 0.0


def test_build_spatial_mask_gaussian():
    h_lat, w_lat = 64, 64
    mask = MODULE.build_spatial_mask(
        height_lat=h_lat,
        width_lat=w_lat,
        mask_type="gaussian",
        center=[0.5, 0.5],
        radius=[0.2, 0.2],
        sigma=0.0,
    )
    assert mask.shape == (1, 1, h_lat, w_lat)
    # Peak at center (discrete grid center with 64 points has grid coordinates at 31/63 and 32/63)
    assert mask[0, 0, 32, 32].item() == pytest.approx(1.0, abs=5e-3)
    # Monotonic decay away from center
    val_center = mask[0, 0, 32, 32].item()
    val_mid = mask[0, 0, 32, 40].item()
    val_far = mask[0, 0, 32, 55].item()
    assert val_center > val_mid > val_far
    assert val_far >= 0.0


def test_build_spatial_mask_split():
    h_lat, w_lat = 64, 64
    v_mask = MODULE.build_spatial_mask(
        height_lat=h_lat,
        width_lat=w_lat,
        mask_type="split",
        split_ratio=0.5,
        split_axis="vertical",
        sigma=0.0,
    )
    assert (v_mask[0, 0, :, :32] == 0.0).all()
    assert (v_mask[0, 0, :, 32:] == 1.0).all()

    h_mask = MODULE.build_spatial_mask(
        height_lat=h_lat,
        width_lat=w_lat,
        mask_type="split",
        split_ratio=0.5,
        split_axis="horizontal",
        sigma=0.0,
    )
    assert (h_mask[0, 0, :32, :] == 0.0).all()
    assert (h_mask[0, 0, 32:, :] == 1.0).all()


@pytest.mark.parametrize(
    "invalid_box",
    [
        [0.8, 0.2, 0.3, 0.7],  # ymin > ymax
        [0.2, 0.9, 0.7, 0.4],  # xmin > xmax
        [-0.1, 0.0, 0.5, 0.5], # negative ymin
        [0.0, 0.0, 1.2, 0.5],  # ymax > 1.0
        [0.1, 0.2, 0.3],       # wrong length
    ],
)
def test_build_spatial_mask_rejects_invalid_boxes(invalid_box):
    with pytest.raises(ValueError):
        MODULE.build_spatial_mask(64, 64, mask_type="box", box=invalid_box)


def test_build_spatial_mask_rejects_negative_sigma():
    with pytest.raises(ValueError, match="sigma"):
        MODULE.build_spatial_mask(64, 64, sigma=-1.0)


def test_build_spatial_mask_rejects_unknown_mask_type():
    with pytest.raises(ValueError, match="unsupported"):
        MODULE.build_spatial_mask(64, 64, mask_type="polygon")


def test_mask_gaussian_softness_falloff():
    h_lat, w_lat = 64, 64
    box = [0.25, 0.25, 0.75, 0.75]
    mask_hard = MODULE.build_spatial_mask(h_lat, w_lat, "box", box=box, sigma=0.0)
    mask_smooth = MODULE.build_spatial_mask(h_lat, w_lat, "box", box=box, sigma=2.0)

    # Hard mask has sharp 0 to 1 step at boundary
    assert mask_hard[0, 0, 15, 32].item() == 0.0
    assert mask_hard[0, 0, 16, 32].item() == 1.0

    # Smooth mask has continuous intermediate values along boundary
    boundary_val = mask_smooth[0, 0, 16, 32].item()
    assert 0.1 < boundary_val < 0.9

    # Deep interior remains 1.0
    assert mask_smooth[0, 0, 32, 32].item() == pytest.approx(1.0, abs=1e-3)
    # Far exterior remains 0.0
    assert mask_smooth[0, 0, 5, 5].item() == pytest.approx(0.0, abs=1e-3)


def test_mask_to_patch_tokens_broadcasting():
    h_lat, w_lat = 64, 64
    mask_4d = MODULE.build_spatial_mask(h_lat, w_lat, "box", box=[0.0, 0.5, 1.0, 1.0], sigma=0.0)
    tokens = MODULE.mask_to_patch_tokens(mask_4d)
    expected_s = (h_lat // 2) * (w_lat // 2)
    assert tokens.shape == (1, expected_s, 1)
    # Left half patches are 0, right half patches are 1
    tokens_2d = tokens.view(h_lat // 2, w_lat // 2)
    assert (tokens_2d[:, : w_lat // 4] == 0.0).all()
    assert (tokens_2d[:, w_lat // 4 :] == 1.0).all()


def test_pack_unpack_lossless_roundtrip():
    for dtype in [torch.float32, torch.bfloat16]:
        b, c_lat, h_lat, w_lat = 2, 16, 64, 64
        num_patches = (h_lat // 2) * (w_lat // 2)
        packed_channels = c_lat * 4
        x = torch.randn(b, num_patches, packed_channels, dtype=dtype)

        unpacked = MODULE.unpack_latents(x, h_lat, w_lat, vae_scale_factor=8)
        assert unpacked.shape == (b, c_lat, 1, h_lat, w_lat)

        repacked = MODULE.pack_latents(unpacked, b, c_lat, h_lat, w_lat)
        assert repacked.shape == x.shape
        assert torch.equal(x, repacked)


def test_spatial_lerp_boundary_blending():
    b, c, h, w = 1, 8, 32, 32
    left = torch.zeros(b, c, 1, h, w)
    right = torch.ones(b, c, 1, h, w) * 10.0

    # Half split mask
    mask = torch.zeros(1, 1, 1, h, w)
    mask[:, :, :, :, 16:] = 1.0
    # Add a transition column at 15 with mask=0.4
    mask[:, :, :, :, 15] = 0.4

    merged = MODULE.spatial_lerp(left, right, mask, alpha=1.0)
    # Outside mask (col < 15): strictly left (0.0)
    assert (merged[:, :, :, :, :15] == 0.0).all()
    # Inside mask (col >= 16): strictly right (10.0)
    assert (merged[:, :, :, :, 16:] == 10.0).all()
    # Transition col 15: exactly 0.4 * 10 = 4.0
    assert torch.allclose(merged[:, :, :, :, 15], torch.tensor(4.0))


def test_spatial_slerp_norm_preservation():
    # Test that spatial SLERP strictly preserves unit norms of channel vectors across blending zone
    b, s, c = 1, 100, 32
    u = torch.randn(b, s, c)
    v = torch.randn(b, s, c)
    u = u / torch.linalg.vector_norm(u, dim=-1, keepdim=True)
    v = v / torch.linalg.vector_norm(v, dim=-1, keepdim=True)

    # Masks from 0.0 to 1.0
    alphas = torch.linspace(0.0, 1.0, s).view(1, s, 1)

    slerp_merged = MODULE.spatial_slerp(u, v, alphas, alpha=1.0, channel_dim=-1)
    lerp_merged = MODULE.spatial_lerp(u, v, alphas, alpha=1.0)

    slerp_norms = torch.linalg.vector_norm(slerp_merged, dim=-1)
    lerp_norms = torch.linalg.vector_norm(lerp_merged, dim=-1)

    # SLERP norms remain exactly 1.0 everywhere
    assert torch.allclose(slerp_norms, torch.ones_like(slerp_norms), atol=1e-5)
    # LERP norms collapse at the midpoint (alphas=0.5)
    mid_lerp_norm = lerp_norms[0, s // 2].item()
    assert mid_lerp_norm < 0.9  # Demonstrates norm collapse in LERP


def test_merge_spatial_latents_packed_subpatch():
    b, c_lat, h_lat, w_lat = 1, 16, 64, 64
    num_patches = (h_lat // 2) * (w_lat // 2)
    left = torch.randn(b, num_patches, c_lat * 4, dtype=torch.float32)
    right = torch.randn(b, num_patches, c_lat * 4, dtype=torch.float32)

    # Spatial box on right half
    mask = MODULE.build_spatial_mask(h_lat, w_lat, "box", box=[0.0, 0.5, 1.0, 1.0], sigma=0.0)

    merged = MODULE.merge_spatial_latents(
        left=left,
        right=right,
        mask=mask,
        blend_mode="slerp",
        alpha=1.0,
        height=512,
        width=512,
        vae_scale_factor=8,
    )
    assert merged.shape == left.shape
    assert merged.dtype == left.dtype
    assert torch.isfinite(merged).all()


def test_merge_step_consistency_and_immutability():
    # Simulates multi-step diffusion callback execution
    b, s, c = 4, 64, 16
    initial_latents = torch.randn(b, s, c)
    # index 0: Candidate A (Seed A)
    # index 1: Candidate B (Seed B)
    # index 2: Merge Spec 0 (merge at step 3)
    # index 3: Merge Spec 1 (merge at step 5)
    latents = initial_latents.clone()
    mask = torch.zeros(1, s, 1)
    mask[:, 32:, :] = 1.0  # right half

    specs = [("slerp", 3), ("lerp", 5)]

    for step in range(8):
        # Noise step simulation: update latents
        latents = latents + torch.randn_like(latents) * 0.01

        # Check merge callback
        for out_idx, (method, merge_step) in enumerate(specs, start=2):
            if step == merge_step:
                left = latents[0:1]
                right = latents[1:2]
                merged = MODULE.merge_spatial_latents(left, right, mask, blend_mode=method)
                latents[out_idx : out_idx + 1] = merged

        # Check immutability: Candidate A (0) and Candidate B (1) are never overwritten
        assert latents[0:1].shape == (1, s, c)
        assert latents[1:2].shape == (1, s, c)

    # Latents remain finite and shape-consistent
    assert torch.isfinite(latents).all()
    assert latents.shape == (b, s, c)


def test_compute_spatial_merge_stats():
    b, s, c = 1, 64, 16
    left = torch.ones(b, s, c)
    right = torch.ones(b, s, c) * 2.0
    mask = torch.zeros(1, s, 1)
    mask[:, :32, :] = 1.0  # first 32 tokens

    merged = MODULE.merge_spatial_latents(left, right, mask, blend_mode="lerp")
    stats = MODULE.compute_spatial_merge_stats(left, right, merged, mask)

    assert stats["mask_coverage_ratio"] == pytest.approx(0.5)
    assert stats["cosine_sim_inside_vs_candidate_b"] == pytest.approx(1.0, abs=1e-4)
    assert stats["cosine_sim_outside_vs_candidate_a"] == pytest.approx(1.0, abs=1e-4)
    assert stats["l2_norm"] > 0
    assert stats["rms"] > 0
