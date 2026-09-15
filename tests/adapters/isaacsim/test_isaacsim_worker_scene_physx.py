"""Tests for the IsaacSim worker scene-level PhysX configuration (F1/F2).

Pure-Python coverage only: the exact values of the original repository's
scene-level ``PhysxCfg`` port, the env grid spacing, and the multi-asset vs
legacy resolution.  The Kit-side application is exercised by the INIT meta
``scene_physx`` readback consumed by ``probes/probe_b2_full_scene_readback.py``.
"""

from __future__ import annotations

from unisim.backend.isaacsim.worker import (
    LEGACY_ENV_GRID_SPACING,
    SIMTOOLREAL_ENV_GRID_SPACING,
    SIMTOOLREAL_ROBOT_INIT_POS,
    SIMTOOLREAL_ROBOT_INIT_ROT_WXYZ,
    SIMTOOLREAL_SCENE_PHYSX_KWARGS,
    resolve_env_grid_spacing,
    resolve_scene_physx_kwargs,
)


def test_simtoolreal_scene_physx_kwargs_match_original_repository():
    # Literal port of simtoolreal_env_cfg.py:515-535 `_default_sim_cfg`.
    assert SIMTOOLREAL_SCENE_PHYSX_KWARGS == {
        "solver_type": 1,  # 1 = TGS (matches legacy)
        "min_position_iteration_count": 8,
        "max_position_iteration_count": 8,
        "min_velocity_iteration_count": 0,
        "max_velocity_iteration_count": 0,
        "bounce_threshold_velocity": 0.2,
        "friction_offset_threshold": 0.04,
        "friction_correlation_distance": 0.025,
        "gpu_max_rigid_contact_count": 2**24,  # 16777216
        "gpu_max_rigid_patch_count": 2**23,  # 8388608
    }


def test_resolve_scene_physx_kwargs_multi_asset_vs_legacy():
    multi_asset = resolve_scene_physx_kwargs([{"name": "robot"}])
    assert multi_asset == SIMTOOLREAL_SCENE_PHYSX_KWARGS
    # The returned mapping is a copy: mutating it must not leak into the
    # module-level constant.
    assert multi_asset is not SIMTOOLREAL_SCENE_PHYSX_KWARGS
    multi_asset["bounce_threshold_velocity"] = 999.0
    assert SIMTOOLREAL_SCENE_PHYSX_KWARGS["bounce_threshold_velocity"] == 0.2
    # Legacy single-asset scenes keep Isaac Lab defaults (byte-identical INIT).
    assert resolve_scene_physx_kwargs([]) is None


def test_env_grid_spacing_values_and_resolution():
    # Original repository env_spacing=1.2 (SimToolReal.yaml:34); the legacy
    # single-asset path keeps its historical 2.0.
    assert SIMTOOLREAL_ENV_GRID_SPACING == 1.2
    assert LEGACY_ENV_GRID_SPACING == 2.0
    assert resolve_env_grid_spacing([{"name": "robot"}]) == 1.2
    assert resolve_env_grid_spacing([]) == 2.0


def test_robot_init_pose_matches_original_repository():
    # scene_utils.py:1811-1842, pos at :1821: the fixed-base iiwa stands
    # 0.8 m behind the table; identity orientation (wxyz).
    assert SIMTOOLREAL_ROBOT_INIT_POS == (0.0, 0.8, 0.0)
    assert SIMTOOLREAL_ROBOT_INIT_ROT_WXYZ == (1.0, 0.0, 0.0, 0.0)
