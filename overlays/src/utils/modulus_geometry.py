"""Geometry helpers for MODULUS-ELF latent-flow experiments.

These utilities are intentionally small and JAX-only. They do not alter upstream
ELF behavior by themselves; callers must explicitly route through them via a
configuration gate such as ``config.latent_geometry``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

Array = jnp.ndarray


def safe_norm(x: Array, axis: int = -1, keepdims: bool = True, eps: float = 1e-6) -> Array:
    """Return numerically safe Euclidean norm."""
    return jnp.maximum(jnp.linalg.norm(x, axis=axis, keepdims=keepdims), eps)


def unit(x: Array, axis: int = -1, eps: float = 1e-6) -> Array:
    """Normalize vectors onto a unit sphere along ``axis``."""
    return x / safe_norm(x, axis=axis, keepdims=True, eps=eps)


def project_tangent(v: Array, base: Array, axis: int = -1, eps: float = 1e-6) -> Array:
    """Project ``v`` onto the tangent space at ``unit(base)``."""
    u = unit(base, axis=axis, eps=eps)
    radial = jnp.sum(v * u, axis=axis, keepdims=True) * u
    return v - radial


def hybrid_sphere_decompose(x: Array, eps: float = 1e-6) -> tuple[Array, Array]:
    """Return tokenwise radius and direction for a hybrid radius+sphere latent."""
    r = safe_norm(x, axis=-1, keepdims=True, eps=eps)
    u = x / r
    return r, u


def hybrid_sphere_compose(radius: Array, direction: Array, eps: float = 1e-6) -> Array:
    """Compose a hybrid latent from radius and normalized direction."""
    return radius * unit(direction, eps=eps)


def angular_lerp(x0: Array, noise: Array, t: Array, eps: float = 1e-6) -> Array:
    """Linear interpolation of directions followed by normalization.

    This is intentionally conservative: it avoids a full geodesic SLERP branch
    until baseline and geometry-off gates are cleared.
    """
    t_expanded = t.reshape(-1, 1, 1)
    u0 = unit(x0, eps=eps)
    un = unit(noise, eps=eps)
    return unit(t_expanded * u0 + (1.0 - t_expanded) * un, eps=eps)


def hybrid_sphere_add_noise(
    x0: Array,
    noise: Array,
    t: Array,
    noise_scale: float,
    eps: float = 1e-6,
) -> Array:
    """Construct a hybrid-sphere noisy latent.

    Radius is interpolated in the upstream Euclidean scale while direction is
    normalized after angular interpolation. This preserves a radius channel for
    T5 latent magnitude information while constraining angular dynamics.
    """
    t_expanded = t.reshape(-1, 1, 1)
    r0, _ = hybrid_sphere_decompose(x0, eps=eps)
    rn, _ = hybrid_sphere_decompose(noise * noise_scale, eps=eps)
    radius = t_expanded * r0 + (1.0 - t_expanded) * rn
    direction = angular_lerp(x0, noise, t, eps=eps)
    return hybrid_sphere_compose(radius, direction, eps=eps)


def geometry_add_noise(
    x0: Array,
    noise: Array,
    t: Array,
    config,
    cond_seq_mask: Array | None = None,
) -> Array:
    """Geometry-aware replacement candidate for upstream ``add_noise``.

    ``latent_geometry='off'`` exactly mirrors the upstream Euclidean formula.
    """
    geometry = getattr(config, "latent_geometry", "off")
    eps = getattr(config, "modulus_eps", 1e-6)
    noise_scale = getattr(config, "denoiser_noise_scale", 1.0)
    t_expanded = t.reshape(-1, 1, 1)

    if geometry == "off":
        z = t_expanded * x0 + (1.0 - t_expanded) * noise * noise_scale
    elif geometry == "sphere":
        z = unit(t_expanded * unit(x0, eps=eps) + (1.0 - t_expanded) * unit(noise, eps=eps), eps=eps)
    elif geometry == "hybrid_sphere":
        z = hybrid_sphere_add_noise(x0, noise, t, noise_scale=noise_scale, eps=eps)
    elif geometry == "channel_sphere":
        # Conservative channel-normalized variant: normalize each token vector but
        # leave upstream scale available through decoder/training diagnostics.
        z_euclidean = t_expanded * x0 + (1.0 - t_expanded) * noise * noise_scale
        z = unit(z_euclidean, eps=eps) * safe_norm(x0, eps=eps)
    else:
        raise ValueError(f"unknown latent_geometry: {geometry}")

    if cond_seq_mask is not None:
        z = cond_seq_mask * x0 + (1.0 - cond_seq_mask) * z
    return z


def geometry_velocity_target(x0: Array, z: Array, t: Array, config) -> Array:
    """Return a velocity target compatible with the chosen geometry.

    For WP00 this deliberately keeps the upstream finite-difference target and
    optionally projects it onto the tangent space. Later work packages may add
    true Riemannian log-map targets.
    """
    t_eps = getattr(config, "t_eps", 5e-2)
    t_expanded = t.reshape(-1, 1, 1)
    v = (x0 - z) / jnp.maximum(1.0 - t_expanded, t_eps)

    if getattr(config, "modulus_project_velocity", True):
        geometry = getattr(config, "latent_geometry", "off")
        if geometry in ("sphere", "hybrid_sphere", "channel_sphere"):
            v = project_tangent(v, z, eps=getattr(config, "modulus_eps", 1e-6))
    return v


def geometry_diagnostics(x: Array, v: Array | None = None, eps: float = 1e-6) -> dict[str, Array]:
    """Return lightweight latent geometry diagnostics for logging."""
    radius = safe_norm(x, eps=eps)
    out = {
        "latent_radius_mean": jnp.mean(radius),
        "latent_radius_std": jnp.std(radius),
        "latent_unit_norm_error": jnp.mean(jnp.abs(safe_norm(unit(x, eps=eps), eps=eps) - 1.0)),
    }
    if v is not None:
        u = unit(x, eps=eps)
        out["latent_velocity_radial_abs_mean"] = jnp.mean(jnp.abs(jnp.sum(v * u, axis=-1)))
    return out
