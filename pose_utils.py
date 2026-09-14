"""Pose conventions shared by every approach

camera_point = scale * rotation @ p_template + translation

- Rotations map mesh template coordiantes into camera coordinates
- Translations are expressed in camera coordinates and meters
The assignment maps an original template point into camera coordinates as
"""

import numpy as np

def transform_points(points, rotation, translation, scale=1.0):
    """Transform template-space (mesh) row points into camera coordinates"""
    points = np.asarray(points, dtype=np.float64)
    rotation = np.asarray(rotation, dtype=np.float64)
    translation = np.asarray(translation, dtype=np.float64)

    # Check dimensionality
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must have shape (N, 3), got {points.shape}")
    if rotation.shape != (3, 3):
        raise ValueError(f"rotation must have shape (3, 3), got {rotation.shape}")
    if translation.shape != (3,):
        raise ValueError(
            f"translation must have shape (3,), got {translation.shape}"
        )
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(f"scale must be finite and positive, got {scale}")

    # Perform calculation; Convert to camera coordinates
    return float(scale) * (points @ rotation.T) + translation

def center_template(vertices):
    """Return centered vertices"""
    vertices = np.asarray(vertices, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"vertices must have shape (N, 3), got {vertices.shape}")

    center = vertices.mean(axis=0)
    return vertices - center, center

def centered_to_original_translation(template_center, rotation, centered_translation, scale=1.0):
    """Convert centered-template translation to the original convention"""
    template_center = np.asarray(template_center, dtype=np.float64)
    rotation = np.asarray(rotation, dtype=np.float64)
    centered_translation = np.asarray(centered_translation, dtype=np.float64)

    return (centered_translation - float(scale) * (rotation @ template_center))

def original_to_centered_translation(template_center, rotation, original_translation, scale=1.0):
    """Convert original convention to centered-template translation"""
    template_center = np.asarray(template_center, dtype=np.float64)
    rotation = np.asarray(rotation, dtype=np.float64)
    original_translation = np.asarray(original_translation, dtype=np.float64)

    return original_translation + float(scale) * (rotation @ template_center)

def prepare_scales(scales, n_frames):
    """Return scalar array representing one sequence-wide scale; repeated for every frame"""
    scales = np.asarray(scales, dtype=np.float64)

    if scales.ndim == 0 or scales.size == 1:
        return np.full(n_frames, float(scales.reshape(-1)[0]))
    if scales.shape != (n_frames,):
        raise ValueError(
            f"scale must be scalar or have shape ({n_frames},), "
            f"got {scales.shape}"
        )
    return scales

def rotation_geodesic(rotation_a, rotation_b):
    """Compute Relative Rotation"""
    relative = np.asarray(rotation_a).T @ np.asarray(rotation_b)
    cosine = (np.trace(relative) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))

def validate_trajectory(rotations, translations, scales, atol=1e-5):
    """Validate trajectory shapes, finite values, scales, and SO(3) rotations"""
    rotations = np.asarray(rotations, dtype=np.float64)
    translations = np.asarray(translations, dtype=np.float64)

    if rotations.ndim != 3 or rotations.shape[1:] != (3, 3):
        raise ValueError(f"R must have shape (T, 3, 3), got {rotations.shape}")

    n_frames = len(rotations)
    if translations.shape != (n_frames, 3):
        raise ValueError(
            f"t must have shape ({n_frames}, 3), got {translations.shape}"
        )

    scales = prepare_scales(scales, n_frames)

    if not np.isfinite(rotations).all():
        raise ValueError("R contains NaN or infinity")
    if not np.isfinite(translations).all():
        raise ValueError("t contains NaN or infinity")
    if not np.isfinite(scales).all() or np.any(scales <= 0):
        raise ValueError("s must contain only finite, positive values")

    identity = np.eye(3)
    orthogonality_errors = np.linalg.norm(np.swapaxes(rotations, 1, 2) @ rotations - identity,
            axis=(1, 2))
    determinant_errors = np.abs(np.linalg.det(rotations) - 1.0)

    max_orthogonality_error = float(orthogonality_errors.max(initial=0.0))
    max_determinant_error = float(determinant_errors.max(initial=0.0))

    if max_orthogonality_error > atol:
        raise ValueError(
            "R contains a non-orthonormal matrix; maximum error is "
            f"{max_orthogonality_error:.3e}"
        )
    if max_determinant_error > atol:
        raise ValueError(
            "R contains a matrix whose determinant is not +1; maximum error "
            f"is {max_determinant_error:.3e}"
        )

    return {
        "n_frames": n_frames,
        "max_orthogonality_error": max_orthogonality_error,
        "max_determinant_error": max_determinant_error,
        "minimum_scale": float(scales.min(initial=np.inf)),
        "maximum_scale": float(scales.max(initial=-np.inf)),
    }

def _self_check():
    """Confirm that centered and original-template formulas are equivalent"""
    rng = np.random.default_rng(0)
    vertices = rng.normal(size=(100, 3))
    centered_vertices, center = center_template(vertices)

    # A known valid rotation: 90 degrees around camera Z
    rotation = np.array([
        [0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    centered_translation = np.array([0.1, -0.2, 2.5])
    scale = 1.03

    via_centered = transform_points(centered_vertices, rotation, centered_translation, scale)
    original_translation = centered_to_original_translation(center, rotation,
                                                            centered_translation, scale)
    via_original = transform_points(vertices, rotation, original_translation, scale)

    np.testing.assert_allclose(via_centered, via_original, atol=1e-12)

    recovered_translation = original_to_centered_translation(center, rotation,
                                                             original_translation, scale)
    np.testing.assert_allclose(recovered_translation, centered_translation, atol=1e-12)

    report = validate_trajectory(rotation[None, :, :], original_translation[None, :], scale)
    print("Pose convention self-check passed.")
    print(report)

if __name__ == "__main__":
    _self_check()
