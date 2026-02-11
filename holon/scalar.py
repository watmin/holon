"""
Scalar Encoding: Continuous value encoding for VSA/HDC.

Mirrors holon-rs/src/scalar.rs for cross-language parity.

Unlike structured data encoding (where {"x": 5} is unrelated to {"x": 6}),
scalar encoding creates vectors where NEARBY VALUES are SIMILAR.

## Quick Reference

- `encode_scalar(value, dimensions, mode, scale, period)` → Encode continuous value
- `encode_scalar_log(value, dimensions, scale)` → Log-scale encoding
- `encode_circular(value, period, dimensions)` → Circular/periodic encoding
- `encode_positional(position, dimensions, scale)` → Transformer-style positional

## Modes

- **linear**: Nearby values have similar vectors, no wrapping.
  Good for: rate, distance, temperature, price

- **circular**: Values wrap at period (0 ≈ period).
  Good for: angle, hour of day, day of week, month
"""

import numpy as np


def encode_circular(
    value: float, period: float, dimensions: int, seed: int = 42
) -> np.ndarray:
    """
    Encode a value on a circle with given period.

    Values that are close on the circle will have similar encodings.
    The encoding wraps: value 0 is similar to value `period`.

    Args:
        value: The value to encode
        period: The period of the circle (e.g., 24 for hours)
        dimensions: Vector dimensionality
        seed: Random seed for phase generation

    Returns:
        Bipolar vector encoding
    """
    rng = np.random.default_rng(seed + int(period * 1000))
    angle = 2 * np.pi * value / period

    # Random phase offsets for each dimension
    phases = rng.uniform(0, 2 * np.pi, dimensions)

    # Project angle onto random directions
    return np.sign(np.cos(angle + phases)).astype(np.int8)


def encode_positional(
    position: float, dimensions: int, scale: float = 10000
) -> np.ndarray:
    """
    Transformer-style positional encoding for linear values.

    Nearby positions have similar encodings, with gradual decay.

    Args:
        position: The position/value to encode
        dimensions: Vector dimensionality
        scale: Controls similarity decay rate (larger = slower decay)

    Returns:
        Bipolar vector encoding
    """
    indices = np.arange(dimensions)
    freqs = 1 / (scale ** (indices / dimensions))

    # Alternate sin/cos
    values = np.where(
        indices % 2 == 0,
        np.sin(position * freqs),
        np.cos(position * freqs),
    )

    return np.sign(values).astype(np.int8)


def encode_scalar(
    value: float,
    dimensions: int,
    mode: str = "linear",
    scale: float = 10000.0,
    period: float = None,
    seed: int = 42,
) -> np.ndarray:
    """
    Encode a continuous scalar value into a vector.

    Creates vectors where nearby values are similar.

    Args:
        value: The scalar value to encode
        dimensions: Vector dimensionality
        mode: "linear" or "circular"
        scale: For linear mode, controls similarity decay rate
        period: For circular mode, the period of wrapping (required)
        seed: Random seed for circular mode phase generation

    Returns:
        Bipolar vector encoding

    Examples:
        # Rate encoding (100 pps similar to 110 pps)
        v100 = encode_scalar(100, 4096, mode="linear")
        v110 = encode_scalar(110, 4096, mode="linear")
        # similarity(v100, v110) ≈ 0.95

        # Hour of day (hour 23 similar to hour 0)
        h23 = encode_scalar(23, 4096, mode="circular", period=24)
        h0 = encode_scalar(0, 4096, mode="circular", period=24)
        # similarity(h23, h0) ≈ 0.90
    """
    if mode == "linear":
        return encode_positional(value, dimensions, scale)
    elif mode == "circular":
        if period is None:
            raise ValueError("period is required for circular mode")
        return encode_circular(value, period, dimensions, seed)
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'linear' or 'circular'.")


def encode_scalar_log(
    value: float,
    dimensions: int,
    scale: float = 1000.0,
) -> np.ndarray:
    """
    Encode a scalar on log scale.

    Equal ratios have equal similarity:
        100 → 1000 is same "distance" as 1000 → 10000

    Args:
        value: The scalar value (must be > 0)
        dimensions: Vector dimensionality
        scale: Controls similarity decay rate

    Returns:
        Bipolar vector encoding of log10(value)

    Example:
        # Rate encoding where 10x change is consistent similarity drop
        v100 = encode_scalar_log(100, 4096)
        v1000 = encode_scalar_log(1000, 4096)
        v10000 = encode_scalar_log(10000, 4096)
        # similarity(v100, v1000) ≈ similarity(v1000, v10000)
    """
    if value <= 0:
        value = 1e-10  # Avoid log(0)
    log_value = np.log10(value)
    return encode_positional(log_value, dimensions, scale)
