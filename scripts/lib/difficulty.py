from __future__ import annotations

from typing import Iterable


def clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


def d_hat_from_features(f_node: float, f_len: float, f_jump: float, f_cross: float) -> float:
    return clip01(0.55 * f_node + 0.15 * f_len + 0.15 * f_jump + 0.15 * f_cross)


def chain_difficulty(
    node_difficulty: Iterable[float],
    edge_confidence: Iterable[float],
    domain_switches: int,
    chain_length: int,
) -> float:
    node_vals = list(node_difficulty)
    edge_vals = list(edge_confidence)

    if chain_length < 2:
        raise ValueError("chain_length must be >= 2")

    f_node = sum(node_vals) / max(1, len(node_vals))
    f_len = (chain_length - 2) / 2.0

    if edge_vals:
        f_jump = sum(1.0 - c for c in edge_vals) / len(edge_vals)
    else:
        f_jump = 0.5

    f_cross = domain_switches / max(1, (chain_length - 1))
    return d_hat_from_features(f_node=f_node, f_len=f_len, f_jump=f_jump, f_cross=f_cross)
