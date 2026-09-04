from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


class ScalarMap(Protocol):
    """How: ベクトル化された実数配列へ同じ局所写像を適用する契約。"""

    def __call__(self, x: FloatArray) -> FloatArray:
        ...

    def derivative(self, x: FloatArray) -> FloatArray:
        ...


@dataclass(frozen=True)
class BooleMap:
    """How: 一般化 Boole 写像 F_alpha(x)=alpha(x-1/x) を実装する。"""

    alpha: float = 0.5

    def __post_init__(self) -> None:
        if not 0.0 < self.alpha < 1.0:
            raise ValueError("alpha は 0<alpha<1 を満たす必要がある")

    @property
    def invariant_scale(self) -> float:
        """How: Cauchy 不変尺度 gamma=sqrt(alpha/(1-alpha)) を返す。"""
        return float(np.sqrt(self.alpha / (1.0 - self.alpha)))

    @property
    def theoretical_lyapunov(self) -> float:
        """How: 不変 Cauchy 測度下の Lyapunov 指数を返す。"""
        return float(2.0 * np.log(np.sqrt(self.alpha) + np.sqrt(1.0 - self.alpha)))

    def __call__(self, x: FloatArray) -> FloatArray:
        values = np.asarray(x, dtype=np.float64)
        # Why not: 状態を clip すると写像と重い裾を変更するため、厳密なゼロだけを machine-neighbour へ送る。
        safe = np.where(values == 0.0, np.nextafter(0.0, 1.0), values)
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            return self.alpha * (safe - 1.0 / safe)

    def derivative(self, x: FloatArray) -> FloatArray:
        values = np.asarray(x, dtype=np.float64)
        safe = np.where(values == 0.0, np.nextafter(0.0, 1.0), values)
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            return self.alpha * (1.0 + 1.0 / (safe * safe))


@dataclass(frozen=True)
class TangentMap:
    """How: T_beta(x)=tan(beta x) を周期還元して数値評価する。"""

    beta: float = 1.2

    def __post_init__(self) -> None:
        if self.beta <= 0.0:
            raise ValueError("beta は正である必要がある")

    def _reduced_argument(self, x: FloatArray) -> FloatArray:
        values = np.asarray(x, dtype=np.float64)
        # Why not: tan の出力を clip せず、周期性を使って入力角だけを主値域へ還元する。
        return np.remainder(self.beta * values + np.pi / 2.0, np.pi) - np.pi / 2.0

    def __call__(self, x: FloatArray) -> FloatArray:
        return np.tan(self._reduced_argument(x))

    def derivative(self, x: FloatArray) -> FloatArray:
        arg = self._reduced_argument(x)
        cos_arg = np.cos(arg)
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            return self.beta / (cos_arg * cos_arg)


@dataclass(frozen=True)
class OutputMixingNetwork:
    """How: 局所写像出力を平均場へ混合し、同期多様体を厳密に保つ。"""

    local_map: ScalarMap
    coupling: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.coupling <= 1.0:
            raise ValueError("coupling は [0,1] に入る必要がある")

    def step(self, states: FloatArray, common_input: FloatArray | float = 0.0) -> FloatArray:
        mapped = self.local_map(np.asarray(states, dtype=np.float64))
        mean_mapped = np.mean(mapped, axis=-1, keepdims=True)
        mixed = (1.0 - self.coupling) * mapped + self.coupling * mean_mapped
        return mixed + np.asarray(common_input, dtype=np.float64)

    def simulate(
        self,
        initial_states: FloatArray,
        n_steps: int,
        common_inputs: FloatArray | None = None,
        *,
        store_initial: bool = True,
    ) -> FloatArray:
        """How: batch×node 状態を固定結合で反復し、batch×time×node 軌道を返す。"""
        if n_steps <= 0:
            raise ValueError("n_steps は正である必要がある")
        state = np.asarray(initial_states, dtype=np.float64).copy()
        if state.ndim != 2:
            raise ValueError("initial_states は [batch,node] である必要がある")
        batch_size = state.shape[0]
        offset = 1 if store_initial else 0
        trajectory = np.empty((batch_size, n_steps + offset, state.shape[1]), dtype=np.float64)
        if store_initial:
            trajectory[:, 0, :] = state
        if common_inputs is None:
            inputs = np.zeros((batch_size, n_steps), dtype=np.float64)
        else:
            inputs = np.asarray(common_inputs, dtype=np.float64)
            if inputs.ndim == 1:
                inputs = np.broadcast_to(inputs[None, :], (batch_size, n_steps))
            if inputs.shape != (batch_size, n_steps):
                raise ValueError("common_inputs は [batch,time] または [time] である必要がある")
        for time_index in range(n_steps):
            state = self.step(state, inputs[:, time_index, None])
            trajectory[:, time_index + offset, :] = state
        return trajectory


def generate_boole_trajectories(
    alphas: FloatArray,
    seeds: NDArray[np.int64],
    *,
    burn_in: int,
    length: int,
) -> FloatArray:
    """How: 各 alpha・seed に対して一般化 Boole 軌道を独立生成する。"""
    alpha_values = np.asarray(alphas, dtype=np.float64)
    seed_values = np.asarray(seeds, dtype=np.int64)
    if alpha_values.shape != seed_values.shape:
        raise ValueError("alphas と seeds の形は一致する必要がある")
    if burn_in < 0 or length <= 0:
        raise ValueError("burn_in>=0, length>0 が必要")
    trajectories = np.empty((len(alpha_values), length), dtype=np.float64)
    for row, (alpha, seed) in enumerate(zip(alpha_values, seed_values, strict=True)):
        local_map = BooleMap(float(alpha))
        rng = np.random.default_rng(int(seed))
        state = local_map.invariant_scale * rng.standard_cauchy()
        for _ in range(burn_in):
            state = float(local_map(np.asarray(state)))
        for time_index in range(length):
            state = float(local_map(np.asarray(state)))
            trajectories[row, time_index] = state
    return trajectories


def simulate_tangent_pair(
    beta: float,
    coupling: float,
    initial_x: FloatArray,
    initial_y: FloatArray,
    *,
    n_steps: int,
    coupling_form: str,
    tolerance: float = 1e-8,
    sustain: int = 50,
    divergence_limit: float = 1e12,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """How: 2自由度タンジェント系を並列反復し、同期・発散・終端誤差を返す。"""
    tangent = TangentMap(beta)
    x = np.asarray(initial_x, dtype=np.float64).copy()
    y = np.asarray(initial_y, dtype=np.float64).copy()
    if x.shape != y.shape:
        raise ValueError("initial_x と initial_y の形は一致する必要がある")
    sustained = np.ones(x.shape, dtype=bool)
    diverged = np.zeros(x.shape, dtype=bool)
    recent_errors = np.full((sustain, x.size), np.inf, dtype=np.float64)
    for time_index in range(n_steps):
        fx = tangent(x)
        fy = tangent(y)
        if coupling_form == "output_cross":
            next_x = (1.0 - coupling) * fx + coupling * fy
            next_y = coupling * fx + (1.0 - coupling) * fy
        elif coupling_form == "state_diffusive":
            next_x = fx + coupling * (y - x)
            next_y = fy + coupling * (x - y)
        else:
            raise ValueError(f"未知の coupling_form: {coupling_form}")
        x, y = next_x, next_y
        invalid = (~np.isfinite(x)) | (~np.isfinite(y)) | (np.abs(x) > divergence_limit) | (np.abs(y) > divergence_limit)
        diverged |= invalid
        x = np.where(invalid, 0.0, x)
        y = np.where(invalid, 0.0, y)
        recent_errors[time_index % sustain, :] = np.abs(x - y)
    final_error = np.max(recent_errors, axis=0)
    sustained &= final_error < tolerance
    sustained &= ~diverged
    return sustained, diverged, final_error
