import logging

import numpy as np

from wpc_rsb_core import RSBModel


def synthetic_wpc(wind_speed: np.ndarray) -> np.ndarray:
    power = np.zeros_like(wind_speed, dtype=float)
    ramp = (wind_speed >= 3.0) & (wind_speed <= 12.0)
    plateau = wind_speed > 12.0
    power[ramp] = ((wind_speed[ramp] - 3.0) / 9.0) ** 3 * 1500.0
    power[plateau] = 1500.0
    return power


def main() -> None:
    logging.getLogger().setLevel(logging.WARNING)
    rng = np.random.default_rng(42)
    wind = rng.uniform(0.0, 25.0, 4000)
    power = synthetic_wpc(wind)
    noisy_power = np.clip(power + rng.normal(0.0, 60.0, size=power.shape), 0.0, None)

    outlier_mask = rng.random(wind.shape[0]) < 0.03
    noisy_power[outlier_mask] *= rng.uniform(0.1, 1.4, size=outlier_mask.sum())

    model = RSBModel()
    model.train(wind, noisy_power)

    grid = np.linspace(0.0, 25.0, 200)
    pred = model.predict(grid)

    print(f"trained_points={wind.size}")
    print(f"pred_grid_points={grid.size}")
    print(f"pred_range_kw=({pred.min():.2f}, {pred.max():.2f})")


if __name__ == "__main__":
    main()
