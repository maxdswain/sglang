# SPDX-License-Identifier: Apache-2.0
"""TRELLIS.2 flow-matching Euler sampler.

Implements the checkpoint's FlowEulerGuidanceIntervalSampler, including its
velocity prediction, timestep rescaling, CFG interval and x0-variance rescale.
Works with dense tensors and with sparse tensors implementing tensor arithmetic,
``replace``, ``std`` and the ``shape``/``device`` properties. Sparse operators and
model implementations are separate milestones of the native port.

Reference: https://github.com/microsoft/TRELLIS.2/blob/main/trellis2/pipelines/samplers/
"""

from collections.abc import Callable
from typing import Any

import torch


class Trellis2FlowEulerSampler:
    def __init__(self, sigma_min: float = 1e-5):
        if not 0 <= sigma_min < 1:
            raise ValueError("sigma_min must be in [0, 1)")
        self.sigma_min = sigma_min

    def _predict(
        self,
        model: Callable[..., Any],
        x_t: Any,
        t: float,
        cond: Any,
        neg_cond: Any,
        guidance_strength: float,
        guidance_rescale: float,
        guidance_interval: tuple[float, float],
        **model_kwargs: Any,
    ) -> Any:
        timestep = torch.full(
            (x_t.shape[0],), 1000 * t, device=x_t.device, dtype=torch.float32
        )
        if not guidance_interval[0] <= t <= guidance_interval[1]:
            guidance_strength = 1.0

        if guidance_strength == 1:
            return model(x_t, timestep, cond, **model_kwargs)
        if guidance_strength == 0:
            return model(x_t, timestep, neg_cond, **model_kwargs)

        positive = model(x_t, timestep, cond, **model_kwargs)
        negative = model(x_t, timestep, neg_cond, **model_kwargs)
        prediction = guidance_strength * positive + (1 - guidance_strength) * negative
        if guidance_rescale > 0:
            # Rescale *x0*, not velocity: this is the TRELLIS.2 CFG convention.
            factor = self.sigma_min + (1 - self.sigma_min) * t
            x0_positive = (1 - self.sigma_min) * x_t - factor * positive
            x0_cfg = (1 - self.sigma_min) * x_t - factor * prediction
            dims = tuple(range(1, len(x_t.shape)))
            std_positive = x0_positive.std(dim=dims, keepdim=True)
            std_cfg = x0_cfg.std(dim=dims, keepdim=True)
            x0_rescaled = x0_cfg * (std_positive / std_cfg)
            x0 = guidance_rescale * x0_rescaled + (1 - guidance_rescale) * x0_cfg
            prediction = ((1 - self.sigma_min) * x_t - x0) / factor
        return prediction

    @torch.no_grad()
    def sample(
        self,
        model: Callable[..., Any],
        noise: Any,
        cond: Any,
        neg_cond: Any,
        *,
        steps: int = 12,
        rescale_t: float = 1.0,
        guidance_strength: float = 7.5,
        guidance_rescale: float = 0.0,
        guidance_interval: tuple[float, float] = (0.0, 1.0),
        **model_kwargs: Any,
    ) -> Any:
        if steps <= 0:
            raise ValueError("steps must be positive")
        if rescale_t <= 0:
            raise ValueError("rescale_t must be positive")
        if not 0 <= guidance_interval[0] <= guidance_interval[1] <= 1:
            raise ValueError("guidance_interval must lie within [0, 1]")
        if not 0 <= guidance_rescale <= 1:
            raise ValueError("guidance_rescale must be in [0, 1]")

        sample = noise
        # Python floats preserve the upstream numpy.linspace schedule semantics.
        for i in range(steps):
            t = 1 - i / steps
            t_prev = 1 - (i + 1) / steps
            t = rescale_t * t / (1 + (rescale_t - 1) * t)
            t_prev = rescale_t * t_prev / (1 + (rescale_t - 1) * t_prev)
            velocity = self._predict(
                model,
                sample,
                t,
                cond,
                neg_cond,
                guidance_strength,
                guidance_rescale,
                guidance_interval,
                **model_kwargs,
            )
            sample = sample - (t - t_prev) * velocity
        return sample
