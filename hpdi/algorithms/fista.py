"""FISTA for 0.5 * ||A x - y||_2^2 + lambda * ||x||_1.

The matrix transpose and exact Lipschitz constant are reused across images and
channels. Iteration order matches the original commented HPDI_train.py routine.
"""

import math

import torch


class FISTASolver:
    """Reuse one fixed real measurement matrix for independent signal solves."""

    @torch.no_grad()
    def __init__(self, matrix, lipschitz=None):
        if not isinstance(matrix, torch.Tensor) or matrix.ndim != 2:
            raise ValueError("Measurement matrix must be a 2-D tensor")
        if not matrix.is_floating_point() or not torch.isfinite(matrix).all():
            raise ValueError("Measurement matrix must contain finite floating-point values")
        if not matrix.shape[0] or not matrix.shape[1]:
            raise ValueError("Measurement matrix cannot be empty")
        # Own a fixed copy so external in-place edits cannot invalidate the cache.
        self.matrix = matrix.detach().clone()
        self.transpose = self.matrix.T
        if lipschitz is None:
            self.lipschitz = torch.linalg.norm(self.transpose @ self.matrix, ord=2)
        else:
            self.lipschitz = torch.as_tensor(lipschitz, device=matrix.device, dtype=matrix.dtype)
        if self.lipschitz.ndim != 0 or not torch.isfinite(self.lipschitz) or self.lipschitz <= 0:
            raise ValueError("Lipschitz constant must be a finite positive scalar")
        self.zero = self.matrix.new_tensor(0.0)

    @torch.no_grad()
    def solve(self, measurement, lambd=1e-4, max_iter=100, tol=1e-6, stopping_rule="legacy"):
        """Solve one signal, retaining legacy early-return semantics by default.

        The original routine tests convergence before assigning x_new to x, so
        it returns the preceding iterate on early termination. Use 'latest' to
        explicitly select the conventional new-iterate return value instead.
        """
        if not math.isfinite(lambd) or lambd < 0:
            raise ValueError("lambda must be finite and nonnegative")
        if not isinstance(max_iter, int) or isinstance(max_iter, bool) or max_iter < 1:
            raise ValueError("max_iter must be a positive integer")
        if not math.isfinite(tol) or tol < 0:
            raise ValueError("tol must be finite and nonnegative")
        if stopping_rule not in {"legacy", "latest"}:
            raise ValueError("stopping_rule must be legacy or latest")
        y = torch.as_tensor(measurement, device=self.matrix.device, dtype=self.matrix.dtype)
        if y.ndim != 1 or y.shape[0] != self.matrix.shape[0]:
            raise ValueError(
                "Measurement must be a vector of length {}".format(self.matrix.shape[0])
            )
        if not torch.isfinite(y).all():
            raise ValueError("Measurement must contain finite values")
        x = self.matrix.new_zeros(self.matrix.shape[1])
        z = x.clone()
        t = 1
        threshold = lambd / self.lipschitz
        for _ in range(max_iter):
            grad = self.transpose @ (self.matrix @ z - y)
            x_new = z - grad / self.lipschitz
            x_new = torch.sign(x_new) * torch.maximum(torch.abs(x_new) - threshold, self.zero)
            t_new = (1 + (1 + 4 * t**2) ** 0.5) / 2
            z = x_new + (t - 1) / t_new * (x_new - x)
            if torch.linalg.norm(x_new - x) < tol:
                return x if stopping_rule == "legacy" else x_new
            x, t = x_new, t_new
        return x

    def solve_images(self, measurements, height=32, width=32, **kwargs):
        """Solve every sample/channel separately, preserving its stopping test."""
        if measurements.ndim != 4:
            raise ValueError("Expected measurements with shape (batch, channels, height, width)")
        if height < 1 or width < 1 or height * width != self.matrix.shape[1]:
            raise ValueError("Output height * width must equal the matrix column count")
        images = []
        for sample in measurements:
            channels = [
                self.solve(channel.reshape(-1), **kwargs).reshape(height, width)
                for channel in sample
            ]
            images.append(torch.stack(channels))
        return torch.stack(images)


def FISTA(A, y, lambd, max_iter=100, tol=1e-6):
    """Original callable signature; reuse FISTASolver for multiple signals."""
    return FISTASolver(A).solve(y, lambd=lambd, max_iter=max_iter, tol=tol)
