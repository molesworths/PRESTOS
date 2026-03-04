"""Fixed transport model for testing."""

from typing import Dict

import numpy as np

from .TransportBase import TransportBase


class Fixed(TransportBase):
    """Fixed diffusivity/conductivity model for testing."""

    def __init__(self, D: float = 1.0, chi: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.external = False  # No external dependencies
        
        self.D = D
        self.chi = chi

    def _evaluate_single(self, state) -> Dict[str, np.ndarray]:
        """Compute fluxes using fixed diffusivities and store on state.transport."""
        x = getattr(state, "roa", state.r / state.a)
        n_roa = len(x)
        n_species = np.asarray(state.ni).shape[1] if np.asarray(state.ni).ndim == 2 else 1

        dne_dr = np.gradient(state.ne, state.r)
        dte_dr = np.gradient(state.te, state.r)
        dti_dr = np.gradient(state.ti, state.r)

        Ge = -self.D * dne_dr
        Gi = np.zeros((n_roa, n_species))
        for i in range(n_species):
            dni_dr = np.gradient(
                state.ni[:, i] if np.asarray(state.ni).ndim == 2 else state.ni,
                state.r,
            )
            Gi[:, i] = -self.D * dni_dr

        Qe = -self.chi * state.ne * dte_dr * 1.6e-3

        Qi = np.zeros((n_roa, n_species))
        for i in range(n_species):
            ni_i = state.ni[:, i] if np.asarray(state.ni).ndim == 2 else state.ni
            Qi[:, i] = -self.chi * ni_i * dti_dr * 1.6e-3

        if not hasattr(state, "transport"):
            class TransportContainer:
                pass

            state.transport = TransportContainer()
        tr = state.transport
        self.model = "Fixed"
        self.Ge = Ge
        self.Gi = Gi
        self.Qe = Qe
        self.Qi = Qi

        R0 = float(getattr(state, "R0", getattr(state, "Rmaj", 1.0)))
        a = float(getattr(state, "a", 1.0))
        kappa = np.asarray(getattr(state, "kappa", np.ones_like(x)))
        aspect_ratio = R0 / max(a, 1e-9)
        dVdx = (2 * np.pi * aspect_ratio) * (2 * np.pi * x * np.sqrt((1 + kappa**2) / 2))
        surfArea = dVdx * a**2
        A_edge = float(surfArea[-1])

        P_e = float(np.asarray(Qe)[-1]) * A_edge
        Qi_sum = np.asarray(Qi)
        if Qi_sum.ndim == 2:
            Qi_edge = float(np.sum(Qi_sum[-1, :]))
        else:
            Qi_edge = float(Qi_sum[-1])
        P_i = Qi_edge * A_edge

        self.labels = ["Pe", "Pi"]
        return {"Pe": np.atleast_1d(P_e), "Pi": np.atleast_1d(P_i)}
