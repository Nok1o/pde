"""Run-local trial conductivity, reused only between accepted physical layers.

Only k is delayed. Time data, capacity and reaction are rebuilt normally.
Acceptance always uses a fresh operator; a rejected trial refreshes k before
the next update. This is not a cache of the nonlinear residual or MC samples.
"""


class ConductivityCache3D:
    def __init__(self, max_layers):
        self.max_layers = max_layers
        self.values = None
        self.layer = None

    def can_reuse(self, layer):
        return self.values is not None and layer - self.layer < self.max_layers

    def refresh(self, problem, iterate, layer):
        self.values = problem.conductivity(iterate)
        self.values.setflags(write=False)
        self.layer = layer
