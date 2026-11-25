"""Shared face options for complete heat runs and frozen-system diagnostics."""

from .boundary3d import FACES


def add_boundary_arguments(parser):
    parser.add_argument("--beta", type=float, default=1.0, help="Общий beta для 3 рода")
    for face in FACES:
        parser.add_argument(
            "--" + face,
            type=int,
            choices=(1, 3),
            default=1,
            help=f"Род условия на грани {face}",
        )
        parser.add_argument(
            "--beta-" + face,
            type=float,
            default=None,
            help=f"beta грани {face}; по умолчанию общий --beta",
        )


def boundary_configuration(args):
    return dict(
        kinds=tuple(getattr(args, face) for face in FACES),
        betas=tuple(
            (
                args.beta
                if getattr(args, "beta_" + face) is None
                else getattr(args, "beta_" + face)
            )
            for face in FACES
        ),
    )
