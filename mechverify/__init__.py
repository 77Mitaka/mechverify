"""mechverify package."""

from .ir import Mechanism, Link, Joint, CouplerPoint
from .kinematics import FourBarGeometry, SweepResult, four_bar_from_mechanism
from .structural import structural_checks, mobility, grashof
from .requirement import evaluate, path_straightness, path_deviation
from .verifier import verify, VerifyResult
from .modelica import generate_modelica, write_model, omc_available, simulate
from .planar_mechanics import generate_planar_mechanics
from .convert import modelica_to_mechanism
from .robustness import verify_robust, RobustResult
from .corpus import Corpus

__version__ = "0.4.0"
