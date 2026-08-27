from .kappa_lib import solve_kappa, solve_kappa_path, SpectrumKappa
from .ridge_theory_lib import (
    compute_kappa_from_lambda,
    compute_df2,
    ridge_error_per_pc_theory,
    ridge_error_total_theory,
    generalization_r2_theory,
    accentuation_alignment,
    accentuation_ratio_moments_theory,
    accentuation_error_theory,
    accentuation_r2_theory,
    peer_review_theory,
    peer_review_r2_theory,
    peer_review_error_theory,
)
from .simulation_lib import (
    sample_gaussian_data,
    ridge_estimator,
    ridge_path_estimators,
    ridge_cross_validated_estimator,
    ridge_error_per_pc_simulation,
    generalization_metrics_simulation,
    accentuation_alignment_simulation,
    accentuation_error_simulation,
    accentuation_r2_simulation,
    peer_review_metrics_simulation,
    run_monte_carlo,
    run_paired_monte_carlo,
    run_paired_cv_monte_carlo,
)
from .spectrum_lib import (
    isotropic_spectrum,
    powerlaw_spectrum,
    vanhateren_spectrum,
    get_spectrum,
)
from .teacher_lib import make_spectral_teacher
from .feature_theory_lib import (
    spectral_feature_de_metrics,
    select_spectral_feature_de_alpha,
)
