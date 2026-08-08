"""Phase 5 — hybrid fusion of the tabular weather model and the satellite vision model.

The OpenMP setup below runs at package import time and must stay before
any numeric library is imported. torch, xgboost and scikit-learn each
vendor their own ``libomp``; this pipeline drives torch (Phase 4 scene
scoring) and xgboost (fusion candidate) in one process, and on macOS the
two runtimes collide — the process either segfaults inside DMatrix
construction or deadlocks, whichever library initialises second.
Constraining OpenMP to a single thread makes both runtimes coexist.

The fusion dataset is ~10^3 rows, so single-threaded training costs
nothing measurable here. See docs/hybrid_ai_architecture.md.
"""

import logging
import os

_logger = logging.getLogger(__name__)

_configured = os.environ.get("OMP_NUM_THREADS")
if _configured is None:
    os.environ["OMP_NUM_THREADS"] = "1"
elif _configured != "1":
    _logger.warning(
        "OMP_NUM_THREADS=%s is set; the fusion pipeline mixes torch and xgboost, which is "
        "known to crash on macOS unless OpenMP is limited to one thread.",
        _configured,
    )
