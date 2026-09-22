"""FinsSim task registrations for Isaac Lab."""

from .tasks.direct import finsrov as _finsrov  # noqa: F401
from .tasks.direct import finsrov_hold_for_position as _finsrov_hold_for_position  # noqa: F401
from .tasks.direct import finsrov_hold_for_position_wrench as _finsrov_hold_for_position_wrench  # noqa: F401
from .tasks.direct import finsrov_t3 as _finsrov_t3  # noqa: F401
from .tasks.direct import finsrov_trajectory_tracking as _finsrov_trajectory_tracking  # noqa: F401
from .tasks.direct import warpauv as _warpauv  # noqa: F401
from .tasks.manager_based import pose_hold as _pose_hold  # noqa: F401
