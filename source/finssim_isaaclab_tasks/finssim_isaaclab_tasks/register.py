"""Hook consumed by Isaac Lab's external callback option."""


def register_environments():
    import finssim_isaaclab_tasks  # noqa: F401

    return None

