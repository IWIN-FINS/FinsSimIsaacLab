"""Convert the checked-in FinsROV FBX to USD with Isaac Sim's asset converter.

Run with ``./isaaclab.sh -p <this-file>`` from the upstream IsaacLab checkout.
The generated USD remains in the private project repository; Unity is untouched.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

async def main() -> None:
    import omni.kit.app
    from pxr import Usd, UsdPhysics

    extension_manager = omni.kit.app.get_app().get_extension_manager()
    extension_manager.set_extension_enabled_immediate("omni.kit.asset_converter", True)
    # The extension owns the Python module and must be enabled after Kit starts.
    import omni.kit.asset_converter

    root = Path(__file__).resolve().parents[1]
    source = root / "source/finssim_isaaclab_tasks/finssim_isaaclab_tasks/assets/finsrov/source/FinsROV_Model.fbx"
    output = root / "source/finssim_isaaclab_tasks/finssim_isaaclab_tasks/assets/finsrov/data/finsrov.usd"
    output.parent.mkdir(parents=True, exist_ok=True)
    context = omni.kit.asset_converter.AssetConverterContext()
    context.keep_all_materials = True
    context.merge_all_meshes = False
    context.baking_scales = True
    task = omni.kit.asset_converter.get_instance().create_converter_task(str(source), str(output), None, context)
    if not await task.wait_until_finished():
        raise RuntimeError(f"FinsROV FBX conversion failed: {source}")

    # The generic FBX converter only authors render geometry.  Isaac Lab's
    # RigidObject requires exactly one RigidBodyAPI prim below the spawn path,
    # so make the converted default prim the vehicle's physical body.  Keep
    # mass in the USD as a useful standalone default; task configs may still
    # override it through MassPropertiesCfg.
    stage = Usd.Stage.Open(str(output))
    if stage is None:
        raise RuntimeError(f"Could not open converted FinsROV USD: {output}")
    body_prim = stage.GetDefaultPrim()
    if not body_prim:
        raise RuntimeError(f"Converted FinsROV USD has no default prim: {output}")
    UsdPhysics.RigidBodyAPI.Apply(body_prim).CreateRigidBodyEnabledAttr(True)
    UsdPhysics.MassAPI.Apply(body_prim).CreateMassAttr(12.11)
    stage.GetRootLayer().Save()
    print(f"Converted {source} -> {output}")


asyncio.get_event_loop().run_until_complete(main())
simulation_app.close()
