# Third-party notices

This repository is not uniformly Apache-2.0. The following components retain
their own copyright notices and licenses. This file is informational and does
not replace the license distributed with any component.

| Component | Location | Provenance and license | Redistribution note |
| --- | --- | --- | --- |
| WarpAUV / `isaac-auv-env` reference | `third_party/isaac-auv-env/` | Copied from [`warplab/isaac-auv-env`](https://github.com/warplab/isaac-auv-env) commit [`7c5ebe7`](https://github.com/warplab/isaac-auv-env/tree/7c5ebe7f7a08acd2570b5fba328e92b7f59f6794). The exact upstream `LICENSE` is retained locally and is BSD-3-Clause, not MIT. | Preserve its copyright, BSD-3-Clause terms, and disclaimer in source and binary redistributions. |
| WarpAUV-derived Lab 3 port | `source/finssim_isaaclab_tasks/.../assets/warpauv/`, `.../tasks/direct/warpauv/`, and related compatibility code | The project describes these files as an adaptation of the pinned WarpAUV reference, but does not yet carry file-level origin headers. | Treat as BSD-3-Clause-derived until a file-level provenance review establishes independent authorship; preserve upstream notices for copied/modified files. |
| WarpAUV USD and generated instanceable meshes | `source/finssim_isaaclab_tasks/finssim_isaaclab_tasks/assets/warpauv/data/` | Imported from the same pinned WarpAUV reference during the Lab 3 adaptation. | Treat as BSD-3-Clause upstream material unless a more specific embedded-asset notice is discovered. |
| Historical WarpAUV checkpoint | `runtime/legacy_checkpoints/warpauv_v2_poshold_dr_2024-09-13.pt` | Retained for regression playback and recorded as copied from the pinned WarpAUV source. The included source license is not an explicit model-weight grant. | Do not publish or redistribute this artifact until its model-weight rights are confirmed in writing; it is excluded from the Apache-2.0 claim. |
| Isaac Lab | Installed separately; imported by the extension | Isaac Lab is BSD-3-Clause. Its dependency and asset notices are supplied by the selected upstream distribution. | This repository does not vendor Isaac Lab. Follow the notice bundle for the exact installed version. |
| Isaac Sim / Omniverse Kit | Installed separately | NVIDIA proprietary/additional terms apply to portions of the Isaac Sim distribution. | This repository neither redistributes nor sublicenses Isaac Sim, Kit, their extensions, or NVIDIA assets. |
| Python dependencies | Installed from package indexes | `numpy`, `torch`, `psutil`, `skrl`, RSL-RL, and transitive dependencies have independent terms. | A binary/container distribution must generate and include a dependency SBOM and notices for the exact lockfiles/environment shipped. |

## FinsROV assets are not a third-party clearance

`assets/finsrov/source/*.fbx` and the generated `*.usd` files originate from
the FinsSim Unity project, as recorded in `asset_manifest.json`. They may be
released as FinsSim-owned material only after the copyright owner confirms that
all meshes, textures, materials, and any source-asset terms permit the intended
public redistribution. Until that confirmation, they are excluded from the
repository-level Apache-2.0 grant; see the audit for the release gate.

## Source modifications

The FinsSim Lab 3 task code is maintained separately from the reference
checkout and does not import Python modules from it at runtime. That separation
does not erase the BSD obligations for copied or derived code/assets, nor does
it grant rights to independently imported third-party materials.
