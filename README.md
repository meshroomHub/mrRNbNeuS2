<div align="center">

# mrRNbNeuS2

### Meshroom Plugin for RNb-NeuS2

<p>
Integrate <a href="https://github.com/meshroomHubWarehouse/RNb-NeuS2">RNb-NeuS2</a> neural surface reconstruction from multi-view normal and reflectance maps directly into your <a href="https://github.com/alicevision/Meshroom">Meshroom</a> photogrammetry pipeline.
</p>

<a href="https://github.com/meshroomHubWarehouse/RNb-NeuS2"><img src="https://img.shields.io/badge/Core-RNb--NeuS2-green" alt="RNb-NeuS2" height="25"></a>

</div>

---

## Overview

A single node is provided:

**`RNbNeuS2`** (category *Neural Reconstruction*): neural surface reconstruction
from multi-view normal and reflectance maps. It runs the **entire** pipeline
internally: it reads AliceVision SfMData, converts them to the testbed format,
normalizes the scene into the unit sphere, trains the CUDA testbed (two-stage, with
optional multi-view albedo scaling), and exports the mesh in world coordinates.
No separate data-preparation node is needed. With albedo maps, it also exports them
scaled as for training (harmonized), ready to texture the mesh with the AliceVision
`Texturing` node (see [Texturing the mesh with the albedo maps](#texturing-the-mesh-with-the-albedo-maps)).

This repository holds the Meshroom side only. The reconstruction code (the `rnb_neus2`
Python package and the CUDA `testbed`) lives in
[RNb-NeuS2](https://github.com/meshroomHubWarehouse/RNb-NeuS2) (fork of
[RobinBruneau/RNb-NeuS2](https://github.com/RobinBruneau/RNb-NeuS2)).

## Requirements

- **[Meshroom](https://github.com/alicevision/Meshroom)** 2025+ (develop branch)
- **NVIDIA GPU** with a driver supporting CUDA 12.4 (prebuilt testbed for compute
  capabilities 7.5, 8.0, 8.6 and 8.9)
- **Python** 3.11 for the prebuilt testbed archive (the Python of Meshroom)

## Installation

### 1. Install the plugin

From Meshroom's plugin manager (*Manage Plugins*), or from the command line:

```bash
meshroom_plugins install mrRNbNeuS2
```

The installer creates the plugin `venv/` with `uv sync`; it installs the `rnb_neus2`
package pinned in `pyproject.toml`.

### 2. Install the RNb-NeuS2 testbed

The compiled testbed is not installed by the plugin manager yet. Download the
archive matching your OS from the
[RNb-NeuS2 v2.0.1 release](https://github.com/RobinBruneau/RNb-NeuS2/releases/tag/v2.0.1),
check its SHA-256 against the one shown on the release page, and extract it into the
plugin folder:

```bash
cd /path/to/plugins/mrRNbNeuS2
sha256sum RNb-NeuS2-2.0.1-linux-x86_64-cu124-py311.tar.gz
tar xzf RNb-NeuS2-2.0.1-linux-x86_64-cu124-py311.tar.gz   # creates RNb-NeuS2/
```

On Windows, extract `RNb-NeuS2-2.0.1-windows-x86_64-cu124-py311.zip` the same way.
The node then finds `RNb-NeuS2/build/testbed` (default of `RNB_NEUS2_TESTBED_PATH`
in `pyproject.toml`, relative to the plugin root). Only `build/`, `configs/` and
`utils/` of the archive are used: the testbed reads its MLP weights from
`../utils/` next to `build/`.

To keep the testbed elsewhere (e.g. a local build), set the environment variable
before launching Meshroom; it overrides the plugin default:

```bash
export RNB_NEUS2_TESTBED_PATH=/path/to/RNb-NeuS2/build/testbed
```

### Manual install (without the plugin manager)

```bash
cd /path/to/plugins
git clone https://github.com/meshroomHub/mrRNbNeuS2.git
cd mrRNbNeuS2
UV_PROJECT_ENVIRONMENT=venv uv sync   # Meshroom looks for venv/ at the plugin root
# then step 2 above
export MESHROOM_PLUGINS_PATH=/path/to/plugins/mrRNbNeuS2
```

## Node: `RNbNeuS2`

### Inputs

| Parameter | Label | Default | Description |
|-----------|-------|---------|-------------|
| `inputNormalSfm` | Normal Maps SfMData | — | SfMData pointing to the normal-map images. **Required.** |
| `inputAlbedoSfm` | Albedo Maps SfMData | — | SfMData pointing to albedo images. If set, enables two-phase training with albedo scaling. |
| `inputMaskSfm` | Mask SfMData | — | SfMData pointing to mask images. |
| `inputMaskFolder` | Mask Folder | — | Folder of masks named by `viewId` (e.g. `12345.png`). Ignored when Mask SfMData is provided. |
| `maxSteps` | Max Training Steps | 15000 | Total iterations for stage 2 (stage 1 uses 2/3 of this). |
| `meshResolution` | Mesh Resolution | 1024 | Marching cubes resolution for the final mesh. |
| `scalingMode` | Scaling Mode | `auto` | Scene normalization: `auto` prefers silhouettes when masks exist, then falls back to landmarks (`pcd`) or camera centers. One of `auto`, `pcd`, `silhouettes` (minimum sphere enclosing the silhouettes), `cameras`, `none`. |
| `silhouetteMargin` | Silhouette Margin | `20` | Margin (pixels) between the silhouettes and the projection of the bounding sphere (`silhouettes` scaling). Advanced. |
| `sphereScale` | Sphere Scale | 1.0 | Target scale within the unit sphere after normalization. |
| `warmupRatio` | Phase 1 Ratio | 0.1 | Fraction of `maxSteps` for the geometry-only warmup (albedo mode only). |
| `maskWeight` | Mask Weight | 1.0 | Weight of the mask loss. |
| `superNormal` | SuperNormal | false | Enable SuperNormal mode (MVPS method). |
| `useL1` | L1 Norm | false | Use L1 color loss instead of L2. |
| `useRgbPlus` | RGB+ | true | Enable the RGB+ reflectance-singularity correction. |
| `useGpu` | Use GPU | true | Use GPU for training (CUDA required). |
| `rnbNeuS2Path` | RNb-NeuS2 Testbed Path | `${RNB_NEUS2_TESTBED_PATH}` | Path to the testbed executable (advanced; from `pyproject.toml` or the environment). |
| `verboseLevel` | Verbose Level | `info` | Logging verbosity (`fatal`, `error`, `warning`, `info`, `debug`, `trace`). |

### Outputs

| Parameter | Label | Description |
|-----------|-------|-------------|
| `outputFolder` | Output Folder | Folder with all training artifacts (`nodeCacheFolder`). |
| `outputMesh` | Output Mesh | Reconstructed `mesh.obj` in world coordinates (viewable in Meshroom). |
| `outputAlbedoFolder` | Harmonized Albedo Folder | `albedo/<viewId>.exr`: the input albedo maps multiplied by the per-view scales used for training, as linear float EXR (not clipped), extended 8 px outside the masks. Written only when `inputAlbedoSfm` is set (the attribute is disabled otherwise). |
| `outputAlbedoSfm` | Harmonized Albedo SfMData | `albedoHarmonized.sfm`: copy of the Albedo Maps SfMData (JSON) whose views point to these maps (views without a map are dropped). Written only when `inputAlbedoSfm` is set. |

The outputs do not take part in the node UID: projects computed with a previous
version keep their cache (the harmonized albedo outputs are then missing until the
node is recomputed).

## Albedo scaling

Photometric-stereo albedo maps are relative: each view is known up to a per-channel
scale (e.g. the light powers of each pose). When an Albedo Maps SfMData is given,
the node trains a geometry-only phase 1, then estimates one scale per view and
channel on the phase-1 mesh (`rnb_neus2/albedo_scaling.py` in RNb-NeuS2,
`compute_albedo_scale_ratios`), rescales the albedos and trains phase 3 with them:

- 20000 masked pixels per view are ray-cast onto the mesh; each hit is compared with
  **every other view** that sees it (in front, inside the image, facing it with a
  viewing angle <= 60 deg, unoccluded, inside its mask, valid albedos in both views);
- every view pair with >= 300 shared samples gives one equation
  `log s_i - log s_j = -median(log a_i - log a_j)`, weighted by `sqrt(n_samples)`;
- the equations are solved jointly (least squares): the result does not depend on the
  order of the views and loops of views are closed; the scales are normalized to an
  arithmetic mean of 1 (same albedo level as before the scaling). Views without any
  overlap keep the scale 1 (warning in the log); separate groups of overlapping views
  are normalized independently (warning).

The log reports the median pairwise albedo mismatch before/after the scaling and, per
view, the scale and the fraction of masked pixels clipped in the 16-bit training PNGs
(the testbed reads the albedos in [0, 1], and with RGB+ its loss depends on the
absolute albedo level, so the level is kept and values above 1 are clipped; the
exported EXR maps are not clipped). `prepared_data/albedo_scales.json` keeps the
scales and the pair statistics, `prepared_data/frames.json` the mapping between the
testbed images (`albedos/00003.png`) and the input views (`pose_id`,
`albedo_view_id`, source paths).

On a 12-pose LED-getter turntable capture (biface), the previous scaling, which
chained each view to the next one in the SfMData (viewId) order, compared views on
opposite sides of the object and left a view 1.5x too bright with respect to the views
seeing the same surface; the joint solve brings the median pairwise mismatch from
22% (raw) to 5-6% (vs 12% with the chain).

## Texturing the mesh with the albedo maps

```
LedGetterPS ─Normal/Albedo SfMData, Mask Folder─► RNbNeuS2 ─Harmonized Albedo SfMData─► Texturing.input
                                                           ─Harmonized Albedo Folder──► Texturing.imagesFolder
                                                           ─Output Mesh───────────────► Texturing.inputMesh
```

The scales of the texture are those of the training. `Texturing` settings for a mesh
that does not come from `Meshing` and an SfMData without landmarks:

| Texturing parameter | Value | Why |
|---------------------|-------|-----|
| `input` | `RNbNeuS2` *Harmonized Albedo SfMData* | Cameras of the albedo maps. |
| `imagesFolder` | `RNbNeuS2` *Harmonized Albedo Folder* | Must be **non-empty**: the *Color Mapping* group is disabled otherwise, `--colorMappingFileType` is not passed and `aliceVision_texturing` writes no texture. Files are matched by `<viewId>`. |
| `inputMesh` | `RNbNeuS2` *Output Mesh* | No transform needed (see below). |
| `visibilityRemappingMethod` | `MeshItself` | The default `PullPush` remaps the visibilities of the SfM landmarks and stops with *"Cannot remap visibilities as there is no reference points"*. `MeshItself` ray-casts the visibility of every vertex. |
| `subdivisionTargetRatio` | `0` | The subdivision targets the (empty) landmark cloud: with the default 0.8 the binary writes its outputs, then aborts (*corrupted double-linked list*, exit 134) and the node fails. The RNbNeuS2 mesh is dense anyway. |
| `workingColorSpace` | `Linear` | Blend the linear albedo in linear space. |
| `outputColorSpace` | `AUTO` | EXR texture: linear albedo; PNG texture: sRGB-encoded, directly viewable. |
| `colorMapping.colorMappingFileType` | `exr` (data) or `png` (preview) | Use two `Texturing` nodes for both. |
| `correctEV` | off | The albedo maps are not exposures. |
| `unwrapMethod` | `Basic` | LSCM/ABF are limited to 600k/300k faces. |
| `textureSide` / `downscale` | 4096 / 1 | With half-resolution albedo maps, `Basic` fills ~25% of a 4096 atlas (one tile); 2048 spills over several UDIM tiles. |

Why EXR: photometric-stereo nodes write 16-bit PNG albedos with **linear** values,
but AliceVision reads any untagged 8/16-bit PNG as sRGB. The harmonized EXR maps are
read as linear data, so the texture color space is right.

The mesh needs no transform: `RNbNeuS2` exports its mesh with y and z negated with
respect to the SfMData world, which is the AliceVision mesh-file convention
(`Meshing`, `MeshFiltering`, `Texturing`), so `Texturing` places it correctly with the
SfMData cameras.

## Usage in Meshroom

1. Produce per-view normal maps (and optionally albedo/reflectance maps) and bring
   them into the graph as SfMData.
2. Add an **`RNbNeuS2`** node and connect:
   - `Normal Maps SfMData` ← your normals SfMData (required),
   - `Albedo Maps SfMData` ← albedo SfMData (optional, for reflectance-based training),
   - `Mask SfMData` *or* `Mask Folder` ← masks (optional).
3. Compute the node. The reconstructed mesh is exposed on `Output Mesh`; with
   albedos, the harmonized albedo maps on `Harmonized Albedo Folder` /
   `Harmonized Albedo SfMData` (to texture the mesh, see above).

## Command-line equivalent

The node is a thin wrapper around the standalone RNb-NeuS2 pipeline
(`rnb-neus2` script of the plugin venv). The same reconstruction can be launched
without Meshroom:

```bash
# From SfMData (normals only)
venv/bin/rnb-neus2 --input normals.sfm --testbed RNb-NeuS2/build/testbed --output out/

# From SfMData with reflectance (two-phase + albedo scaling), exporting the
# harmonized albedo maps for texturing
venv/bin/rnb-neus2 --input normals.sfm --albedo-sfm albedos.sfm \
    --mask-sfm masks.sfm --has-albedo --testbed RNb-NeuS2/build/testbed --output out/ \
    --harmonized-albedo-folder out/albedo --harmonized-albedo-sfm out/albedoHarmonized.sfm
```

## Notes

- The node works entirely in unit-sphere coordinates internally and exports the
  mesh back to world coordinates.
- Scene normalization is automatic (`auto` prefers silhouettes when masks are
  available, then falls back to 3D landmarks or camera centers).
- Albedo scaling runs only when an Albedo SfMData is provided (see
  [Albedo scaling](#albedo-scaling)).

## License

The code of this plugin is released under the [Mozilla Public License 2.0](LICENSE).
It calls RNb-NeuS2 (Python package and testbed), which is distributed under its own
license ([NVIDIA Source Code License](https://github.com/RobinBruneau/RNb-NeuS2/blob/main/LICENSE.txt),
non-commercial use only).
