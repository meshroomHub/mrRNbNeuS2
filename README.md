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

- **[Meshroom](https://github.com/alicevision/Meshroom)** 2026.1 or later (develop branch until its release):
  Meshroom 2025.1.x has neither the plugin manager nor the `[tool.meshroom]` settings of `pyproject.toml`
- **NVIDIA GPU** with a driver supporting CUDA 12 (570 or newer for RTX 50xx); prebuilt testbed for compute
  capabilities 7.5, 8.0, 8.6, 8.9, 9.0 and 12.0
- **Linux**: glibc 2.34 or newer (Ubuntu 22.04, Rocky / RHEL 9 and newer) for the prebuilt testbed

## Installation

### 1. Install the plugin

From Meshroom's plugin manager (*Manage Plugins*), or from the command line:

```bash
meshroom_plugins install mrRNbNeuS2
```

The installer creates the plugin `venv/` with `uv sync`; it installs the `rnb_neus2`
package pinned in `pyproject.toml`.

### 2. Install the RNb-NeuS2 testbed

The plugin manager installs the compiled testbed with the plugin, when it supports
release assets: it picks the archive of your OS, checks its SHA-256 and extracts it
into `RNb-NeuS2/`. There is nothing else to do.

Otherwise, the `testbed` asset of `pyproject.toml` is ignored with a warning, and the
archive is installed by hand. Download the archive matching your OS from the
[RNb-NeuS2 v2.0.3 release](https://github.com/RobinBruneau/RNb-NeuS2/releases/tag/v2.0.3),
check its SHA-256 against the one shown on the release page, and extract it into the
plugin folder:

Linux:

```bash
cd /path/to/plugins/mrRNbNeuS2
sha256sum /path/to/RNb-NeuS2-2.0.3-linux-x86_64-cu128.tar.gz
tar xzf /path/to/RNb-NeuS2-2.0.3-linux-x86_64-cu128.tar.gz   # gives mrRNbNeuS2/RNb-NeuS2/build/testbed
```

Windows (PowerShell):

```powershell
cd C:\path\to\plugins\mrRNbNeuS2
Get-FileHash C:\path\to\RNb-NeuS2-2.0.3-windows-x86_64-cu128.zip   # SHA256
Expand-Archive C:\path\to\RNb-NeuS2-2.0.3-windows-x86_64-cu128.zip -DestinationPath .   # gives mrRNbNeuS2\RNb-NeuS2\build\testbed.exe
```

The node then finds `RNb-NeuS2/build/testbed` (`testbed.exe` on Windows; default of `RNB_NEUS2_TESTBED_PATH`
in `pyproject.toml`, relative to the plugin root). The archive holds the compiled testbed
only: `build/` and the `configs/` and `utils/` folders it reads next to `build/`.

To keep the testbed elsewhere (e.g. a local build), set the environment variable
before launching Meshroom; it overrides the plugin default:

```bash
export RNB_NEUS2_TESTBED_PATH=/path/to/RNb-NeuS2/build/testbed
```

On Windows (PowerShell): `$env:RNB_NEUS2_TESTBED_PATH = "C:\path\to\RNb-NeuS2\build\testbed.exe"`.

A plugin manager without release assets replaces the plugin folder when updating or
reinstalling it, `RNb-NeuS2/` included: extract the release archive again afterwards,
or keep it outside the plugin folder with `RNB_NEUS2_TESTBED_PATH`.

### Manual install (without the plugin manager)

```bash
cd /path/to/plugins
git clone https://github.com/meshroomHub/mrRNbNeuS2.git
cd mrRNbNeuS2
UV_PROJECT_ENVIRONMENT=venv uv sync   # Meshroom looks for venv/ at the plugin root
# then step 2 above
export MESHROOM_PLUGINS_PATH=/path/to/plugins/mrRNbNeuS2:$MESHROOM_PLUGINS_PATH
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

Photometric-stereo albedo maps are known up to one scale per view and channel. With an Albedo Maps SfMData, the node
trains on the geometry first, then solves these scales jointly from all the view pairs that see the same mesh points
and trains on the rescaled albedos (`compute_albedo_scale_ratios` in RNb-NeuS2; the scales are kept in
`prepared_data/albedo_scales.json`).

## Texturing the mesh with the albedo maps

Add the AliceVision nodes `MeshRemoveUnseenFaces` (on the mesh, otherwise the atlases can come out empty), then
`Texturing`. The mesh needs no transform: `RNbNeuS2` exports it in the AliceVision mesh-file convention. `Texturing`
settings:

| Parameter | Value | Why |
|-----------|-------|-----|
| `input` | *Harmonized Albedo SfMData* | Cameras of the albedo maps |
| `imagesFolder` | *Harmonized Albedo Folder* | Required: no texture is written otherwise |
| `visibilityRemappingMethod` | `MeshItself` | The SfMData has no landmarks |
| `subdivisionTargetRatio` | `0` | The default aborts without landmarks |
| `workingColorSpace` | `Linear` | The albedo maps are linear |
| `correctEV` | off | The albedo maps are not exposures |
| `textureSide` / `downscale` | 4096 / 1 | |
| `colorMappingFileType` | `exr` (linear data) or `png` (preview) | |

## Usage in Meshroom

Connect `Normal Maps SfMData` (required), `Albedo Maps SfMData` and `Mask SfMData` or `Mask Folder` (optional) to the
outputs of a photometric stereo node, e.g. [mrLINOUniPS](https://github.com/meshroomHub/mrLINOUniPS) or
[mrLedGetter](https://github.com/meshroomHub/mrLedGetter), then compute the node.

---

## Pipeline template

`meshroom/linoUniPsRnbNeus2.mgt` (*File > New Pipeline > Lino Uni Ps Rnb Neus2*) runs the structure-from-motion of
AliceVision, segments the object with SAM 3, exports the undistorted images, estimates the normals with `LINOUniPS` and
reconstructs the mesh with `RNbNeuS2` (quick settings: 1000 steps, resolution 512).

It needs [mrLINOUniPS](https://github.com/meshroomHub/mrLINOUniPS), from Meshroom's plugin manager, and
[mrSegmentation](https://github.com/meshroomHub/mrSegmentation) (node `ImageSegmentationSam3`), not in the plugin
registry yet: install it by hand.

Images: one folder per pose, holding the single-light images and one reference image lit by all the lights, whose
filename contains `ambient` (CameraInit makes it the pose reference) and `all_leds` (`SfMFilter_1` keeps these images
for the structure from motion and the segmentation). Set the SAM 3 prompt (`ImageSegmentationSam3_1`, default `stone`)
to your object.

---

## Command-line equivalent

The node wraps the `rnb-neus2` script of the plugin venv, which runs the same reconstruction without Meshroom:

```bash
# Normals only
venv/bin/rnb-neus2 --input normals.sfm --testbed RNb-NeuS2/build/testbed --output out/

# Normals, albedos and masks, exporting the harmonized albedo maps for texturing
venv/bin/rnb-neus2 --input normals.sfm --albedo-sfm albedos.sfm --has-albedo --mask-folder masks/ \
    --testbed RNb-NeuS2/build/testbed --output out/ \
    --harmonized-albedo-folder out/albedo --harmonized-albedo-sfm out/albedoHarmonized.sfm
```

---

## Related Projects

| Project | Description |
|---------|-------------|
| [RNb-NeuS2](https://github.com/meshroomHubWarehouse/RNb-NeuS2) | Neural surface reconstruction from normal and reflectance maps (IJCV 2025) |
| [mrLINOUniPS](https://github.com/meshroomHub/mrLINOUniPS) | Meshroom plugin for LINO-UniPS photometric stereo |
| [mrSDMUniPS](https://github.com/meshroomHub/mrSDMUniPS) | Meshroom plugin for SDM-UniPS photometric stereo |
| [mrUniMSPS](https://github.com/meshroomHub/mrUniMSPS) | Meshroom plugin for Uni-MS-PS photometric stereo |
| [mrLedGetter](https://github.com/meshroomHub/mrLedGetter) | Meshroom plugin for LED-getter near-light photometric stereo |
| [mrOpenRNb](https://github.com/meshroomHub/mrOpenRNb) | Meshroom plugin for Open-RNb neural surface reconstruction |

---

## License

The code of this plugin is released under the [Mozilla Public License 2.0](LICENSE).
It calls RNb-NeuS2 (Python package and testbed), which is distributed under its own
license ([NVIDIA Source Code License](https://github.com/RobinBruneau/RNb-NeuS2/blob/main/LICENSE.txt),
non-commercial use only).
