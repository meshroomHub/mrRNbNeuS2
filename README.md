# mrRNbNeuS2

Meshroom plugin for [RNb-NeuS2](https://github.com/meshroomHubWarehouse/RNb-NeuS2): neural surface reconstruction from
multi-view normal and reflectance maps (R. Bruneau, B. Brument et al., IJCV 2025).

The node reads AliceVision SfMData, trains the CUDA testbed and exports the mesh in world coordinates. With albedo maps,
it also exports them scaled as for training (harmonized), ready to texture the mesh with the AliceVision `Texturing` node
(see [Texturing the mesh with the albedo maps](#texturing-the-mesh-with-the-albedo-maps)). This repository holds the
Meshroom side only: the reconstruction code lives in [RNb-NeuS2](https://github.com/meshroomHubWarehouse/RNb-NeuS2) (fork
of [RobinBruneau/RNb-NeuS2](https://github.com/RobinBruneau/RNb-NeuS2)).

| Node | Step | Output |
|------|------|--------|
| `RNbNeuS2` | Mesh from normal maps, and optional albedo maps and masks | `mesh.obj`, harmonized albedo maps, `albedoHarmonized.sfm` |

---

## Requirements

- **[Meshroom](https://github.com/alicevision/Meshroom)** 2026.1 or later (develop branch until its release, tested on
  commit [`edc4b6a2`](https://github.com/alicevision/Meshroom/commit/edc4b6a2)): Meshroom 2025.1.x has neither the plugin
  manager nor the `[tool.meshroom]` settings of `pyproject.toml`
- **NVIDIA GPU** with a driver supporting CUDA 12 (570 or newer for RTX 50xx); prebuilt testbed for compute
  capabilities 7.5, 8.0, 8.6, 8.9, 9.0 and 12.0
- **Python** 3.10 or newer
- **Linux**: glibc 2.34 or newer (Ubuntu 22.04, Rocky / RHEL 9 and newer) for the prebuilt testbed

---

## Installation

### 1. Install the plugin

From Meshroom's plugin manager (*Manage Plugins*), or from the command line:

```bash
meshroom_plugins install mrRNbNeuS2
```

The installer creates the plugin `venv/` with `uv sync`; it installs the `rnb_neus2` package pinned in `pyproject.toml`.
The `RNbNeuS2` node then appears in the *Neural Reconstruction* category.

### 2. Install the testbed

Automatic download of the compiled testbed by the plugin manager is work in progress and coming soon. For now, install
it by hand (the warning about the `testbed` asset during the plugin install is expected): download the archive matching
your OS from the [RNb-NeuS2 v2.0.3 release](https://github.com/RobinBruneau/RNb-NeuS2/releases/tag/v2.0.3), check its
SHA-256 against the one shown on the release page, and extract it into the plugin folder:

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

The node then finds `RNb-NeuS2/build/testbed` (`testbed.exe` on Windows). Updating or reinstalling the plugin replaces the
plugin folder, `RNb-NeuS2/` included: extract the archive again afterwards.

### 3. Manual install (without the plugin manager)

Requires [uv](https://docs.astral.sh/uv/).

```bash
cd /path/to/plugins
git clone https://github.com/meshroomHub/mrRNbNeuS2.git
cd mrRNbNeuS2
UV_PROJECT_ENVIRONMENT=venv uv sync   # Meshroom looks for venv/ at the plugin root
# then step 2 above
export MESHROOM_PLUGINS_PATH=/path/to/plugins/mrRNbNeuS2:$MESHROOM_PLUGINS_PATH
```

On Windows (PowerShell): `$env:UV_PROJECT_ENVIRONMENT = "venv"; uv sync`, then
`$env:MESHROOM_PLUGINS_PATH = "C:\path\to\plugins\mrRNbNeuS2"`.

---

## Pipeline template

`meshroom/linoUniPsRnbNeus2.mgt` (*File > New Pipeline > Lino Uni Ps Rnb Neus2*) runs the structure-from-motion of
AliceVision, segments the object with SAM 3, exports the undistorted images, estimates the normals with `LINOUniPS` and
reconstructs the mesh with `RNbNeuS2` (quick settings: 1000 steps, resolution 512).

It needs [mrLINOUniPS](https://github.com/meshroomHub/mrLINOUniPS), from Meshroom's plugin manager, and
[mrSegmentation](https://github.com/meshroomHub/mrSegmentation) (node `ImageSegmentationSam3`), not in the plugin
registry yet: install it by hand.

CameraInit groups the images of a folder into one pose only if the folder name starts with `PS_`. Elsewhere, each image
becomes its own pose and the photometric stereo nodes find no multi-lighting pose, so images must follow this layout:

```
dataset/
└── PS/
    ├── PS_01/
    │   ├── ambient_all_leds.jpg   # reference image, lit by all the lights
    │   ├── led_01.jpg             # single-light images
    │   ├── led_02.jpg
    │   └── ...
    ├── PS_02/
    └── ...
```

The filename of the reference image must contain `ambient` (CameraInit makes it the pose reference) and `all_leds`
(`SfMFilter_1` keeps these images for the structure from motion and the segmentation).

Drag and drop the parent folder `dataset/` into the Image Gallery of Meshroom, then set the SAM 3 prompt
(`ImageSegmentationSam3_1`, default `stone`) to your object.

---

## Nodes

### `RNbNeuS2`

Connect *Normal Maps SfMData* (required), *Albedo Maps SfMData* and *Mask SfMData* or *Mask Folder* (optional) to the
outputs of a photometric stereo node, e.g. [mrLINOUniPS](https://github.com/meshroomHub/mrLINOUniPS) or
[mrLedGetter](https://github.com/meshroomHub/mrLedGetter), then compute the node.

#### Inputs

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

#### Outputs

| Parameter | Label | Description |
|-----------|-------|-------------|
| `outputFolder` | Output Folder | Folder with all training artifacts (`nodeCacheFolder`). |
| `outputMesh` | Output Mesh | Reconstructed `mesh.obj` in world coordinates (viewable in Meshroom). |
| `outputAlbedoFolder` | Harmonized Albedo Folder | `albedo/<viewId>.exr`: the input albedo maps multiplied by the per-view scales used for training, as linear float EXR (not clipped), extended 8 px outside the masks. Written only when `inputAlbedoSfm` is set (the attribute is disabled otherwise). |
| `outputAlbedoSfm` | Harmonized Albedo SfMData | `albedoHarmonized.sfm`: copy of the Albedo Maps SfMData (JSON) whose views point to these maps (views without a map are dropped). Written only when `inputAlbedoSfm` is set. |

---

## Albedo scaling

Photometric-stereo albedo maps are known up to one scale per view and channel. With an Albedo Maps SfMData, the node
trains on the geometry first, then solves these scales jointly from all the view pairs that see the same mesh points
and trains on the rescaled albedos (`compute_albedo_scale_ratios` in RNb-NeuS2; the scales are kept in
`prepared_data/albedo_scales.json`).

---

## Texturing the mesh with the albedo maps

Add the AliceVision nodes `MeshRemoveUnseenFaces` (on the mesh, otherwise the atlases can come out empty), then
`Texturing`, with these settings:

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

---

## Development

### Command line

The node wraps the `rnb-neus2` script of the plugin venv: see the [RNb-NeuS2 README](https://github.com/meshroomHubWarehouse/RNb-NeuS2#readme)
to run the same reconstruction without Meshroom.

### Working on a local RNb-NeuS2 clone

The node imports the `rnb_neus2` package from the plugin `venv/`. To run a local clone instead, install it in editable mode in that venv:

```bash
uv pip install --python venv/bin/python -e /path/to/RNb-NeuS2
```

Updating or reinstalling the plugin from the plugin manager recreates `venv/`: run this command again afterwards.

To use a local build of the testbed, set the environment variable before launching Meshroom (it overrides the plugin
default):

```bash
export RNB_NEUS2_TESTBED_PATH=/path/to/RNb-NeuS2/build/testbed
```

On Windows (PowerShell): `$env:RNB_NEUS2_TESTBED_PATH = "C:\path\to\RNb-NeuS2\build\testbed.exe"`.

### Plugin structure

```
mrRNbNeuS2/
├── meshroom/
│   ├── linoUniPsRnbNeus2.mgt      # Pipeline template
│   └── RNb-NeuS2/
│       ├── __init__.py
│       └── RNbNeuS2.py            # Meshroom node definition
├── RNb-NeuS2/                     # Compiled testbed (release archive)
├── venv/                          # Created by the installer or by uv sync
├── pyproject.toml                 # Dependencies and Meshroom settings (uv)
└── README.md
```

For more details on how Meshroom plugins work, see:
- [Meshroom Plugin Install Guide](https://github.com/alicevision/Meshroom/blob/develop/INSTALL_PLUGINS.md)
- [mrHelloWorld](https://github.com/meshroomHub/mrHelloWorld): step-by-step tutorials for building Meshroom plugins

---

## Acknowledgements

This work is supported by [**DOPAMIn**](https://www.cnrsinnovation.com/actualite/une-seconde-promotion-pour-le-programme-open-7-nouveaux-logiciels-scientifiques-a-valoriser/) (*Diffusion Open de Photogrammetrie par AliceVision/Meshroom pour l'Industrie*), selected in the 2024 cohort of the [**OPEN**](https://www.cnrsinnovation.com/open/) programme run by [CNRS Innovation](https://www.cnrsinnovation.com/). OPEN supports the valorization of open-source scientific software by providing dedicated developer resources, governance expertise, and industry partnership support.

**Lead researcher:** [Jean-Denis Durou](https://cv.hal.science/jean-denis-durou), [IRIT](https://www.irit.fr/) (INP-Toulouse)
**Co-lead:** [Lilian Calvet](https://fr.linkedin.com/in/lilian-calvet-42b1a689), [Balgrist University Hospital](https://www.balgrist.ch/)

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

## Citation

- **Multi-view Surface Reconstruction Using Normal and Reflectance Cues**  
  R. Bruneau, B. Brument, Y. Quéau, J. Mélou, F. B. Lauze, J.-D. Durou, L. Calvet (2025)  
  In *International Journal of Computer Vision (IJCV)*.  
  [arXiv](https://arxiv.org/abs/2506.04115)

- **RNb-NeuS: Reflectance and Normal-based Multi-View 3D Reconstruction**  
  B. Brument, R. Bruneau, Y. Quéau, J. Mélou, F. Lauze, J.-D. Durou, L. Calvet (2024)  
  In *IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*.  
  [Project page](https://robinbruneau.github.io/publications/rnb_neus.html)

---

## License

The code of this plugin is released under the [Mozilla Public License 2.0](LICENSE).
It calls RNb-NeuS2 (Python package and testbed), which is distributed under its own
license ([NVIDIA Source Code License](https://github.com/RobinBruneau/RNb-NeuS2/blob/main/LICENSE.txt),
non-commercial use only).
