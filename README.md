# VFM-Registration-MasterThesis

Code for the master's thesis **"Visual Foundation Models for Dual-Plane 2D/3D
Registration Acceleration in Knee Surgery."**

This is a **code-only** repository: the thesis text and documents are kept
elsewhere. It contains the segmentation, registration, and GUI code, plus a
small set of non-patient result figures.

## Layout

```
code/
  segmentation/   SAM2/SAM3 zero-shot + interactive scripts; E1-E3 experiment drivers
  registration/   Dual-plane registration (register.py, evaluation.py,
                  run_R1-R3.py, smoke_test.py) and the R1/R3/geometry plot scripts
  gui/            OpenCV interactive segmentation GUI (03_tool/test_sam2.py)
  sam2_lib/       SAM2 / MedSAM2 framework (library + training source; weights excluded)
envs/             pip requirements for the SAM2 / SAM3 environments
results/          Selected non-patient result figures (plots + synthetic renders)
```

## Intentionally NOT in this public repository

Excluded via `.gitignore`:

- **Patient imagery** — all clinical knee X-rays and any patient-bearing
  figure/overlay/sample (`data/`, `code/gui/02_view_results/`,
  `code/gui/04_samples/`, etc.). Only synthetic, diagrammatic, and plot
  figures are published under `results/`.
- **Model weights** — SAM2/SAM3 checkpoints and the MedSAM2 fine-tune
  (`*.pt`, `checkpoints/`, `code/gui/03_tool/weights/`). Get SAM2 weights from
  `facebookresearch/sam2`; the fine-tuned checkpoint is available on request.
- **`sam2_lib` heavy assets** — the SAM2 library and `training/` source are
  included, but its `checkpoints/`, `demo/`, `notebooks/`, `sav_dataset/`, and
  video assets are not.
- **Other third-party clones** — the SAM3 code (`code/gui/03_tool/sam3_code/`)
  and the Veriserum `dupla_renderers` package (unpublished renderer). Obtain
  from their sources.
- **Thesis text / documents** — LaTeX chapters, bibliography, and project
  docs are maintained separately.
