# SAM3 Bone Segmentation Delivery Notes (2026-03-20)

## 1. Delivery Goals
- Two types of deliverables are provided to the client:
- `Viewable`: inspect the segmentation results directly, without running any code.
- `Runnable`: launch the interactive annotation interface to continue annotating and revising.

## 2. Directory Description
- `02_view_results/images`: original images (currently used for viewing/re-annotation).
- `02_view_results/labels`: annotation output directory.
- `02_view_results/index.csv`: result index and statistics.
- `03_tool/test_sam2.py`: main interactive annotation program.
- `03_tool/weights/sam3.pt`: SAM3 base model weights.
- `03_tool/sam3_code`: SAM3 code dependencies.
- `03_tool/run_gui_auto.bat`: automatically select the device (GPU if CUDA is available).
- `03_tool/run_gui_gpu.bat`: force GPU.
- `03_tool/run_gui_cpu.bat`: force CPU.
- `04_samples`: sample images and their corresponding 5 output files.

## 3. The 5 Output Files Per Image
- `*_labels.json`: annotation metadata, point prompts, pixel statistics, output paths.
- `*_vis.png`: original image overlaid with the segmented regions for visualization.
- `*_label_map.png`: single-channel class map (0 background, 1 femur, 2 tibia, 3 overlap).
- `*_label_color.png`: color label map.
- `*_points.png`: original image overlaid with positive/negative prompt points for visualization.

## 4. How to Run
1. Enter the `03_tool` directory.
2. After activating the `conda` environment, double-click any of the following scripts:
- `run_gui_auto.bat`
- `run_gui_gpu.bat`
- `run_gui_cpu.bat`

## 5. Notes on the Current Model
- The current delivery uses the base weights `sam3.pt` by default.
- The fine-tuned weights are not included in this package; the program automatically skips loading the fine-tuned model when `--ft-ckpt` is empty or the path does not exist.

## 6. Acceptance Recommendations
- Randomly spot-check whether each sample under `02_view_results/labels` has all 5 outputs.
- Use `index.csv` to cross-check the pixel statistics and file paths.
- After launching the GUI, press `A/D` to flip through images and confirm that auto-save works correctly.
