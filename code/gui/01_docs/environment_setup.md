# Environment Setup Guide (Delivery Version)

## 1. Recommended System
- Windows 10/11 x64
- Python 3.10 (recommended)
- Conda (Miniconda/Anaconda)
- NVIDIA GPU (optional; faster when CUDA is available)

## 2. Quick Install (New Environment)
Run the following in `Anaconda Prompt` or `PowerShell`:

```powershell
conda create -n sam3_delivery python=3.10 -y
conda activate sam3_delivery
pip install -r G:\sam3\2026-3-20\03_tool\requirements.txt
```

## 3. GPU Version (Optional)
If you need to force GPU, first confirm that `torch.cuda.is_available()==True`.  
Then run:

```powershell
python G:\sam3\2026-3-20\03_tool\test_sam2.py --ckpt G:\sam3\2026-3-20\03_tool\weights\sam3.pt --ft-ckpt "" --device cuda --image G:\sam3\2026-3-20\02_view_results\images --out-dir G:\sam3\2026-3-20\02_view_results\labels
```

If `--device cuda` raises an error, switch to `--device cpu`, or install a PyTorch CUDA version matching your graphics card driver.

## 4. CPU Version (Fallback)
```powershell
python G:\sam3\2026-3-20\03_tool\test_sam2.py --ckpt G:\sam3\2026-3-20\03_tool\weights\sam3.pt --ft-ckpt "" --device cpu --image G:\sam3\2026-3-20\02_view_results\images --out-dir G:\sam3\2026-3-20\02_view_results\labels
```

## 5. One-Click Launch Scripts
- Auto device: `G:\sam3\2026-3-20\03_tool\run_gui_auto.bat`
- Force GPU: `G:\sam3\2026-3-20\03_tool\run_gui_gpu.bat`
- Force CPU: `G:\sam3\2026-3-20\03_tool\run_gui_cpu.bat`

## 6. Common Issues
- Error: missing `imagecodecs` (LZW TIF):
```powershell
pip install imagecodecs
```

- Error: `No module named 'triton'`:
This is an optional post-processing hint from SAM3. It does not affect the main segmentation pipeline and can be ignored.

- Fine-tuned weights path does not exist:
This delivery uses `sam3.pt` by default. If `--ft-ckpt` is left empty or the path does not exist, it is skipped automatically.
