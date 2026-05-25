@echo off
setlocal
chcp 65001 >nul
set "BASE=%~dp0"
cd /d "%BASE%"
python "%BASE%test_sam2.py" ^
  --ckpt "%BASE%weights\sam3.pt" ^
  --ft-ckpt "" ^
  --device cpu ^
  --image "%BASE%..\02_view_results\images" ^
  --out-dir "%BASE%..\02_view_results\labels"
endlocal
