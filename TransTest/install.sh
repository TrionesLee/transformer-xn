#!/usr/bin/env bash
# 一键创建虚拟环境并安装依赖。
#
#   ./install.sh            # CPU 版 torch（约 200MB，无需显卡）
#   ./install.sh --gpu      # CUDA 版 torch（约 2.5GB，需要 NVIDIA 显卡）
#   ./install.sh --build    # 额外装 PyInstaller，用于打包可执行程序
#   ./install.sh --here     # 不建虚拟环境，直接装进当前 Python 环境
#
# 装完后按提示激活虚拟环境即可运行 train.py / predict.py / handcalc.py。
set -euo pipefail
cd "$(dirname "$0")"

VENV=".venv"
GPU=0; BUILD=0; HERE=0
while [ $# -gt 0 ]; do
  case "$1" in
    --gpu)   GPU=1;   shift ;;
    --build) BUILD=1; shift ;;
    --here)  HERE=1;  shift ;;
    --venv)  VENV="$2"; shift 2 ;;
    -h|--help) sed -n '2,10p' "$0"; exit 0 ;;
    *) echo "未知参数: $1"; exit 1 ;;
  esac
done

if [ "$HERE" = "1" ]; then
  PIP="python3 -m pip"
  echo "==> 直接安装到当前 Python 环境"
else
  if [ ! -x "$VENV/bin/python" ]; then
    echo "==> 创建虚拟环境 $VENV"
    python3 -m venv "$VENV"
  else
    echo "==> 复用已有的虚拟环境 $VENV"
  fi
  PIP="$VENV/bin/pip"
  $PIP install -q --upgrade pip
fi

REQ="requirements.txt"
[ "$BUILD" = "1" ] && REQ="requirements.txt -r requirements-build.txt"

if [ "$GPU" = "1" ]; then
  echo "==> 安装 CUDA 版 PyTorch（约 2.5GB，首次较慢）"
  $PIP install -r $REQ
else
  echo "==> 安装 CPU 版 PyTorch（约 200MB）"
  $PIP install -r $REQ --index-url https://download.pytorch.org/whl/cpu \
                       --extra-index-url https://pypi.org/simple
fi

echo
echo "==> 安装完成，实测环境:"
if [ "$HERE" = "1" ]; then PY="python3"; else PY="$VENV/bin/python"; fi
$PY - <<'PY'
import torch, sys
print(f'   Python  : {sys.version.split()[0]}')
print(f'   PyTorch : {torch.__version__}')
print(f'   CUDA    : {"可用 — " + torch.cuda.get_device_name(0) if torch.cuda.is_available() else "不可用（将使用 CPU）"}')
PY

echo
if [ "$HERE" != "1" ]; then
  echo "使用前先激活虚拟环境:"
  echo "   source $VENV/bin/activate"
  echo
fi
echo "然后就可以跑了:"
echo "   python3 train.py --input \"I like easy course\" --report 300"
echo "   python3 predict.py --prompt \"i li\""
echo "   python3 handcalc.py --prompt \"i li\" --verify"
[ "$BUILD" = "1" ] && echo "   ./build.sh          # 打包成可执行程序"
