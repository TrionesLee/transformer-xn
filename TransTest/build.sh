#!/usr/bin/env bash
# 把项目打包成免安装 Python 的可执行程序。
#
#   ./build.sh                  目录版(推荐): dist/minitf/minitf, 启动快
#   ./build.sh --onefile        单文件版:     dist/minitf, 拷贝方便但启动慢
#   ./build.sh --cpu            精简版:       用 CPU-only 的 torch 打包(体积小 5 倍)
#   ./build.sh --onefile --cpu  单文件 + 精简
#   ./build.sh --name mytool    自定义可执行文件名
#
# Windows 下在 PowerShell/CMD 里执行同样的 pyinstaller 命令即可得到 minitf.exe:
#   pyinstaller --clean --noconfirm minitf.spec
set -euo pipefail
cd "$(dirname "$0")"

export MINITF_ONEFILE=0 MINITF_CPU=0 MINITF_NAME=minitf
while [ $# -gt 0 ]; do
  case "$1" in
    --onefile) MINITF_ONEFILE=1; shift ;;
    --cpu)     MINITF_CPU=1;     shift ;;
    --name)    MINITF_NAME="$2"; shift 2 ;;
    -h|--help) sed -n '2,14p' "$0"; exit 0 ;;
    *) echo "未知参数: $1"; exit 1 ;;
  esac
done

# 选择用哪个 Python 环境打包
if [ "$MINITF_CPU" = "1" ]; then
  # CUDA 版 torch 在 import 阶段就硬依赖 libtorch_cuda.so，删库必崩，
  # 所以精简版必须用 CPU-only 的 torch 轮子，装在独立 venv 里（不污染主环境）。
  VENV="${MINITF_VENV:-.venv-cpu}"
  if [ ! -x "$VENV/bin/pyinstaller" ]; then
    echo "==> 准备 CPU-only 打包环境 $VENV （首次需下载约 200MB，请联网）"
    python3 -m venv "$VENV"
    "$VENV/bin/pip" install -q --upgrade pip
    "$VENV/bin/pip" install -q torch --index-url https://download.pytorch.org/whl/cpu
    "$VENV/bin/pip" install -q numpy pyinstaller
  fi
  PY="$VENV/bin/python"
  PYI="$VENV/bin/pyinstaller"
else
  command -v pyinstaller >/dev/null || { echo "缺少 pyinstaller，请先: pip install pyinstaller"; exit 1; }
  PY="python3"
  PYI="pyinstaller"
fi

# --- 打包期间的临时环境处理(结束后一律自动还原, 不永久改动你的 Python 环境) ---
# 1) pathlib 向后兼容包: Python 2 时代遗留, PyInstaller 明确拒绝在其存在时打包
# 2) conda 的 backports/__init__.py: 把 backports 变成普通包, 遮蔽了 setuptools
#    依赖的 backports.tarfile, 导致 PyInstaller 的 setuptools 钩子崩溃
SP="$($PY -c 'import site; print(site.getsitepackages()[0])')"
STASH=""
restore_env() {
  [ -n "$STASH" ] || return 0
  [ -f "$STASH/pathlib.py" ] && mv "$STASH/pathlib.py" "$SP"/ 2>/dev/null || true
  for d in "$STASH"/pathlib-*; do [ -e "$d" ] && mv "$d" "$SP"/ 2>/dev/null || true; done
  [ -f "$STASH/backports__init__.py" ] && \
    mv "$STASH/backports__init__.py" "$SP/backports/__init__.py" 2>/dev/null || true
  rmdir "$STASH" 2>/dev/null || true
  echo "    已还原打包期间临时移走的文件"
}
STASH="$(mktemp -d)"
trap restore_env EXIT INT TERM
if [ -e "$SP/pathlib.py" ] || ls -d "$SP"/pathlib-*.dist-info >/dev/null 2>&1; then
  echo "    临时移走过时的 pathlib 向后兼容包"
  mv "$SP/pathlib.py" "$STASH"/ 2>/dev/null || true
  for d in "$SP"/pathlib-*.dist-info "$SP"/pathlib-*.egg-info; do
    [ -e "$d" ] && mv "$d" "$STASH"/ 2>/dev/null || true
  done
fi
if [ -f "$SP/backports/__init__.py" ] && \
   ! $PY -c 'import setuptools' >/dev/null 2>&1; then
  echo "    临时移走 conda 的 backports/__init__.py (它挡住了 setuptools)"
  mv "$SP/backports/__init__.py" "$STASH/backports__init__.py" 2>/dev/null || true
fi

echo "==> 打包中 (名称=$MINITF_NAME, 单文件=$MINITF_ONEFILE, CPU精简=$MINITF_CPU)"
echo "    首次打包需要几分钟，PyTorch 体积较大，请耐心等待"
rm -rf build "dist/$MINITF_NAME"
"$PYI" --clean --noconfirm --log-level=WARN minitf.spec

if [ "$MINITF_ONEFILE" = "1" ]; then
  BIN="dist/$MINITF_NAME"
else
  BIN="dist/$MINITF_NAME/$MINITF_NAME"
fi
echo
echo "==> 完成: $BIN"
du -sh "dist/$MINITF_NAME" 2>/dev/null | awk '{print "    体积: "$1}'
echo
echo "试运行:"
echo "  $BIN info"
echo "  $BIN train --steps 500"
echo "  $BIN predict --prompt \"the qu\""
