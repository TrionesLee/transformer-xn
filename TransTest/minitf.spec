# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller 打包配置。由 build.sh 通过环境变量控制:
    MINITF_ONEFILE=1   打成单个可执行文件(启动慢, 但只有一个文件)
    MINITF_CPU=1       剔除 CUDA 相关的大体积库, 只保留 CPU 推理/训练
    MINITF_NAME=xxx    可执行文件名(默认 minitf)
"""
import os
from PyInstaller.utils.hooks import collect_all

ONEFILE = os.environ.get('MINITF_ONEFILE') == '1'
CPU_ONLY = os.environ.get('MINITF_CPU') == '1'
NAME = os.environ.get('MINITF_NAME', 'minitf')

datas, binaries, hiddenimports = collect_all('torch')
hiddenimports += ['data', 'model', 'train', 'predict', 'walkthrough',
                  'inspect_model', 'visualize', 'report', 'export_params', 'handcalc']

# 这些包本项目用不到, 排除掉能明显减小体积
EXCLUDES = [
    'matplotlib', 'pandas', 'scipy', 'IPython', 'jupyter', 'notebook',
    'tkinter', 'PIL', 'pytest', 'setuptools._distutils', 'torchvision',
    'torchaudio', 'tensorboard', 'onnx', 'triton',
    # torch 的 datapipes 会可选地引用这些 NLP/科学计算库, 本项目完全用不到;
    # 不排除的话 PyInstaller 会去 import 它们, 环境里任何一个装坏了都会中断打包
    'nltk', 'sklearn', 'numba', 'dill', 'fsspec', 'requests', 'aiohttp',
    'gensim', 'spacy', 'transformers', 'datasets', 'h5py', 'zmq',
    'pyarrow', 'numpy.f2py', 'distutils', 'lib2to3', 'pydoc_data',
]

a = Analysis(
    ['cli.py'],
    pathex=['.'],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=EXCLUDES,
    noarchive=False,
)

if CPU_ONLY:
    # 剔除 CUDA / cuDNN / NCCL 等 GPU 专用动态库(通常占总体积的 80% 以上)
    DROP = ('libtorch_cuda', 'libtorch_hip', 'libcudnn', 'libcublas', 'libcufft',
            'libcurand', 'libcusolver', 'libcusparse', 'libnvrtc', 'libnccl',
            'libcupti', 'libcudart', 'libnvjitlink', 'libnvToolsExt', 'cudnn',
            'nvidia/', 'nvidia\\', 'cuda_runtime', 'cufile')

    def keep(entry):
        name = (entry[0] or '').lower().replace('\\', '/')
        src = (entry[1] or '').lower().replace('\\', '/')
        return not any(k.lower().replace('\\', '/') in name or
                       k.lower().replace('\\', '/') in src for k in DROP)

    n0, m0 = len(a.binaries), len(a.datas)
    a.binaries = TOC([x for x in a.binaries if keep(x)])
    a.datas = TOC([x for x in a.datas if keep(x)])
    print(f'[minitf.spec] CPU 精简: 二进制 {n0} -> {len(a.binaries)}, '
          f'数据 {m0} -> {len(a.datas)}')

pyz = PYZ(a.pure)

if ONEFILE:
    exe = EXE(pyz, a.scripts, a.binaries, a.datas, [],
              name=NAME, debug=False, bootloader_ignore_signals=False,
              strip=False, upx=False, console=True, disable_windowed_traceback=False,
              argv_emulation=False, target_arch=None, codesign_identity=None,
              entitlements_file=None)
else:
    exe = EXE(pyz, a.scripts, [], exclude_binaries=True,
              name=NAME, debug=False, bootloader_ignore_signals=False,
              strip=False, upx=False, console=True, disable_windowed_traceback=False,
              argv_emulation=False, target_arch=None, codesign_identity=None,
              entitlements_file=None)
    coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name=NAME)
