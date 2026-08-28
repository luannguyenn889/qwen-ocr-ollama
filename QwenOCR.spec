# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path
import sys
from PyInstaller.utils.hooks import collect_all

block_cipher = None

# Collect all non-Python data files, submodules, and configs from all OCR and Layout packages
paddlex_datas, paddlex_binaries, paddlex_hiddenimports = collect_all('paddlex')
paddle_datas, paddle_binaries, paddle_hiddenimports = collect_all('paddle')
paddleocr_datas, paddleocr_binaries, paddleocr_hiddenimports = collect_all('paddleocr')
cython_datas, cython_binaries, cython_hiddenimports = collect_all('Cython')
pyclipper_datas, pyclipper_binaries, pyclipper_hiddenimports = collect_all('pyclipper')
shapely_datas, shapely_binaries, shapely_hiddenimports = collect_all('shapely')
skimage_datas, skimage_binaries, skimage_hiddenimports = collect_all('skimage')

added_files = [
    ('app/core/vietnamese_lexicon.json', 'app/core'),
] + paddlex_datas + paddle_datas + paddleocr_datas + cython_datas + pyclipper_datas + shapely_datas + skimage_datas

all_binaries = paddlex_binaries + paddle_binaries + paddleocr_binaries + cython_binaries + pyclipper_binaries + shapely_binaries + skimage_binaries

hidden_imports = [
    'paddle',
    'paddlex',
    'paddleocr',
    'Cython',
    'pyclipper',
    'shapely',
    'skimage',
    'pymupdf',
    'fitz',
    'cv2',
    'PIL',
    'PIL.Image',
    'PIL.ImageTk',
    'PIL.ImageDraw',
    'PIL.ImageFilter',
    'PIL.ImageOps',
    'ollama',
    'numpy',
    'rapidfuzz',
    'unittest',
    'app',
    'app.core',
    'app.core.batch_ocr',
    'app.core.layout_detector',
    'app.core.block_assembler',
    'app.core.markdown_normalizer',
    'app.core.quality_gate',
    'app.core.vietnamese_spell_corrector',
    'app.core.image_preprocessor',
    'app.core.formula_ocr',
    'app.gui',
    'app.gui.run_gui',
] + paddlex_hiddenimports + paddle_hiddenimports + paddleocr_hiddenimports + cython_hiddenimports + pyclipper_hiddenimports + shapely_hiddenimports + skimage_hiddenimports

a = Analysis(
    ['run_gui.py'],
    pathex=['.'],
    binaries=all_binaries,
    datas=added_files,
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['tkinter.test', 'pytest'],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='QwenOCR_App',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='QwenOCR_App',
)
