# -*- mode: python ; coding: utf-8 -*-
import sys
import os
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None

# Collect TTS package data (model configs, etc.)
tts_datas = collect_data_files('TTS')
trainer_datas = collect_data_files('trainer')

a = Analysis(
    ['gui_app.py'],
    pathex=['.'],
    binaries=[
        ('ffmpeg.exe', '.'),
    ],
    datas=tts_datas + trainer_datas,
    hiddenimports=[
        # TTS / Coqui
        'TTS',
        'TTS.api',
        'TTS.utils',
        'TTS.utils.audio',
        'TTS.utils.audio.processor',
        'TTS.tts',
        'TTS.tts.configs',
        'TTS.tts.configs.xtts_config',
        'TTS.tts.models',
        'TTS.tts.models.xtts',
        'TTS.tts.layers',
        'TTS.tts.layers.xtts',
        'TTS.tts.layers.xtts.gpt',
        'TTS.tts.layers.xtts.hifigan_decoder',
        'TTS.tts.layers.xtts.stream_generator',
        'TTS.tts.layers.xtts.tokenizer',
        'TTS.tts.layers.xtts.perceiver_encoder',
        'TTS.vocoder',
        'TTS.encoder',
        'TTS.config',
        # Torch
        'torch',
        'torch.nn',
        'torchaudio',
        'torchaudio.transforms',
        'torchaudio.functional',
        # Audio
        'librosa',
        'librosa.core',
        'librosa.effects',
        'librosa.feature',
        'soundfile',
        'pydub',
        'pydub.audio_segment',
        'pydub.effects',
        # Video
        'moviepy',
        'moviepy.editor',
        'moviepy.video.io',
        'moviepy.audio.io',
        # Scipy / Numpy
        'scipy',
        'scipy.signal',
        'scipy.io',
        'scipy.io.wavfile',
        'scipy.special._ufuncs_cxx',
        'numpy',
        'numpy.core._dtype_ctypes',
        # Japanese support (optional but included to avoid import errors)
        'cutlet',
        'fugashi',
        'unidic_lite',
        # Sklearn (used internally by librosa)
        'sklearn',
        'sklearn.utils',
        'sklearn.utils._typedefs',
        'sklearn.utils._heap',
        'sklearn.utils._sorting',
        'sklearn.utils._vector_sentinel',
        'sklearn.neighbors._partition_nodes',
        # Numba (used by librosa)
        'numba',
        'numba.core',
        # Misc
        'packaging',
        'fsspec',
        'huggingface_hub',
        'transformers',
        'einops',
        'inflect',
        'anyascii',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'faster_whisper',
        'whisper',
        'deep_translator',
        'tkinter.test',
    ],
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
    name='VoiceClone',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='VoiceClone',
)
