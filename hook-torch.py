"""PyInstaller hook for PyTorch — collects DLLs and data files."""
from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

datas = collect_data_files('torch')
binaries = collect_dynamic_libs('torch')
