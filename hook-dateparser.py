"""
PyInstaller hook for dateparser
Ensures dateparser data files are included in the bundle
"""
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

# Collect all data files from dateparser
datas = collect_data_files('dateparser')

# Collect all submodules
hiddenimports = collect_submodules('dateparser')

# Also ensure regex is included (dateparser dependency)
hiddenimports += ['regex', 'pytz']
