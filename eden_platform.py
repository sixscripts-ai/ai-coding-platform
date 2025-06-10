"""Import wrapper to expose `eden-platform.py` as `eden_platform`."""

from pathlib import Path
import importlib.util
import sys

_path = Path(__file__).with_name('eden-platform.py')
spec = importlib.util.spec_from_file_location('eden_platform_main', _path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
sys.modules.setdefault('eden_platform_main', module)

for name in dir(module):
    if not name.startswith('_'):
        globals()[name] = getattr(module, name)

