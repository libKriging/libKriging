import importlib.util
import sys
from pathlib import Path

# This file a hack to redirect calls from libKriging root to bindings/Python


# To help to find check_requirements.py which is in a non local part (according to the current file)

python_binding_path = Path(__file__).absolute().parent / "bindings" / "Python"
sys.path.append(str(python_binding_path))

# Since this file and the file to import have the same name, we need a manual loading

module_name = "setup"
spec = importlib.util.spec_from_file_location(module_name, python_binding_path / "setup.py")
module = importlib.util.module_from_spec(spec)
sys.modules[module_name] = module
spec.loader.exec_module(module)

# Redirection is now ready

module.main()
