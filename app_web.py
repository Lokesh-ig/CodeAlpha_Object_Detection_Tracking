import sys
import os
import runpy

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

frontend_app = os.path.join(current_dir, "frontend", "app_web.py")
runpy.run_path(frontend_app, run_name="__main__")
