import sys
import os

frontend_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend")
if frontend_dir not in sys.path:
    sys.path.insert(0, frontend_dir)

from frontend.app_web import main

if __name__ == "__main__":
    main()
