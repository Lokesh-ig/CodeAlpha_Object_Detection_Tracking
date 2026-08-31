import sys
import os

# Delegate to frontend/app_gui.py
frontend_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend")
if frontend_dir not in sys.path:
    sys.path.insert(0, frontend_dir)

from frontend.app_gui import DashboardWindow, QApplication

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DashboardWindow()
    window.showMaximized()
    sys.exit(app.exec())
