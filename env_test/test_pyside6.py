import sys
from PySide6.QtWidgets import QApplication, QLabel
app = QApplication(sys.argv)
label = QLabel("Hello, PySide6 on Windows 11!")
label.resize(300, 80)
label.show()
sys.exit(app.exec())