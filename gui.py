import sys
from PyQt5.QtWidgets import (
    QWidget, QHBoxLayout, QLabel, QSplitter
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage 

class AppGui(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Face Detect") # @TODO: add as param
        self.init_ui()
    
    def init_ui(self):
        main_layout = QHBoxLayout(self)

        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.video_label)

        self.setLayout(main_layout)
        self.resize(800, 600)
    
    def update_frame(self, frame):
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        qt_img = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_img)
        self.video_label.setPixmap(pixmap.scaled(
            self.video_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        ))