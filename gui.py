import sys
from PyQt5.QtWidgets import (
    QDialog, QFormLayout, QDialogButtonBox, QLineEdit, QLabel, QComboBox, QSplitter, QVBoxLayout, QHBoxLayout, QWidget, QListWidget, QFrame, QPushButton, QSizePolicy, QAbstractItemView
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtPrintSupport import QPrinter, QPrintDialog
from db import Session, get_all_students
import time


class PromptDialog(QDialog):
    def __init__(self, frame=None):
        super().__init__()
        self.setWindowTitle("Recognize Student")
        self.layout = QVBoxLayout(self)

        self.student_combo = QComboBox(self)
        self.student_combo.addItem("Add new student")
        with Session() as session:
            self.students = get_all_students(session)
            for student in self.students:
                self.student_combo.addItem(f"{student.name} ({student.group})")
        self.student_combo.currentIndexChanged.connect(self.on_student_selected)
        self.layout.addWidget(QLabel("select existing or add:"))
        self.layout.addWidget(self.student_combo)


        self.form_layout = QFormLayout()
        self.name_input = QLineEdit(self)
        self.group_input = QLineEdit(self)
        self.form_layout.addRow("Name:", self.name_input)
        self.form_layout.addRow("Group:", self.group_input)
        self.layout.addLayout(self.form_layout)

        if frame is not None:
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            qt_img = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_img)
            self.image_label = QLabel(self)
            self.image_label.setPixmap(pixmap.scaled(
                200, 200,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            ))
            self.layout.addWidget(self.image_label)

        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        self.layout.addWidget(self.buttons)

        self.on_student_selected(0)

    def on_student_selected(self, idx):
        is_new = (idx == 0)
        self.name_input.setEnabled(is_new)
        self.group_input.setEnabled(is_new)

    def get_selection(self):
        idx = self.student_combo.currentIndex()
        if idx == 0:
            return {
                "new": True,
                "name": self.name_input.text(),
                "group": self.group_input.text(),
                "student": None
            }
        else:
            return {
                "new": False,
                "name": None,
                "group": None,
                "student": self.students[idx-1]
            }


class InfoDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Enter class info")
        self.layout = QFormLayout(self)
        self.subject_input = QLineEdit(self)
        self.group_input = QLineEdit(self)
        self.layout.addRow("Subject:", self.subject_input)
        self.layout.addRow("Group:", self.group_input)
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        self.layout.addWidget(self.buttons)
    
    def get_subject_and_group(self):
        return self.subject_input.text(), self.group_input.text()

class AppGui(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Face Recognition Attendance")
        self.subject_name = ""
        self.group = ""
        self.prompt_for_subject_and_group()
        self.init_ui()
        self.students = set()

    def prompt_for_subject_and_group(self):
        dialog = InfoDialog()
        if dialog.exec_() == QDialog.Accepted:
            self.subject_name, self.group = dialog.get_subject_and_group()
        else:
            self.subject_name, self.group = "Unknown", "Unknown"
    
    def init_ui(self):
        main_layout = QVBoxLayout(self) 
        splitter = QSplitter(Qt.Horizontal)
        self.subject_label = QLabel(f"Attendance for {self.subject_name} (Group: {self.group}) | {time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.subject_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.subject_label)

        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(960, 720)
        splitter.addWidget(self.video_label)

        self.student_list = QListWidget()
        self.student_list_frame = QFrame()
        vbox = QVBoxLayout(self.student_list_frame)
        vbox.addWidget(QLabel("Recognized Students"))
        vbox.addWidget(self.student_list)
        splitter.addWidget(self.student_list_frame)
        splitter.setSizes([700, 200])
        self.print_btn = QPushButton("Print List")
        self.print_btn.clicked.connect(self.print_students)
        vbox.addWidget(self.print_btn)
        main_layout.addWidget(splitter)
        self.setLayout(main_layout)

        self.setLayout(main_layout)
        self.resize(1400, 800)
    
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
    
    def prompt_for_info(self, frame):
        dialog = PromptDialog(frame)
        if dialog.exec_() == QDialog.Accepted:
            return dialog.get_selection()
        return None

    def add_student(self, name):
        if name not in self.students:
            self.students.add(name)
            self.student_list.addItem(name)
    
    def print_students(self):
        printer = QPrinter()
        dialog = QPrintDialog(printer, self)
        if dialog.exec_() == QPrintDialog.Accepted:
            text = f"Generated attendance report for {self.subject_name} (Group: {self.group})\n"
            text += f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            text += "\n------------------ Recognized students: ------------------\n"
            for idx, name in enumerate(self.students, start=1):
                text += f"{idx}. {name}\n"
            from PyQt5.QtGui import QTextDocument
            doc = QTextDocument()
            doc.setPlainText(text)
            doc.print_(printer)

