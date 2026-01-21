from isaacgym import gymapi
import numpy as np
from PyQt5.QtWidgets import QWidget, QLabel, QApplication
from PyQt5.QtGui import QPixmap, QImage, QPainter
from PyQt5.QtCore import Qt, QTimer
import imageio
from io import BytesIO
import sys

class PhyPlanUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PhyPlan")
        # self.gym = gym
        # self.sim = sim
        # self.env = env

        # Labels for 3 views
        self.view1 = QLabel("")
        self.view2 = QLabel("")
        self.view3 = QLabel("")
        self.view4 = QLabel("")
        self.view5 = QLabel("Contexts: __/__")
        self.view6 = QLabel("Best Pred. Reward: ____")
        self.view7 = QLabel("Trial: __/__")
        self.view8 = QLabel("Ideal Reward: ____")
        self.view9 = QLabel("Tool Combination: ____")
        self.view10 = QLabel("Best Chosen Action: ____")
        self.view1.setParent(self)
        self.view2.setParent(self)
        self.view3.setParent(self)
        self.view4.setParent(self)
        self.view5.setParent(self)
        self.view6.setParent(self)
        self.view7.setParent(self)
        self.view8.setParent(self)
        self.view9.setParent(self)
        self.view10.setParent(self)
        self.view1.setFixedSize(480, 480)
        self.view2.setFixedSize(1280, 160)
        self.view3.setFixedSize(400, 400)
        self.view4.setFixedSize(400, 400)
        self.view5.setFixedSize(400, 40)
        self.view6.setFixedSize(400, 40)
        self.view7.setFixedSize(400, 40)
        self.view8.setFixedSize(400, 40)
        self.view9.setFixedSize(400, 40)
        self.view10.setFixedSize(400, 40)
        self.view1.move(30, 300)
        self.view2.move(510, 70)
        self.view3.move(682, 380)
        self.view4.move(1398, 380)
        self.view5.move(682, 852)
        self.view6.move(682, 902)
        self.view7.move(1398, 852)
        self.view8.move(1398, 902)
        self.view9.move(682, 952)
        self.view10.move(1398, 952)
        self.setGeometry(0, 0, 1920, 1080)
        self.label_dict = {"IsaacGym": self.view4, 
                                            "PINN": self.view3, 
                                            "MCTS": self.view2, 
                                            "GFN": self.view1, 
                                            "Context": self.view5,
                                            "PredRew": self.view6,
                                            "Trial": self.view7,
                                            "IdealRew": self.view8,
                                            "Tool": self.view9,
                                            "ChosAct": self.view10}
        self.setStyleSheet("background-color: white;")
        self.view5.setStyleSheet("color: black;")
        self.view6.setStyleSheet("color: black;")
        self.view7.setStyleSheet("color: black;")
        self.view8.setStyleSheet("color: black;")
        self.view9.setStyleSheet("color: black;")
        self.view10.setStyleSheet("color: black;")
        self.view1.show()
        self.view2.show()
        self.view3.show()
        self.view4.show()
        self.view5.show()
        self.view6.show()
        self.view7.show()
        self.view8.show()
        self.view9.show()
        self.view10.show()
        self.show()

    # def _update_label(self, label, img_buf):
    #     img = QImage()
    #     img.loadFromData(img_buf.getvalue())
    #     pixmap = QPixmap.fromImage(img).scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
    #     pixmap_filled = QPixmap(label.size())
    #     pixmap_filled.fill(Qt.transparent)
    #     painter = QPainter(pixmap_filled)
    #     x_offset = (label.width() - pixmap.width()) // 2
    #     y_offset = (label.height() - pixmap.height()) // 2
    #     painter.drawPixmap(x_offset, y_offset, pixmap)
    #     painter.end()
    #     label.setPixmap(pixmap_filled)
    #     label.show()

def start_gui(queue, training_process):
    app = QApplication([])
    window = PhyPlanUI()
    window.show()

    def poll_queue():
        while not queue.empty():
            label_name, img_buf = queue.get()
            label = window.label_dict[label_name]
            if isinstance(img_buf, str):
                label.setText(img_buf)
            else:
                img = QImage()
                img.loadFromData(img_buf.getvalue())
                pixmap = QPixmap.fromImage(img).scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                pixmap_filled = QPixmap(label.size())
                pixmap_filled.fill(Qt.transparent)
                painter = QPainter(pixmap_filled)
                x_offset = (label.width() - pixmap.width()) // 2
                y_offset = (label.height() - pixmap.height()) // 2
                painter.drawPixmap(x_offset, y_offset, pixmap)
                painter.end()
                label.setPixmap(pixmap_filled)
                label.show()
    def check_training_finished():
        if not(training_process.is_alive()):
            app.quit()
    timer = QTimer()
    timer.timeout.connect(poll_queue)
    timer.start(100)
    watchdog = QTimer()
    watchdog.timeout.connect(check_training_finished)
    watchdog.start(500)

    sys.exit(app.exec_())