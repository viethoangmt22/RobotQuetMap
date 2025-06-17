from PyQt5.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QFileDialog, QHBoxLayout, QPushButton, QLabel, QLineEdit, QSlider, QMessageBox
from PyQt5.QtCore import QTimer, Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from math import pi
import threading

class LidarWindow(QMainWindow):
    def __init__(self, lidar):
        super().__init__()
        self.lidar = lidar
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Robot Control with Lidar Mapping")
        self.setGeometry(100, 100, 800, 600)
        self.setFocusPolicy(Qt.StrongFocus)

        # Tạo widget trung tâm và layout chính
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        vbox = QVBoxLayout(central_widget)

        # Tích hợp Matplotlib canvas từ figure của LidarData
        self.canvas = FigureCanvas(self.lidar.fig)
        vbox.addWidget(self.canvas)

        # Kết nối sự kiện nhấp chuột vào canvas
        self.canvas.mpl_connect('button_press_event', self.on_canvas_click)

        # Layout cho cài đặt kết nối: nhập địa chỉ IP ESP32 và nút kết nối
        conn_layout = QHBoxLayout()
        self.ip_label = QLabel("ESP32 IP:")
        self.ip_input = QLineEdit()
        self.ip_input.setPlaceholderText("Nhập địa chỉ IP")
        self.connect_btn = QPushButton("Kết nối")
        self.connect_btn.clicked.connect(self.reconnect_device)
        conn_layout.addWidget(self.ip_label)
        conn_layout.addWidget(self.ip_input)
        conn_layout.addWidget(self.connect_btn)
        vbox.addLayout(conn_layout)

        # Layout cho thanh trượt offset
        offset_layout = QHBoxLayout()
        self.offset_label = QLabel("Heading Offset (degrees):")
        self.offset_slider = QSlider(Qt.Horizontal)
        self.offset_slider.setRange(-180, 180)
        self.offset_slider.setValue(0)
        self.offset_value_label = QLabel("0")
        self.offset_slider.valueChanged.connect(self.change_offset)
        offset_layout.addWidget(self.offset_label)
        offset_layout.addWidget(self.offset_slider)
        offset_layout.addWidget(self.offset_value_label)
        vbox.addLayout(offset_layout)

        # --- Save / Load map buttons ---
        io_layout = QHBoxLayout()
        self.save_map_btn = QPushButton("Save Map…")
        self.save_map_btn.clicked.connect(self.on_save_map)
        self.load_map_btn = QPushButton("Load Map…")
        self.load_map_btn.clicked.connect(self.on_load_map)
        io_layout.addWidget(self.save_map_btn)
        io_layout.addWidget(self.load_map_btn)
        vbox.addLayout(io_layout)

        # Layout chứa các nút điều khiển di chuyển và nhãn trạng thái
        btn_layout = QHBoxLayout()
        self.forward_btn = QPushButton("Tiến")
        self.forward_btn.pressed.connect(lambda: self.lidar.send_command("forward"))
        self.forward_btn.released.connect(lambda: self.lidar.send_command("stop"))
        self.backward_btn = QPushButton("Lùi")
        self.backward_btn.pressed.connect(lambda: self.lidar.send_command("reverse"))
        self.backward_btn.released.connect(lambda: self.lidar.send_command("stop"))
        self.left_btn = QPushButton("Xoay trái")
        self.left_btn.pressed.connect(lambda: self.lidar.send_command("left"))
        self.left_btn.released.connect(lambda: self.lidar.send_command("stop"))
        self.right_btn = QPushButton("Xoay phải")
        self.right_btn.pressed.connect(lambda: self.lidar.send_command("right"))
        self.right_btn.released.connect(lambda: self.lidar.send_command("stop"))
        self.status_label = QLabel("Robot Distance: 0.00 cm")
        btn_layout.addWidget(self.forward_btn)
        btn_layout.addWidget(self.backward_btn)
        btn_layout.addWidget(self.left_btn)
        btn_layout.addWidget(self.right_btn)
        btn_layout.addWidget(self.status_label)
        vbox.addLayout(btn_layout)

        # Layout MOVE / ROTATE
        mr_layout = QHBoxLayout()
        mr_layout.addWidget(QLabel("Move (m):"))
        self.move_input = QLineEdit()
        self.move_input.setPlaceholderText("ví dụ 1.2")
        mr_layout.addWidget(self.move_input)
        self.move_btn = QPushButton("MOVE")
        self.move_btn.clicked.connect(self.on_move)
        mr_layout.addWidget(self.move_btn)
        mr_layout.addSpacing(20)
        mr_layout.addWidget(QLabel("Rotate (°):"))
        self.rotate_input = QLineEdit()
        self.rotate_input.setPlaceholderText("ví dụ -90")
        mr_layout.addWidget(self.rotate_input)
        self.rotate_btn = QPushButton("ROTATE")
        self.rotate_btn.clicked.connect(self.on_rotate)
        mr_layout.addWidget(self.rotate_btn)
        vbox.addLayout(mr_layout)

        # Thêm giao diện điều hướng (navigation UI)
        self.init_navigation_ui(vbox)

        # QTimer để cập nhật giao diện định kỳ
        self.timer = QTimer()
        self.timer.setInterval(100)
        self.timer.timeout.connect(self.update_gui)
        self.timer.start()

    def init_navigation_ui(self, vbox):
        # Tạo layout cho điều khiển di chuyển đến tọa độ
        nav_layout = QHBoxLayout()
        nav_layout.addWidget(QLabel("Target X (mm):"))
        self.target_x_edit = QLineEdit()
        self.target_x_edit.setPlaceholderText("Nhập target X (mm)")
        nav_layout.addWidget(self.target_x_edit)
        nav_layout.addWidget(QLabel("Target Y (mm):"))
        self.target_y_edit = QLineEdit()
        self.target_y_edit.setPlaceholderText("Nhập target Y (mm)")
        nav_layout.addWidget(self.target_y_edit)
        self.navigate_button = QPushButton("Di chuyển đến vị trí")
        self.navigate_button.clicked.connect(self.on_navigate_button_clicked)
        nav_layout.addWidget(self.navigate_button)
        vbox.addLayout(nav_layout)

    def on_canvas_click(self, event):
        # Kiểm tra xem nhấp chuột có nằm trong axes của bản đồ không
        if event.inaxes is not self.lidar.ax:
            return
        # Lấy tọa độ thế giới từ sự kiện nhấp chuột
        x_world, y_world = event.xdata, event.ydata
        if x_world is not None and y_world is not None:
            # Cập nhật tọa độ vào các trường nhập liệu
            self.target_x_edit.setText(f"{x_world:.2f}")
            self.target_y_edit.setText(f"{y_world:.2f}")

    def on_navigate_button_clicked(self):
        try:
            target_x = float(self.target_x_edit.text())
            target_y = float(self.target_y_edit.text())
        except ValueError:
            QMessageBox.warning(self, "Invalid Input", "Vui lòng nhập số hợp lệ cho tọa độ!")
            return
        # Khởi chạy quá trình điều hướng trên thread riêng
        nav_thread = threading.Thread(target=self.lidar.navigate_to_target, args=(target_x, target_y), daemon=True)
        nav_thread.start()
        self.status_label.setText(f"Navigating to: ({target_x:.2f}, {target_y:.2f}) mm")

    def reconnect_device(self):
        new_ip = self.ip_input.text().strip()
        if new_ip:
            success = self.lidar.reconnect(new_ip)
            if success:
                self.status_label.setText(f"Kết nối thành công tới {new_ip}")
            else:
                self.status_label.setText("Kết nối thất bại. Kiểm tra lại địa chỉ IP.")

    def update_gui(self):
        if not self.lidar.plot_queue.empty():
            while not self.lidar.plot_queue.empty():
                self.lidar.plot_queue.get()
            self.lidar._plot_map()
            self.canvas.draw()
        self.status_label.setText(f"Robot Distance: {self.lidar.robot_distance:.2f} cm")

    def keyPressEvent(self, event):
        if event.isAutoRepeat():
            return super().keyPressEvent(event)
        key = event.key()
        if key == Qt.Key_Up:
            self.lidar.send_command("forward")
        elif key == Qt.Key_Down:
            self.lidar.send_command("reverse")
        elif key == Qt.Key_Left:
            self.lidar.send_command("left")
        elif key == Qt.Key_Right:
            self.lidar.send_command("right")
        else:
            super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        if event.isAutoRepeat():
            return super().keyReleaseEvent(event)
        key = event.key()
        if key in (Qt.Key_Up, Qt.Key_Down, Qt.Key_Left, Qt.Key_Right):
            self.lidar.send_command("stop")
        else:
            super().keyReleaseEvent(event)

    def on_move(self):
        text = self.move_input.text().strip()
        try:
            d = float(text)
        except ValueError:
            QMessageBox.warning(self, "Invalid Input", "Distance phải là số (m) hợp lệ!")
            return
        cmd = f"MOVE {d}"
        self.lidar.send_command(cmd)
        self.status_label.setText(f"Sent: {cmd}")

    def on_rotate(self):
        text = self.rotate_input.text().strip()
        try:
            deg = float(text)
        except ValueError:
            QMessageBox.warning(self, "Invalid Input", "Angle phải là số (°) hợp lệ!")
            return
        rad = deg * pi / 180.0
        cmd = f"ROTATE {rad}"
        self.lidar.send_command(cmd)
        self.status_label.setText(f"Sent: ROTATE {deg:.1f}° → {rad:.3f}rad")

    def closeEvent(self, event):
        self.lidar.cleanup()
        event.accept()


    def change_offset(self, value):
        self.offset_value_label.setText(str(value))
        offset_rad = value * (pi / 180)
        self.lidar.heading_offset = offset_rad
        self.lidar.reset_map()
        self.lidar._plot_map()
        self.canvas.draw()

    def on_save_map(self):
        # Mở file dialog để chọn nơi lưu
        fname, _ = QFileDialog.getSaveFileName(self, "Save map as…", "", "NumPy compressed (*.npz)")
        if not fname:
            return
        if not fname.lower().endswith(".npz"):
            fname += ".npz"
        try:
            self.lidar.save_map(fname)
            QMessageBox.information(self, "Saved", f"Map saved to:\n{fname}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Không thể lưu map:\n{e}")

    def on_load_map(self):
        # Mở file dialog để chọn file map
        fname, _ = QFileDialog.getOpenFileName(self, "Load map from…", "", "NumPy compressed (*.npz)")
        if not fname:
            return
        try:
            self.lidar.load_map(fname)
            # Khi load xong, vẽ lại và cập nhật canvas
            self.lidar._plot_map()
            self.canvas.draw()
            QMessageBox.information(self, "Loaded", f"Map loaded from:\n{fname}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Không thể load map:\n{e}")
