import sys
import socket
import threading
import queue
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from math import pi, cos, sin, atan2, sqrt
from PyQt5.QtWidgets import QApplication
import logging
import heapq
import scipy.optimize as opt  # Dùng SciPy cho pose-graph optimization
from scipy.ndimage import distance_transform_edt

from lidar_ui import LidarWindow  # Nhập giao diện từ file lidar_ui.py

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

def start_navigation(lidar, target_x, target_y):
    nav_thread = threading.Thread(target=lidar.navigate_to_target, args=(target_x, target_y), daemon=True)
    nav_thread.start()

def heuristic(a, b):
    # Sử dụng khoảng cách Euclid
    return sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

def a_star_search(grid, start, goal):
    rows, cols = grid.shape
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    cost_so_far = {start: 0}

    while open_set:
        current_priority, current = heapq.heappop(open_set)
        if current == goal:
            break
        x, y = current
        # 8 hướng di chuyển
        neighbors = [(x-1, y), (x+1, y), (x, y-1), (x, y+1),
                     (x-1, y-1), (x-1, y+1), (x+1, y-1), (x+1, y+1)]
        for next in neighbors:
            nx, ny = next
            if 0 <= nx < cols and 0 <= ny < rows and grid[ny, nx] != 255:
                # Sử dụng √2 cho bước chéo, 1 cho bước ngang/dọc
                step_cost = sqrt(2) if (nx - x != 0 and ny - y != 0) else 1
                new_cost = cost_so_far[current] + step_cost
                if next not in cost_so_far or new_cost < cost_so_far[next]:
                    cost_so_far[next] = new_cost
                    priority = new_cost + heuristic(goal, next)
                    heapq.heappush(open_set, (priority, next))
                    came_from[next] = current

    path = []
    current = goal
    while current != start:
        path.append(current)
        current = came_from.get(current)
        if current is None:
            return []  # Không tìm được đường đi
    path.append(start)
    path.reverse()
    return path

# ------------------------------
# Hàm Bresenham cho việc tìm các ô trong grid theo đường ray
# ------------------------------
def bresenham_line(x0, y0, x1, y1):
    """
    Trả về danh sách các ô (x, y) theo thuật toán Bresenham từ (x0, y0) đến (x1, y1)
    """
    points = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    x, y = x0, y0
    sx = -1 if x0 > x1 else 1
    sy = -1 if y0 > y1 else 1
    if dx > dy:
        err = dx / 2.0
        while x != x1:
            points.append((x, y))
            err -= dy
            if err < 0:
                y += sy
                err += dx
            x += sx
    else:
        err = dy / 2.0
        while y != y1:
            points.append((x, y))
            err -= dx
            if err < 0:
                x += sx
                err += dy
            y += sy
    points.append((x1, y1))
    return points

def is_collision_free(p1, p2, grid):
    """
    Kiểm tra đường thẳng từ p1 đến p2 trong occupancy grid có va chạm với chướng ngại vật hay không.
    p1, p2: tuple (x, y) của chỉ số ô (grid coordinates)
    grid: occupancy grid với 0 là free, 255 là obstacle.
    Giả sử các ô có giá trị 127 (unknown) cũng được coi là free.
    """
    cells = bresenham_line(p1[0], p1[1], p2[0], p2[1])
    for cell in cells:
        x, y = cell
        # Nếu ô có giá trị 255 (occupied) thì coi là va chạm
        if grid[y, x] == 255:
            return False
    return True

def smooth_path(path, grid):
    if not path or len(path) < 2:
        return path
    smoothed = [path[0]]
    i = 0
    while i < len(path) - 1:
        for j in range(len(path) - 1, i, -1):
            if is_collision_free(smoothed[-1], path[j], grid):
                smoothed.append(path[j])
                i = j
                break
        else:
            # Nếu không tìm được điểm nào xa hơn, thêm điểm tiếp theo
            smoothed.append(path[i + 1])
            i += 1
    return smoothed
# ------------------------------
# Lớp EKF SLAM
# ------------------------------
class EKFSLAM:
    def __init__(self, initial_pose):
        # State vector gồm: [x, y, theta]
        self.state = np.array(initial_pose, dtype=float)
        # Ma trận hiệp phương sai khởi tạo
        self.cov = np.eye(3) * 0.1

    def predict(self, delta_s, delta_theta, motion_cov):
        theta = self.state[2]
        dx = delta_s * cos(theta + delta_theta / 2)
        dy = delta_s * sin(theta + delta_theta / 2)
        self.state[0] += dx
        self.state[1] += dy
        self.state[2] += delta_theta
        self.state[2] = atan2(sin(self.state[2]), cos(self.state[2]))
        F = np.array([
            [1, 0, -delta_s * sin(theta + delta_theta / 2)],
            [0, 1,  delta_s * cos(theta + delta_theta / 2)],
            [0, 0, 1]
        ])
        self.cov = F @ self.cov @ F.T + motion_cov

    def update(self, z, landmark_pos, measurement_cov):
        dx = landmark_pos[0] - self.state[0]
        dy = landmark_pos[1] - self.state[1]
        q = dx**2 + dy**2
        expected_range = sqrt(q)
        expected_bearing = atan2(dy, dx) - self.state[2]
        z_hat = np.array([expected_range, expected_bearing])
        y = z - z_hat
        y[1] = atan2(sin(y[1]), cos(y[1]))
        H = np.array([
            [-dx / expected_range, -dy / expected_range, 0],
            [dy / q,              -dx / q,             -1]
        ])
        S = H @ self.cov @ H.T + measurement_cov
        K = self.cov @ H.T @ np.linalg.inv(S)
        self.state = self.state + K @ y
        self.cov = (np.eye(3) - K @ H) @ self.cov

# ------------------------------
# Lớp xử lý dữ liệu LiDAR, odometry và xây dựng bản đồ toàn cục
# ------------------------------
class LidarData:
    DATA_LENGTH = 7
    MAX_DISTANCE = 3000  # mm
    MIN_DISTANCE = 50    # mm
    MAX_DATA_SIZE = 180  # Tích lũy nhiều điểm trước khi vẽ
    NEIGHBOR_RADIUS = 48
    MIN_NEIGHBORS = 4
    GRID_SIZE = 50      # mm

    def __init__(self, host='192.168.100.148', port=80, neighbor_radius=48, min_neighbors=4):
        self.host = host
        self.port = port
        self.sock = None
        self.data = {
            'angles': [], 'distances': [], 'speed': [],
            'x_coords': [], 'y_coords': []
        }
        self.grid = None
        self.robot_distance = 0.0  # Tổng quãng đường đã đi (mm)
        self.data_queue = queue.Queue()
        self.command_queue = queue.Queue()
        self.plot_queue = queue.Queue()

        self.NEIGHBOR_RADIUS = neighbor_radius
        self.MIN_NEIGHBORS = min_neighbors

        self.data_lock = threading.Lock()
        self.data_event = threading.Event()
        self.running = True

        self.mapping_enabled = True
        # Các thuộc tính khác
        self.pose_x = 0.0
        self.pose_y = 0.0
        self.pose_theta = 0.0
        # Khởi tạo ekf_slam ngay từ đầu
        self.ekf_slam = EKFSLAM([self.pose_x, self.pose_y, self.pose_theta])

        # ---- Smart keyframe parameters ----
        # Ngưỡng dịch chuyển (mm) để cập nhật keyframe
        self.keyframe_trans_thresh = 500.0    # ví dụ: 0.5 m
        # Ngưỡng xoay (rad) để cập nhật keyframe
        self.keyframe_rot_thresh   = 0.8  # ~10°
        # Lưu pose (x, y, theta) của keyframe gần nhất
        self.last_keyframe_pose = np.array([self.pose_x, self.pose_y, self.pose_theta])


        # Thông số encoder
        self.wheel_diameter = 70  # mm
        self.ppr = 500            # pulses per revolution
        self.pi = 3.1416
        self.wheel_circumference = self.wheel_diameter * self.pi  # mm

        # Biến định vị (pose) của robot: [x, y, theta]
        self.pose_x = 0.0
        self.pose_y = 0.0
        self.pose_theta = 0.0
        self.heading_offset = 0.0  # Offset góc ban đầu là 0 (radian)

        self.last_encoder_left = None
        self.last_encoder_right = None
        self.wheel_base = 105  # mm

        # Khởi tạo figure cho matplotlib
        self.fig, self.ax = plt.subplots()
        self.ax.set_title("Global Map")
        self.ax.grid(True)

        # Khởi tạo global map (occupancy grid)
        self.map_size_mm = 10000  # mm, tức là 10m x 10m
        self.global_map_dim = int(self.map_size_mm / self.GRID_SIZE)
        self.global_map = np.full((self.global_map_dim, self.global_map_dim), 127, dtype=np.uint8)

        self.pose_lock = threading.Lock()

        # --- Pose-graph SLAM tự triển khai ---
        # Danh sách keyframe pose (x [m], y [m], theta [rad])
        self.keyframes = []
        # Danh sách constraint: (i, j, measurement [dx, dy, dtheta], information_matrix 3x3)
        self.edges = []
        # Thêm keyframe đầu tiên
        self._add_keyframe(self.pose_x, self.pose_y, self.pose_theta)

        # Queue dành cho IPC thread
        self.icp_queue = queue.Queue()
        self.keyframe_points = None
        # Thread riêng để chạy ICP
        self.icp_thread = threading.Thread(target=self._icp_worker, daemon=True)
        self.icp_thread.start()
        if not self._connect_wifi():
            logging.error("Kết nối ban đầu không thành công. Vui lòng nhập IP ESP32 thủ công qua giao diện.")
        else:
            self.send_command("RESET_ENCODERS")
            logging.info("Đã gửi lệnh reset encoders")

        self.command_thread = threading.Thread(target=self._handle_commands, daemon=True)
        self.command_thread.start()
        self.process_thread = threading.Thread(target=self.process_data, daemon=True)
        self.process_thread.start()

    def _add_keyframe(self, x_mm, y_mm, theta):
        x = x_mm / 1000.0
        y = y_mm / 1000.0
        # Chỉ thêm edge nếu đã có keyframe trước
        if self.keyframes:
            idx_prev = len(self.keyframes) - 1
            x0, y0, t0 = self.keyframes[idx_prev]
            dx = x - x0
            dy = y - y0
            dtheta = atan2(sin(theta - t0), cos(theta - t0))
            info = np.eye(3) * 100  # thông tin (có thể tinh chỉnh)
            self.edges.append((idx_prev, idx_prev + 1, np.array([dx, dy, dtheta]), info))
        # Thêm new keyframe
        self.keyframes.append(np.array([x, y, theta]))

    def _optimize_pose_graph(self):
        def residuals(vars):
            res = []
            for i, j, meas, info in self.edges:
                xi = vars[3 * i:3 * i + 3]
                xj = vars[3 * j:3 * j + 3]
                # relative pose from i to j
                dx_ij = xj - xi
                dx_ij[2] = atan2(sin(dx_ij[2]), cos(dx_ij[2]))
                err = dx_ij - meas
                # weight by sqrt of information
                W = np.linalg.cholesky(info)
                res.extend((W @ err).tolist())
            return np.array(res)

        # Thêm prior để cố định node 0
        def full_res(vars):
            r = residuals(vars)
            prior = vars[0:3] - self.keyframes[0]
            W0 = np.eye(3) * 1000
            pr = (W0 @ prior).tolist()
            return np.hstack([r, pr])

        # Biến ban đầu
        vars0 = np.hstack(self.keyframes)
        sol = opt.least_squares(full_res, vars0, verbose=0)
        vars_opt = sol.x
        # Cập nhật keyframes và state
        N = len(self.keyframes)
        for k in range(N):
            self.keyframes[k] = vars_opt[3 * k:3 * k + 3]
        x, y, th = self.keyframes[-1]
        self.pose_x, self.pose_y, self.pose_theta = x * 1000, y * 1000, th
    def set_heading(self, theta):
        with self.pose_lock:
            self.pose_theta = theta
            if hasattr(self, 'ekf_slam'):
                self.ekf_slam.state[2] = theta
        logging.info(f"Heading adjusted to θ={theta:.3f} rad")

    def _connect_wifi(self):
        try:
            logging.info("Attempting to connect to %s:%s...", self.host, self.port)
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.settimeout(10)
            self.sock.connect((self.host, self.port))
            self.sock.setblocking(False)
            logging.info("Connected to %s:%s", self.host, self.port)
            return True
        except (socket.error, socket.timeout) as e:
            logging.error("WiFi connection error: %s", e)
            return False

    def reconnect(self, new_host):
        self.host = new_host
        if self.sock:
            try:
                self.sock.close()
            except Exception as e:
                logging.error("Error closing socket: %s", e)
            self.sock = None
        if not self._connect_wifi():
            logging.error("Reconnection to %s failed.", new_host)
            return False
        else:
            logging.info("Reconnected successfully to %s.", new_host)
            return True

    # --------------------------
    # Các hàm xử lý dữ liệu cảm biến
    # --------------------------
    def _filter_data(self, angles, distances):
        angles_np = np.array(angles)
        distances_np = np.array(distances)
        mask = (distances_np >= self.MIN_DISTANCE) & (distances_np <= self.MAX_DISTANCE)
        return angles_np[mask].tolist(), distances_np[mask].tolist()

    def _to_cartesian(self, angles, distances):
        angles_np = np.array(angles)
        distances_np = np.array(distances)
        x_coords = distances_np * np.cos(angles_np)
        y_coords = distances_np * np.sin(angles_np)
        return x_coords.tolist(), y_coords.tolist()

    def _to_global_coordinates(self, angles, distances):
        angles_np = np.array(angles) + self.heading_offset
        distances_np = np.array(distances)
        global_x = self.pose_x + distances_np * np.cos(self.pose_theta + angles_np)
        global_y = self.pose_y + distances_np * np.sin(self.pose_theta + angles_np)
        return global_x.tolist(), global_y.tolist()

    def _remove_outliers(self, x_coords, y_coords):
        if len(x_coords) < self.MIN_NEIGHBORS:
            return [], []
        points = np.array(list(zip(x_coords, y_coords)))
        tree = cKDTree(points)
        distances, _ = tree.query(points, k=self.MIN_NEIGHBORS)
        distances_to_kth_neighbor = distances[:, self.MIN_NEIGHBORS - 1]
        mean_distance = np.mean(distances_to_kth_neighbor)
        std_distance = np.std(distances_to_kth_neighbor)
        dynamic_radius = mean_distance + 2 * std_distance
        neighbor_counts = tree.query_ball_point(points, r=dynamic_radius, return_length=True)
        mask = neighbor_counts >= self.MIN_NEIGHBORS
        filtered_points = points[mask]
        distances_from_origin = np.linalg.norm(filtered_points, axis=1)
        mean_dist = np.mean(distances_from_origin)
        std_dist = np.std(distances_from_origin)
        stat_mask = distances_from_origin <= (mean_dist + 2 * std_dist)
        final_points = filtered_points[stat_mask]
        return final_points[:, 0].tolist(), final_points[:, 1].tolist()

    def voxel_downsample(self, x_coords, y_coords, voxel_size=20):
        voxel_dict = {}
        for x, y in zip(x_coords, y_coords):
            voxel_x = int(x // voxel_size)
            voxel_y = int(y // voxel_size)
            key = (voxel_x, voxel_y)
            if key not in voxel_dict:
                voxel_dict[key] = []
            voxel_dict[key].append((x, y))
        downsampled_x = []
        downsampled_y = []
        for points in voxel_dict.values():
            avg_x = sum(p[0] for p in points) / len(points)
            avg_y = sum(p[1] for p in points) / len(points)
            downsampled_x.append(avg_x)
            downsampled_y.append(avg_y)
        return downsampled_x, downsampled_y

    # --------------------------
    # Cập nhật global map bằng thuật toán map merging
    # --------------------------
    def update_global_map(self, scan_x, scan_y):
        robot_grid_x = int((self.pose_x + self.map_size_mm/2) / self.GRID_SIZE)
        robot_grid_y = int((self.pose_y + self.map_size_mm/2) / self.GRID_SIZE)
        for x, y in zip(scan_x, scan_y):
            meas_grid_x = int((x + self.map_size_mm/2) / self.GRID_SIZE)
            meas_grid_y = int((y + self.map_size_mm/2) / self.GRID_SIZE)
            line_cells = bresenham_line(robot_grid_x, robot_grid_y, meas_grid_x, meas_grid_y)
            for cell in line_cells[:-1]:
                cx, cy = cell
                if 0 <= cx < self.global_map_dim and 0 <= cy < self.global_map_dim:
                    self.global_map[cy, cx] = 0
            if 0 <= meas_grid_x < self.global_map_dim and 0 <= meas_grid_y < self.global_map_dim:
                self.global_map[meas_grid_y, meas_grid_x] = 255

    def _plot_map(self):
        """
        Vẽ:
         - always: bản đồ đã load (static map) & pose robot
         - khi mapping_enabled=True: merge thêm scan mới vào map
        """
        # 1) Xóa axes cũ
        self.ax.clear()

        # 2) Vẽ background map (static hoặc đã merge trước đó)
        extent = (
            -self.map_size_mm / 2, self.map_size_mm / 2,
            -self.map_size_mm / 2, self.map_size_mm / 2
        )
        # vmin, vmax cố định để grayscale không bị scale lại
        self.ax.imshow(
            self.global_map,
            cmap='gray',
            origin='lower',
            extent=extent,
            vmin=0, vmax=255
        )

        # 3) Nếu vẫn đang mapping (chưa load map) thì merge scan mới
        if self.mapping_enabled:
            angles, distances = self._filter_data(self.data['angles'], self.data['distances'])
            if angles:
                gx, gy = self._to_global_coordinates(angles, distances)
                fx, fy = self._remove_outliers(gx, gy)
                dx, dy = self.voxel_downsample(fx, fy, voxel_size=self.GRID_SIZE)
                with self.data_lock:
                    self.data['x_coords'].extend(dx)
                    self.data['y_coords'].extend(dy)
                    self.data['angles'].clear()
                    self.data['distances'].clear()
                # Merge vào self.global_map
                self.update_global_map(dx, dy)

        # 4) Vẽ robot & heading (luôn thực hiện)
        self.ax.plot(self.pose_x, self.pose_y, 'ro', markersize=8)
        arrow_len = 500  # mm
        ex = self.pose_x + arrow_len * cos(self.pose_theta)
        ey = self.pose_y + arrow_len * sin(self.pose_theta)
        self.ax.plot([self.pose_x, ex], [self.pose_y, ey], 'r-', linewidth=2)

        # 5) Tiêu đề, lưới
        mode = "Mapping" if self.mapping_enabled else "Localization"
        self.ax.set_title(f"Mode: {mode} — Dist: {self.robot_distance:.1f} mm")
        self.ax.grid(True)

        # 6) Vẽ lên màn hình
        self.fig.canvas.draw()

    def process_data(self):
        while self.running:
            if self.data_event.wait(timeout=0.1):
                while not self.data_queue.empty():
                    # LUÔN enqueue để _plot_map được gọi dù đã load map
                    self.plot_queue.put(True)
                    try:
                        self.data_queue.get_nowait()
                    except queue.Empty:
                        break
                self.data_event.clear()

    def update_data(self):
        buffer = ""
        # Biến đếm vòng quét để khởi động ICP sau khoảng thời gian ổn định
        if not hasattr(self, 'scan_count'):
            self.scan_count = 0

        while self.running:
            if self.sock is None:
                time.sleep(0.1)
                continue
            try:
                data = self.sock.recv(4096)
                if not data:
                    logging.warning("Connection closed by server.")
                    self.sock = None
                    continue
                data = data.decode('utf-8', errors='ignore')
                buffer += data
            except BlockingIOError:
                time.sleep(0.01)
                continue
            except Exception as e:
                logging.error("Exception in update_data: %s", e)
                self.sock = None
                continue
            while '\n' in buffer:
                line, buffer = buffer.split('\n', 1)
                logging.info("Raw data received: %s", line)
                sensor_data = line.strip().split('\t')
                if len(sensor_data) != 10:
                    logging.warning("Sensor data length mismatch: %s", sensor_data)
                    continue
                try:
                    base_angle = int(sensor_data[0])
                    speed = int(sensor_data[1])
                    distances = np.array(sensor_data[2:6], dtype=float)
                    encoder_count = int(sensor_data[7].strip())
                    encoder_count2 = int(sensor_data[8].strip())
                    gyro_z = float(sensor_data[9].strip())
                    #logging.info("Raw data: encoder_count=%d, encoder_count2=%d, gyroZ=%.2f rad/s",
                    #             encoder_count, encoder_count2, gyro_z)
                except ValueError as e:
                    logging.error("Error parsing sensor data %s: %s", sensor_data, e)
                    continue

                # Xử lý odometry như cũ...
                if self.last_encoder_left is None:
                    self.last_encoder_left = encoder_count
                    self.last_encoder_right = encoder_count2
                    self.initial_encoder_left = encoder_count
                    self.initial_encoder_right = encoder_count2
                    self.robot_distance = 0.0
                    self.last_time = time.time()
                    logging.info("Initialized encoders: left=%d, right=%d", encoder_count, encoder_count2)
                    continue  # ⬅️ THÊM DÒNG NÀY để bỏ qua bước cập nhật vận tốc lần đầu

                else:
                    current_time = time.time()
                    delta_t = current_time - self.last_time
                    self.last_time = current_time
                    delta_left = (encoder_count - self.last_encoder_left) * self.wheel_circumference / self.ppr
                    delta_right = (encoder_count2 - self.last_encoder_right) * self.wheel_circumference / self.ppr
                    delta_s = (delta_left + delta_right) / 2.0
                    delta_theta_enc = (delta_right - delta_left) / self.wheel_base
                    delta_theta_gyro = gyro_z * delta_t
                    #logging.info(
                    #    "Odometry: delta_left=%.2f mm, delta_right=%.2f mm, delta_theta_enc=%.4f rad, delta_theta_gyro=%.4f rad",
                    #    delta_left, delta_right, delta_theta_enc, delta_theta_gyro)
                    # Giả sử dùng trọng số hiện tại
                    delta_theta = 1 * delta_theta_gyro + 0* delta_theta_enc
                    if not hasattr(self, 'ekf_slam'):
                        self.ekf_slam = EKFSLAM([self.pose_x, self.pose_y, self.pose_theta])
                    motion_cov = np.diag([1e-3, 1e-3, 3e-3])
                    self.ekf_slam.predict(delta_s, delta_theta, motion_cov)
                    self.pose_x, self.pose_y, self.pose_theta = self.ekf_slam.state
                    #logging.info("Pose updated (EKF): x=%.2f mm, y=%.2f mm, theta=%.4f rad",
                    #            self.pose_x, self.pose_y, self.pose_theta)
                    self.last_encoder_left = encoder_count
                    self.last_encoder_right = encoder_count2

                self.robot_distance = (((encoder_count - self.initial_encoder_left) +
                                        (encoder_count2 - self.initial_encoder_right)) / 2
                                       * self.wheel_circumference / self.ppr)
                #logging.info("Total distance traveled: %.2f mm", self.robot_distance)

                # --- Xử lý dữ liệu LiDAR (Motion Distortion Correction) ---
                n_points = 4
                SCAN_DURATION = 0.01
                dt = SCAN_DURATION / n_points

                # Tính vận tốc tuyến tính và góc
                current_time = time.time()
                delta_t = current_time - self.last_time
                self.last_time = current_time

                # Sử dụng delta_s đã tính từ encoder
                v = delta_s / delta_t if delta_t > 0 else 0  # Vận tốc tuyến tính (mm/s)
                omega = gyro_z  # Vận tốc góc (rad/s)

                # Pose tại thời điểm bắt đầu quét
                x0, y0, theta0 = self.ekf_slam.state

                # Thời gian tương đối cho mỗi điểm
                t_i = np.arange(n_points) * dt

                # Ước lượng pose tại mỗi thời điểm t_i
                theta_i = theta0 + omega * t_i
                x_i = x0 + v * t_i * np.cos(theta0 + omega * t_i / 2)
                y_i = y0 + v * t_i * np.sin(theta0 + omega * t_i / 2)

                # Góc ban đầu của các điểm
                raw_angles = (np.arange(n_points) + base_angle) * (pi / 180)
                distances_np = np.array(distances)  # Chuyển distances thành NumPy array nếu chưa phải

                # Lọc các điểm hợp lệ
                valid_mask = (distances_np >= self.MIN_DISTANCE) & (distances_np <= self.MAX_DISTANCE)
                valid_angles = raw_angles[valid_mask]
                valid_distances = distances_np[valid_mask]
                valid_x_i = x_i[valid_mask]
                valid_y_i = y_i[valid_mask]
                valid_theta_i = theta_i[valid_mask]

                # Tính tọa độ toàn cục cho các điểm hợp lệ
                global_x = valid_x_i + valid_distances * np.cos(valid_theta_i + valid_angles)
                global_y = valid_y_i + valid_distances * np.sin(valid_theta_i + valid_angles)

                # Lưu dữ liệu vào self.data để xử lý tiếp
                with self.data_lock:
                    self.data['angles'].extend(valid_angles.tolist())
                    self.data['distances'].extend(valid_distances.tolist())
                    self.data['speed'].extend([speed] * len(valid_angles))
                    self.data['x_coords'].extend(global_x.tolist())
                    self.data['y_coords'].extend(global_y.tolist())

                    # Kiểm tra kích thước dữ liệu
                    if len(self.data['angles']) >= self.MAX_DATA_SIZE:
                        self.data_queue.put(True)
                        self.data_event.set()
                    if len(self.data['angles']) > 500:
                        sample_size = min(200, len(self.data['angles']))  # tránh out of range
                        indices = np.random.choice(len(self.data['angles']), sample_size, replace=False)
                        self.data['angles'] = [self.data['angles'][i] for i in indices]
                        self.data['distances'] = [self.data['distances'][i] for i in indices]
                        self.data['speed'] = [self.data['speed'][i] for i in indices]

                # --- KẾT THÚC xử lý LiDAR ---

                if valid_angles.size > 0:
                    for i in range(len(valid_angles)):
                        meas_range = valid_distances[i]
                        meas_bearing = valid_angles[i] - self.pose_theta
                        z = np.array([meas_range, meas_bearing])
                        lx = self.ekf_slam.state[0] + meas_range * cos(self.ekf_slam.state[2] + meas_bearing)
                        ly = self.ekf_slam.state[1] + meas_range * sin(self.ekf_slam.state[2] + meas_bearing)
                        landmark_pos = np.array([lx, ly])
                        measurement_cov = np.diag([2e-2, 3e-2])
                        self.ekf_slam.update(z, landmark_pos, measurement_cov)
                with self.data_lock:
                    self.data['angles'].extend(valid_angles.tolist())
                    self.data['distances'].extend(valid_distances.tolist())
                    self.data['speed'].extend([speed] * int(np.sum(valid_mask)))
                    if len(self.data['angles']) >= self.MAX_DATA_SIZE:
                        self.data_queue.put(True)
                        self.data_event.set()
                    if len(self.data['angles']) > 500:
                        indices = np.random.choice(len(self.data['angles']), 200, replace=False)
                        self.data['angles'] = [self.data['angles'][i] for i in indices]
                        self.data['distances'] = [self.data['distances'][i] for i in indices]
                        self.data['speed'] = [self.data['speed'][i] for i in indices]

                # --- Xử lý Scan Matching ---
                global_x, global_y = self._to_global_coordinates(valid_angles.tolist(), valid_distances.tolist())
                current_scan_points = np.array(list(zip(global_x, global_y)))

                # Tăng biến đếm vòng quét
                self.scan_count += 1

                # Chỉ thực hiện ICP sau một số vòng quét đầu tiên (ví dụ sau 5 quét) và với số điểm hiện có
                if self.scan_count > 5 and current_scan_points.shape[0] > 0:
                    self.icp_queue.put((self.ekf_slam.state.copy(), current_scan_points.copy()))
                    if hasattr(self, 'prev_scan_points') and self.prev_scan_points is not None and \
                            self.prev_scan_points.shape[0] > 0:
                        R_icp, t_icp = self.icp(self.prev_scan_points, current_scan_points)
                        dTheta_icp = atan2(R_icp[1, 0], R_icp[0, 0])
                        # Nếu kết quả ICP hợp lý (vd: dịch chuyển không vượt quá ngưỡng)
                        if abs(t_icp[0, 0]) < 100 and abs(t_icp[1, 0]) < 100 and abs(dTheta_icp) < 0.5:
                            # Cập nhật pose theo một hệ số nhẹ để làm mịn hiệu chỉnh
                            alpha = 0.2
                            self.pose_x = (1 - alpha) * self.pose_x + alpha * (self.pose_x + t_icp[0, 0])
                            self.pose_y = (1 - alpha) * self.pose_y + alpha * (self.pose_y + t_icp[1, 0])
                            self.pose_theta += alpha * dTheta_icp
                            self.pose_theta = atan2(sin(self.pose_theta), cos(self.pose_theta))
                            #logging.info("Scan Matching Update: dx=%.2f mm, dy=%.2f mm, dtheta=%.4f rad", t_icp[0, 0], t_icp[1, 0], dTheta_icp)
                    # Cập nhật quét tham chiếu
                    self.prev_scan_points = current_scan_points.copy()

    def _icp_worker(self):
        """
        Nhận (state, scan_points) từ queue:
          1) Thêm constraint odometry giữa keyframe trước và hiện tại
          2) Thêm loop-closure nếu phát hiện scan hiện tại gần một keyframe cũ
          3) Tối ưu toàn cục pose-graph bằng least-squares
          4) Cập nhật lại pose_x, pose_y, pose_theta
        """
        while self.running:
            try:
                state, scan_points = self.icp_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            init_x, init_y, init_theta = state

            # --- Keyframe đầu tiên ---
            if len(self.keyframes) == 1:
                # Thêm node 1 (keyframe thứ hai)
                self._add_keyframe(init_x, init_y, init_theta)
                self._keyframe_scans.append(scan_points.copy())
                self.last_kf_x, self.last_kf_y, self.last_kf_th = init_x, init_y, init_theta
                self.icp_queue.task_done()
                continue

            # --- 1) Odometry constraint ---
            # Tính relative pose từ keyframe trước đến hiện tại (đơn vị mét/radian)
            dx = (init_x - self.last_kf_x) / 1000.0
            dy = (init_y - self.last_kf_y) / 1000.0
            dtheta = atan2(sin(init_theta - self.last_kf_th),
                           cos(init_theta - self.last_kf_th))

            idx_prev = len(self.keyframes) - 1
            idx_new = idx_prev + 1
            info_odom = np.eye(3) * 50  # ma trận information (cân nhắc tinh chỉnh)
            # Thêm edge between(prev, new)
            self.edges.append((idx_prev, idx_new,
                               np.array([dx, dy, dtheta]),
                               info_odom))

            # Thêm keyframe mới và lưu scan tương ứng
            self.keyframes.append(np.array([init_x / 1000.0, init_y / 1000.0, init_theta]))
            self._keyframe_scans.append(scan_points.copy())

            # --- 2) Loop-closure ---
            # Với mỗi keyframe cũ, nếu khoảng cách trong bán kính, thêm constraint
            for k, (xk, yk, thk) in enumerate(self.keyframes[:-1]):
                dist = sqrt((init_x / 1000.0 - xk) ** 2 + (init_y / 1000.0 - yk) ** 2)
                if dist < self.loop_closure_radius:
                    # Dùng ICP để ước lượng transform từ scan cũ sang scan hiện tại
                    R_clos, t_clos = self.icp(self._keyframe_scans[k], scan_points)
                    dtheta_c = atan2(R_clos[1, 0], R_clos[0, 0])
                    meas = np.array([t_clos[0, 0] / 1000.0,
                                     t_clos[1, 0] / 1000.0,
                                     dtheta_c])
                    info_clos = np.eye(3) * 10
                    self.edges.append((k, idx_new, meas, info_clos))

            # Cập nhật last keyframe để lần sau đo odometry
            self.last_kf_x, self.last_kf_y, self.last_kf_th = init_x, init_y, init_theta

            # --- 3) Tối ưu toàn cục ---
            self._optimize_pose_graph()

            # Đánh dấu xử lý xong
            self.icp_queue.task_done()

    def send_command(self, command):
        self.command_queue.put(command)

    def _handle_commands(self):
        while self.running:
            try:
                command = self.command_queue.get(timeout=0.1)
                if self.sock:
                    self.sock.sendall((command + '\n').encode('utf-8'))
                    logging.info("Sent command: %s", command)
                self.command_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logging.error("Failed to send command: %s", e)
                time.sleep(0.01)

    def get_coordinates(self):
        with self.data_lock:
            return self.data['x_coords'][:], self.data['y_coords'][:]

    def cleanup(self):
        self.running = False
        self.data_event.set()
        if self.sock:
            self.sock.close()
            logging.info("WiFi connection closed.")
        plt.close(self.fig)
        logging.info("Plot closed.")

    def reset_map(self):
        self.global_map = np.full((self.global_map_dim, self.global_map_dim), 127, dtype=np.uint8)
        with self.data_lock:
            self.data['x_coords'].clear()
            self.data['y_coords'].clear()
        print("Đã xóa bản đồ.")

    def save_map(self, filename_npz):
        """
        Lưu occupancy grid và metadata (grid size, origin) vào file .npz
        """
        import numpy as _np
        _np.savez_compressed(filename_npz,
                             global_map=self.global_map,
                             grid_size=self.GRID_SIZE,
                             map_size_mm=self.map_size_mm)
        print(f"[LidarData] Saved map to {filename_npz}")

    def localize_robot(self, num_particles=1000, max_iterations=30):
        import numpy as np
        from math import atan2, sin, cos, sqrt

        # --- Systematic resampling helper ---
        def systematic_resample(weights):
            N = len(weights)
            positions = (np.arange(N) + np.random.rand()) / N
            cum_sum = np.cumsum(weights)
            return np.searchsorted(cum_sum, positions)

        # --- 0) Motion update from odometry ---
        if not hasattr(self, 'last_reloc_pose'):
            self.last_reloc_pose = (self.pose_x, self.pose_y, self.pose_theta)
        dx = self.pose_x - self.last_reloc_pose[0]
        dy = self.pose_y - self.last_reloc_pose[1]
        delta_s = sqrt(dx ** 2 + dy ** 2)
        delta_theta = atan2(sin(self.pose_theta - self.last_reloc_pose[2]),
                            cos(self.pose_theta - self.last_reloc_pose[2]))
        self.last_reloc_pose = (self.pose_x, self.pose_y, self.pose_theta)

        sigma_trans = 5.0  # mm
        sigma_rot = 0.02  # rad

        x0, y0, theta0 = self.pose_x, self.pose_y, self.pose_theta
        R_init = 1000
        particles = np.empty((num_particles, 3))
        particles[:, 0] = np.random.normal(x0, R_init, num_particles)
        particles[:, 1] = np.random.normal(y0, R_init, num_particles)
        particles[:, 2] = np.random.vonmises(theta0, 4.0, num_particles)
        d2 = (particles[:, 0] - x0) ** 2 + (particles[:, 1] - y0) ** 2
        weights = np.exp(-0.5 * (d2 / (R_init ** 2)))
        weights += 1e-300
        weights /= weights.sum()

        angles, distances = self._filter_data(self.data['angles'], self.data['distances'])
        if len(angles) < 10:
            print("Không đủ dữ liệu LIDAR để localize!")
            return
        idx = np.linspace(0, len(angles) - 1, num=30, dtype=int)
        beams = np.vstack((
            np.array(distances)[idx] * np.cos(np.array(angles)[idx] + self.heading_offset),
            np.array(distances)[idx] * np.sin(np.array(angles)[idx] + self.heading_offset)
        )).T

        for _ in range(max_iterations):
            particles[:, 0] += delta_s * np.cos(particles[:, 2])
            particles[:, 1] += delta_s * np.sin(particles[:, 2])
            particles[:, 2] += delta_theta
            particles[:, 2] = np.arctan2(np.sin(particles[:, 2]), np.cos(particles[:, 2]))

            noise_t = np.random.normal(0, sigma_trans, num_particles)
            noise_r = np.random.normal(0, sigma_rot, num_particles)
            particles[:, 0] += noise_t * np.cos(particles[:, 2])
            particles[:, 1] += noise_t * np.sin(particles[:, 2])
            particles[:, 2] += noise_r
            particles[:, 2] = np.arctan2(np.sin(particles[:, 2]), np.cos(particles[:, 2]))

            thetas = particles[:, 2][:, None]
            cos_t = np.cos(thetas)
            sin_t = np.sin(thetas)
            px = particles[:, 0][:, None]
            py = particles[:, 1][:, None]
            X = beams[None, :, 0] * cos_t - beams[None, :, 1] * sin_t + px
            Y = beams[None, :, 0] * sin_t + beams[None, :, 1] * cos_t + py

            gx = ((X + self.map_size_mm / 2) / self.GRID_SIZE).astype(int)
            gy = ((Y + self.map_size_mm / 2) / self.GRID_SIZE).astype(int)
            valid = (gx >= 0) & (gx < self.global_map_dim) & (gy >= 0) & (gy < self.global_map_dim)

            ds = np.full(gx.shape, self.map_size_mm, dtype=float)
            ds[valid] = self.dist_field[gy[valid], gx[valid]]
            weights = np.exp(-0.5 * (ds / self.sigma_z) ** 2).mean(axis=1)

            weights += 1e-300
            weights /= weights.sum()
            H = -np.sum(weights * np.log(weights))
            if H < 0.5:
                break

            idxs = systematic_resample(weights)
            particles = particles[idxs]
            weights.fill(1.0 / num_particles)

        self.pose_x, self.pose_y, self.pose_theta = np.average(particles, axis=0, weights=weights)
        self.pose_theta = np.arctan2(np.sin(self.pose_theta), np.cos(self.pose_theta))

        # --- ICP refinement ---
        angles_full, distances_full = self._filter_data(self.data['angles'], self.data['distances'])
        gx, gy = self._to_global_coordinates(angles_full, distances_full)
        scan = np.vstack((gx, gy)).T

        # Tạo tập điểm từ bản đồ (biên obstacle)
        ys, xs = np.where(self.global_map == 255)
        map_x = xs * self.GRID_SIZE - self.map_size_mm / 2 + self.GRID_SIZE / 2
        map_y = ys * self.GRID_SIZE - self.map_size_mm / 2 + self.GRID_SIZE / 2
        map_points = np.vstack((map_x, map_y)).T

        R_icp, t_icp = self.icp(map_points, scan)
        dtheta = atan2(R_icp[1, 0], R_icp[0, 0])
        self.pose_x += t_icp[0, 0]
        self.pose_y += t_icp[1, 0]
        self.pose_theta = atan2(sin(self.pose_theta + dtheta), cos(self.pose_theta + dtheta))

        self.ekf_slam.state = np.array([self.pose_x, self.pose_y, self.pose_theta])
        print(f"[Localize] x={self.pose_x:.1f}, y={self.pose_y:.1f}, θ={self.pose_theta:.3f} (refined with ICP)")

    def load_map(self, filename_npz):
        import numpy as _np
        import time
        import threading  # thêm nếu chưa có

        data = _np.load(filename_npz)
        self.global_map = data["global_map"]
        self.GRID_SIZE = int(data["grid_size"])
        self.map_size_mm = float(data["map_size_mm"])
        self.global_map_dim = self.global_map.shape[0]
        self.mapping_enabled = False
        self.pose_x = 0.0
        self.pose_y = 0.0
        self.pose_theta = 0.0
        print(f"[LidarData] Loaded map from {filename_npz}")

        # ——— TÍCH HỢP LIKELIHOOD FIELD ———
        from scipy.ndimage import distance_transform_edt
        occ = (self.global_map == 255).astype(_np.uint8)
        self.dist_field = distance_transform_edt(1 - occ) * self.GRID_SIZE
        self.sigma_z = 100.0
        # ——————————————————————————————

        # Xóa dữ liệu lidar cũ
        with self.data_lock:
            self.data['angles'].clear()
            self.data['distances'].clear()

        print("Đang chờ dữ liệu LiDAR để localize...")

        # Chờ đến khi có đủ tia hoặc timeout
        timeout = 5.0  # giây
        start_t = time.time()
        while True:
            angles, distances = self._filter_data(self.data['angles'], self.data['distances'])
            if len(angles) >= 30 or (time.time() - start_t) > timeout:
                break
            time.sleep(0.1)

        # Gọi localize_robot trong thread riêng
        threading.Thread(target=self.localize_robot, daemon=True).start()

    def move(self, distance_m):
        """
        Di chuyển với giá trị lệnh gửi cho ESP nhận theo đơn vị mét.
        Tuy nhiên, pose được cập nhật theo mm, nên khi so sánh convert về mét.
        """
        initial_x, initial_y = self.pose_x, self.pose_y
        # Gửi lệnh với đơn vị là mét (giá trị distance_m)
        cmd = f"MOVE {distance_m}"
        self.send_command(cmd)

        tolerance = 0.05  # Sai số cho phép: 5 cm
        start_time = time.time()
        timeout = 10  # timeout là 10 giây

        while True:
            # pose_x, pose_y tính theo mm → chuyển về mét để so sánh
            current_distance_m = sqrt((self.pose_x - initial_x) ** 2 + (self.pose_y - initial_y) ** 2) / 1000.0
            if current_distance_m >= distance_m - tolerance:
                break
            if time.time() - start_time > timeout:
                print("Timeout khi di chuyển!")
                break
            time.sleep(0.1)

    def rotate(self, angle_rad):
        """
        Quay theo góc angle_rad (đơn vị radian) và gửi lệnh trực tiếp.
        """
        target_theta = self.pose_theta + angle_rad
        target_theta = atan2(sin(target_theta), cos(target_theta))
        tolerance = 0.05  # sai số cho phép: 0.1 rad
        cmd = f"ROTATE {angle_rad}"
        self.send_command(cmd)

        start_time = time.time()
        timeout = 10

        while True:
            error = abs(atan2(sin(self.pose_theta - target_theta), cos(self.pose_theta - target_theta)))
            if error < tolerance:
                break
            if time.time() - start_time > timeout:
                print("Timeout khi quay!")
                break
            time.sleep(0.1)

    def navigate_to_target(self, target_x, target_y):
        """
        Định hướng robot từ vị trí hiện tại đến tọa độ target (mm) trên bản đồ.
        Tái lập kế hoạch định kỳ để phản ứng kịp thời với môi trường thay đổi dựa trên bản đồ được cập nhật.
        """
        # Sử dụng tolerance là bán kính tối thiểu xác định đã đến gần đích
        tolerance = 4*self.GRID_SIZE  # ví dụ: nửa ô của grid (15 mm khi GRID_SIZE=30)

        while True:
            # Tính vị trí hiện tại trên grid và đích theo đơn vị grid
            grid_start = (
                int((self.pose_x + self.map_size_mm / 2) / self.GRID_SIZE),
                int((self.pose_y + self.map_size_mm / 2) / self.GRID_SIZE)
            )
            grid_target = (
                int((target_x + self.map_size_mm / 2) / self.GRID_SIZE),
                int((target_y + self.map_size_mm / 2) / self.GRID_SIZE)
            )

            # Tìm đường đi bằng A*
            path = a_star_search(self.global_map, grid_start, grid_target)
            if not path:
                print("Không tìm được đường đi đến đích!")
                return

            # Làm mượt đường đi
            smooth_waypoints = smooth_path(path, self.global_map)
            # Chuyển đổi tọa độ grid sang tọa độ thực (mm)
            waypoints_x = [
                cell[0] * self.GRID_SIZE - self.map_size_mm / 2 + self.GRID_SIZE / 2
                for cell in smooth_waypoints
            ]
            waypoints_y = [
                cell[1] * self.GRID_SIZE - self.map_size_mm / 2 + self.GRID_SIZE / 2
                for cell in smooth_waypoints
            ]

            # Vẽ đường đi lên bản đồ (nếu muốn hiển thị)
            self.ax.plot(waypoints_x, waypoints_y, 'bo-', markersize=5, label='Smoothed Waypoints')
            self.ax.legend()
            self.fig.canvas.draw()

            # Chọn waypoint đầu tiên nằm xa vị trí hiện tại một cách đáng kể (để tránh lặp lại quá nhỏ)
            next_waypoint = None
            for wx, wy in zip(waypoints_x, waypoints_y):
                distance = sqrt((wx - self.pose_x) ** 2 + (wy - self.pose_y) ** 2)
                if distance > tolerance:
                    next_waypoint = (wx, wy)
                    break
            # Nếu không tìm thấy waypoint nào thỏa, coi như đã đến đích
            if next_waypoint is None:
                print("Đã đến gần đích")
                return

            wx, wy = next_waypoint
            # Tính góc cần quay
            desired_angle = atan2(wy - self.pose_y, wx - self.pose_x)
            angle_diff = desired_angle - self.pose_theta
            angle_diff = atan2(sin(angle_diff), cos(angle_diff))
            print(f"Di chuyển đến waypoint tại ({wx:.1f}, {wy:.1f}); cần xoay {angle_diff:.3f} rad")

            # Thực hiện quay và di chuyển
            self.rotate(angle_diff)
            # Chuyển khoảng cách từ mm sang mét (do hàm move gửi lệnh với đơn vị mét)
            distance_m = sqrt((wx - self.pose_x) ** 2 + (wy - self.pose_y) ** 2) / 1000.0
            self.move(distance_m)

            # Sau mỗi bước di chuyển, kiểm tra khoảng cách còn lại đến đích
            overall_distance = sqrt((target_x - self.pose_x) ** 2 + (target_y - self.pose_y) ** 2)
            print(f"Khoảng cách còn lại đến đích: {overall_distance:.1f} mm")
            if overall_distance < tolerance:
                print("Đã hoàn thành di chuyển đến vị trí đích")
                return

    def icp(self, A, B, max_iterations=30, tolerance=1e-4):
        """
        Thực hiện ICP giữa hai tập điểm A và B.
        A, B: numpy arrays kích thước (N,2)
        Trả về (R, t): ma trận quay (2x2) và vector dịch chuyển (2x1),
            đại diện cho phép biến đổi từ B sang A.
        """
        if A.shape[0] == 0 or B.shape[0] == 0:
            # Nếu một trong hai tập rỗng, trả về không có thay đổi
            return np.eye(2), np.zeros((2, 1))

        prev_error = float('inf')
        R = np.eye(2)
        t = np.zeros((2, 1))
        B_transformed = B.copy()

        for i in range(max_iterations):
            # Tính khoảng cách giữa mỗi điểm của A và tất cả các điểm của B_transformed
            distances = np.linalg.norm(A[:, None] - B_transformed[None, :], axis=2)
            # Nếu B_transformed không có điểm nào, trả về không thay đổi
            if distances.size == 0:
                return np.eye(2), np.zeros((2, 1))
            indices = np.argmin(distances, axis=0)
            closest_points = A[indices]

            centroid_A = np.mean(closest_points, axis=0)
            centroid_B = np.mean(B_transformed, axis=0)

            AA = closest_points - centroid_A
            BB = B_transformed - centroid_B

            H = BB.T @ AA
            U, S, Vt = np.linalg.svd(H)
            R_iter = Vt.T @ U.T
            if np.linalg.det(R_iter) < 0:
                Vt[1, :] *= -1
                R_iter = Vt.T @ U.T
            t_iter = centroid_A.reshape(2, 1) - R_iter @ centroid_B.reshape(2, 1)

            B_transformed = (R_iter @ B_transformed.T).T + t_iter.T

            mean_error = np.mean(np.linalg.norm(closest_points - B_transformed, axis=1))
            if abs(prev_error - mean_error) < tolerance:
                break
            prev_error = mean_error
            R = R_iter @ R
            t = R_iter @ t + t_iter

        return R, t


# ------------------------------
# Chương trình chính
# ------------------------------
if __name__ == "__main__":
    lidar = LidarData(host='192.168.1.104', port=80, neighbor_radius=30, min_neighbors=7)
    data_thread = threading.Thread(target=lidar.update_data, daemon=True)
    data_thread.start()
    app = QApplication(sys.argv)
    window = LidarWindow(lidar)
    window.show()
    try:
        sys.exit(app.exec_())
    finally:
        lidar.cleanup()