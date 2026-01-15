"""
软体机械臂 PCC 模型渲染器
基于 Piecewise Constant Curvature 模型
两块正三角形板子通过三根气动 origami actuator 连接
Actuator 是等曲率弧形

坐标系（倒挂配置）：
- 固定端（Base）在上方 z=0
- 自由端（End）向下延伸（负 Z 方向）
- X 轴指向 actuator 0 方向
- Y 轴满足右手定则

PCC 模型约束：
- 单段 PCC 只有 2 个自由度：弯曲方向 φ 和弯曲角 θ_bend
- 所有 actuator 是同心圆弧，垂直于两端平台
- 不能产生扭转（Yaw）

正向运动学输入：
- 方式 1: 弯曲方向 φ + 弯曲角 θ_bend + 弧长 s
- 方式 2: 末端法向量方向（自动投影到有效 PCC 配置）
"""

import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
import math

# 常量定义
TRIANGLE_SIDE = 40.0      # 正三角形边长 (cm)
TRIANGLE_THICKNESS = 7.0  # 三角形板子厚度 (cm)
ACTUATOR_MIN = 35.0       # actuator 最短长度 (cm)
ACTUATOR_MAX = 65.0       # actuator 最长长度 (cm)
ARC_SEGMENTS = 32         # 弧形分段数
DEFAULT_ARC_LENGTH = 50.0 # 默认中轴弧长 (cm)
MAX_BEND_ANGLE = math.radians(60)  # 最大弯曲角度

# 窗口设置
WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 800

# 颜色定义 (RGB, 0-1 范围)
COLOR_TRIANGLE_FIXED = (0.2, 0.6, 0.9, 0.85)    # 固定端（上方）- 蓝色
COLOR_TRIANGLE_FREE = (0.9, 0.4, 0.2, 0.85)     # 自由端（下方）- 橙色
COLOR_TRIANGLE_SIDE = (0.4, 0.4, 0.5, 0.7)      # 三棱柱侧面
COLOR_ACTUATOR_0 = (0.95, 0.25, 0.3, 1.0)       # actuator 0 - 红色 (X轴方向)
COLOR_ACTUATOR_1 = (0.25, 0.95, 0.35, 1.0)      # actuator 1 - 绿色 (120°方向)
COLOR_ACTUATOR_2 = (0.35, 0.25, 0.95, 1.0)      # actuator 2 - 蓝色 (240°方向)
COLOR_GRID = (0.3, 0.3, 0.3, 0.5)               # 网格颜色
COLOR_CENTER_LINE = (1.0, 1.0, 0.3, 1.0)        # 中心线 - 黄色
COLOR_MOUNT = (0.5, 0.5, 0.6, 0.9)              # 固定支架颜色


def rotation_matrix_axis_angle(axis, angle):
    """
    根据轴-角表示创建旋转矩阵（罗德里格斯公式）
    """
    axis = np.array(axis, dtype=float)
    norm = np.linalg.norm(axis)
    if norm < 1e-10:
        return np.eye(3)
    axis = axis / norm
    
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])
    
    R = np.eye(3) + math.sin(angle) * K + (1 - math.cos(angle)) * (K @ K)
    return R


def get_equilateral_triangle_vertices(side_length, center=np.array([0, 0, 0])):
    """
    获取正三角形顶点坐标
    顶点 0 在 X 轴正方向
    """
    R = side_length / math.sqrt(3)
    vertices = []
    for i in range(3):
        angle = i * 2 * math.pi / 3
        x = center[0] + R * math.cos(angle)
        y = center[1] + R * math.sin(angle)
        z = center[2]
        vertices.append(np.array([x, y, z]))
    return vertices


# ============================================================
# PCC 正向运动学（正确的约束模型）
# ============================================================

class PCCForwardKinematics:
    """
    PCC (Piecewise Constant Curvature) 正向运动学
    
    单段 PCC 的自由度：
    - 弯曲方向 φ (phi): 在 XY 平面内的角度，决定弯曲朝向
    - 弯曲角度 θ_bend (theta_bend): 弯曲的程度
    - 弧长 s: 中心线的长度
    
    约束：
    - 曲率 κ = θ_bend / s
    - 所有 actuator 垂直于两端平台
    - 不能产生扭转
    
    倒挂配置：
    - 固定端在 z=0，自由端向下（-Z 方向）
    """
    
    def __init__(self, triangle_side=TRIANGLE_SIDE):
        self.triangle_side = triangle_side
        self.r = triangle_side / math.sqrt(3)  # 外接圆半径
        self.actuator_angles = [0, 2*math.pi/3, 4*math.pi/3]
    
    def forward_kinematics(self, phi, theta_bend, arc_length):
        """
        正向运动学：从 PCC 参数计算机械臂状态
        
        参数：
            phi: 弯曲方向角（弧度），0 表示向 X+ 弯曲，π/2 表示向 Y+ 弯曲
            theta_bend: 弯曲角度（弧度），正值表示弯曲
            arc_length: 中轴弧长 s (cm)
        
        返回：
            dict: PCC 参数和几何信息
        """
        s = arc_length
        theta_bend = max(0, min(MAX_BEND_ANGLE, abs(theta_bend)))  # 限制弯曲角度
        
        if theta_bend < 1e-6:
            # 无弯曲（直立状态）
            kappa = 0
            
            end_center = np.array([0, 0, -s])
            end_rotation = np.eye(3)
            end_normal = np.array([0, 0, -1])
            
            # 所有 actuator 长度相等
            actuator_lengths = [s, s, s]
        else:
            # 有弯曲
            kappa = theta_bend / s
            R_bend = 1.0 / kappa
            
            # 末端位置
            # 在弯曲平面内：
            # x_local = R * (1 - cos(θ))（向弯曲方向偏移）
            # z_local = -R * sin(θ)（向下）
            x_local = R_bend * (1 - math.cos(theta_bend))
            z_local = -R_bend * math.sin(theta_bend)
            
            end_center = np.array([
                x_local * math.cos(phi),
                x_local * math.sin(phi),
                z_local
            ])
            
            # 末端姿态：绕弯曲轴旋转 theta_bend
            # 弯曲轴 = (-sin(φ), cos(φ), 0)，垂直于弯曲方向
            bend_axis = np.array([-math.sin(phi), math.cos(phi), 0])
            end_rotation = rotation_matrix_axis_angle(bend_axis, theta_bend)
            end_normal = end_rotation @ np.array([0, 0, -1])
            
            # 计算 actuator 长度
            # l_i = s * (1 + r * κ * cos(θ_i - φ))
            # 注意：这里的符号确保弯曲方向正确
            actuator_lengths = []
            for i in range(3):
                theta_i = self.actuator_angles[i]
                l_i = s * (1 + self.r * kappa * math.cos(theta_i - phi))
                actuator_lengths.append(l_i)
        
        return {
            'phi': phi,
            'theta_bend': theta_bend,
            's': s,
            'kappa': kappa,
            'end_center': end_center,
            'end_rotation': end_rotation,
            'end_normal': end_normal,
            'actuator_lengths': actuator_lengths,
            'bend_axis': np.array([-math.sin(phi), math.cos(phi), 0]) if theta_bend > 1e-6 else np.array([0, 1, 0])
        }
    
    def forward_kinematics_from_tilt(self, tilt_x, tilt_y, arc_length):
        """
        从倾斜角度计算 PCC 参数
        
        这是一种更直观的输入方式：
        - tilt_x: 末端相对于垂直方向在 XZ 平面内的倾斜角度
        - tilt_y: 末端相对于垂直方向在 YZ 平面内的倾斜角度
        
        这两个角度会被转换为 PCC 的 (phi, theta_bend)
        """
        # 计算末端法向量的方向
        # 初始法向量是 (0, 0, -1)，倾斜后变为...
        
        # 将 tilt 角度转换为 PCC 参数
        # theta_bend = sqrt(tilt_x^2 + tilt_y^2)
        # phi = atan2(tilt_y, tilt_x)
        
        theta_bend = math.sqrt(tilt_x**2 + tilt_y**2)
        theta_bend = min(theta_bend, MAX_BEND_ANGLE)
        
        if theta_bend < 1e-6:
            phi = 0
        else:
            phi = math.atan2(tilt_y, tilt_x)
        
        return self.forward_kinematics(phi, theta_bend, arc_length)
    
    def get_center_arc_points(self, fk_result, num_segments=ARC_SEGMENTS):
        """获取中心线弧形上的点"""
        s = fk_result['s']
        kappa = fk_result['kappa']
        phi = fk_result['phi']
        
        points = []
        
        if kappa < 1e-6:
            for i in range(num_segments + 1):
                t = i / num_segments
                z = -t * s
                points.append(np.array([0, 0, z]))
        else:
            R_bend = 1.0 / kappa
            theta_total = fk_result['theta_bend']
            
            for i in range(num_segments + 1):
                t = i / num_segments
                theta = t * theta_total
                
                x_local = R_bend * (1 - math.cos(theta))
                z_local = -R_bend * math.sin(theta)
                
                x = x_local * math.cos(phi)
                y = x_local * math.sin(phi)
                z = z_local
                
                points.append(np.array([x, y, z]))
        
        return points
    
    def get_actuator_arc_points(self, fk_result, actuator_index, num_segments=ARC_SEGMENTS):
        """
        获取特定 actuator 弧形上的点
        
        严格等曲率圆弧模型：
        1. 所有 actuator 曲率相同 κ
        2. 起点切线垂直于固定端板
        3. 终点切线垂直于自由端板（自动满足，因为同心圆弧）
        4. 三角形会变形（物理必然）
        
        每个 actuator 的弯曲平面包含：起点、起点切线、和弯曲方向
        """
        kappa = fk_result['kappa']
        phi = fk_result['phi']
        theta_bend = fk_result['theta_bend']
        actuator_angle = self.actuator_angles[actuator_index]
        actuator_length = fk_result['actuator_lengths'][actuator_index]
        
        # 起点（固定端三角形顶点）
        P0 = np.array([
            self.r * math.cos(actuator_angle),
            self.r * math.sin(actuator_angle),
            0
        ])
        
        points = [P0.copy()]
        
        if kappa < 1e-6:
            # 直线情况
            for i in range(1, num_segments + 1):
                t = i / num_segments
                z = -t * actuator_length
                points.append(np.array([P0[0], P0[1], z]))
            return points
        
        # 该 actuator 的弯曲半径
        # R_i = l_i / theta_bend（因为弧长 = 半径 × 角度）
        R_act = actuator_length / theta_bend
        
        # 弯曲轴（所有 actuator 共享同一弯曲轴，垂直于弯曲方向）
        bend_axis = np.array([-math.sin(phi), math.cos(phi), 0])
        
        # 起点切线方向（垂直于固定端板，向下）
        T0 = np.array([0, 0, -1])
        
        # 圆心位置：从起点沿垂直于切线的方向偏移 R_act
        # 垂直于 T0 且在弯曲平面内的方向
        # 弯曲平面由 T0 和弯曲方向定义
        bend_dir = np.array([math.cos(phi), math.sin(phi), 0])
        
        # 圆心方向（垂直于 T0，在弯曲方向上）
        # 对于向 phi 方向弯曲，圆心在 -bend_dir 方向
        center_dir = -bend_dir
        
        # 圆心位置
        center = P0 + R_act * center_dir
        
        # 生成圆弧点
        for i in range(1, num_segments + 1):
            t = i / num_segments
            angle = t * theta_bend
            
            # 从起点绕弯曲轴旋转
            # 起点相对于圆心的向量
            v0 = P0 - center
            
            # 绕弯曲轴旋转 angle
            rot = rotation_matrix_axis_angle(bend_axis, angle)
            v = rot @ v0
            
            point = center + v
            points.append(point)
        
        return points
    
    def verify_perpendicularity(self, fk_result):
        """
        验证 actuator 是否垂直于两端平台
        
        在 PCC 模型中，actuator 切线应该与中心线切线平行（都垂直于截面盘）
        
        检查方法：比较 actuator 切线与中心线切线的夹角
        """
        angles = []
        
        # 获取中心线点以计算切线
        center_arc = self.get_center_arc_points(fk_result)
        
        for i in range(3):
            arc_points = self.get_actuator_arc_points(fk_result, i)
            
            angle_start = 0
            angle_end = 0
            
            # 在起点（固定端）检查
            if len(arc_points) >= 2 and len(center_arc) >= 2:
                tangent_act = np.array(arc_points[1]) - np.array(arc_points[0])
                tangent_center = np.array(center_arc[1]) - np.array(center_arc[0])
                
                norm_act = np.linalg.norm(tangent_act)
                norm_center = np.linalg.norm(tangent_center)
                
                if norm_act > 1e-10 and norm_center > 1e-10:
                    tangent_act = tangent_act / norm_act
                    tangent_center = tangent_center / norm_center
                    dot_start = abs(np.dot(tangent_act, tangent_center))
                    angle_start = math.degrees(math.acos(np.clip(dot_start, 0, 1)))
            
            # 在终点（自由端）检查
            if len(arc_points) >= 2 and len(center_arc) >= 2:
                tangent_act = np.array(arc_points[-1]) - np.array(arc_points[-2])
                tangent_center = np.array(center_arc[-1]) - np.array(center_arc[-2])
                
                norm_act = np.linalg.norm(tangent_act)
                norm_center = np.linalg.norm(tangent_center)
                
                if norm_act > 1e-10 and norm_center > 1e-10:
                    tangent_act = tangent_act / norm_act
                    tangent_center = tangent_center / norm_center
                    dot_end = abs(np.dot(tangent_act, tangent_center))
                    angle_end = math.degrees(math.acos(np.clip(dot_end, 0, 1)))
            
            angles.append({
                'actuator': i,
                'angle_at_fixed_end': angle_start,
                'angle_at_free_end': angle_end
            })
        
        return angles


# ============================================================
# 软体机械臂类
# ============================================================

class SoftRobotArm:
    """
    软体机械臂类（倒挂配置）
    
    使用正确的 PCC 参数：
    - phi: 弯曲方向
    - theta_bend: 弯曲角度
    - arc_length: 弧长
    """
    
    def __init__(self):
        self.fk = PCCForwardKinematics(TRIANGLE_SIDE)
        
        # PCC 参数
        self.phi = 0.0              # 弯曲方向
        self.theta_bend = 0.0       # 弯曲角度
        self.arc_length = DEFAULT_ARC_LENGTH
        
        # 更直观的控制：倾斜角度
        self.tilt_x = 0.0  # X 方向倾斜
        self.tilt_y = 0.0  # Y 方向倾斜
        
        # 固定端位置
        self.fixed_z = 0.0
        
        self._update_geometry()
    
    def set_pcc_params(self, phi, theta_bend, arc_length):
        """直接设置 PCC 参数"""
        self.phi = phi
        self.theta_bend = max(0, min(MAX_BEND_ANGLE, theta_bend))
        self.arc_length = max(ACTUATOR_MIN, min(ACTUATOR_MAX, arc_length))
        
        # 更新 tilt 值以保持同步
        self.tilt_x = self.theta_bend * math.cos(self.phi)
        self.tilt_y = self.theta_bend * math.sin(self.phi)
        
        self._update_geometry()
    
    def set_tilt(self, tilt_x, tilt_y):
        """使用倾斜角度设置姿态"""
        self.tilt_x = tilt_x
        self.tilt_y = tilt_y
        
        # 转换为 PCC 参数
        self.theta_bend = math.sqrt(tilt_x**2 + tilt_y**2)
        self.theta_bend = min(self.theta_bend, MAX_BEND_ANGLE)
        
        if self.theta_bend > 1e-6:
            self.phi = math.atan2(tilt_y, tilt_x)
        
        self._update_geometry()
    
    def set_arc_length(self, s):
        """设置弧长"""
        self.arc_length = max(ACTUATOR_MIN, min(ACTUATOR_MAX, s))
        self._update_geometry()
    
    def _update_geometry(self):
        """更新几何状态"""
        self.pcc_result = self.fk.forward_kinematics(
            self.phi, self.theta_bend, self.arc_length
        )
        
        # 固定端三棱柱
        # 顶面在 z=0，底面在 z=-TRIANGLE_THICKNESS
        self.fixed_vertices_top = get_equilateral_triangle_vertices(
            TRIANGLE_SIDE, center=np.array([0, 0, self.fixed_z])
        )
        self.fixed_vertices_bottom = get_equilateral_triangle_vertices(
            TRIANGLE_SIDE, center=np.array([0, 0, self.fixed_z - TRIANGLE_THICKNESS])
        )
        
        # 计算中心线和 Actuator 弧线
        center_arc_raw = self.fk.get_center_arc_points(self.pcc_result)
        self.center_arc = [pt - np.array([0, 0, TRIANGLE_THICKNESS]) for pt in center_arc_raw]
        
        self.actuator_arcs = []
        for i in range(3):
            arc = self.fk.get_actuator_arc_points(self.pcc_result, i)
            adjusted_arc = [pt - np.array([0, 0, TRIANGLE_THICKNESS]) for pt in arc]
            self.actuator_arcs.append(adjusted_arc)
        
        # 自由端三角形顶点 = 圆弧终点（三角形会变形，这是物理必然）
        self.free_vertices_top = []
        for i in range(3):
            top_vertex = self.actuator_arcs[i][-1].copy()
            self.free_vertices_top.append(top_vertex)
        
        # 板的法向量：所有 actuator 终点切线相同（同心圆弧特性）
        # 使用第一个 actuator 的终点切线
        arc = self.actuator_arcs[0]
        if len(arc) >= 2:
            tangent = np.array(arc[-1]) - np.array(arc[-2])
            tangent = tangent / np.linalg.norm(tangent)
            self.end_tangent = tangent
        else:
            self.end_tangent = self.pcc_result['end_normal']
        
        # 自由端底面顶点 = 顶面顶点沿法向量方向偏移
        self.free_vertices_bottom = []
        for i in range(3):
            top_vertex = self.free_vertices_top[i]
            bottom_vertex = top_vertex + self.end_tangent * TRIANGLE_THICKNESS
            self.free_vertices_bottom.append(bottom_vertex)


# ============================================================
# 渲染器
# ============================================================

class Renderer:
    """OpenGL 渲染器"""
    
    def __init__(self):
        pygame.init()
        pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), DOUBLEBUF | OPENGL)
        pygame.display.set_caption("软体机械臂 PCC 模型 - 倒挂配置 (正确约束)")
        
        self._setup_opengl()
        
        self.camera_distance = 200.0
        self.camera_angle_x = -20.0
        self.camera_angle_y = 45.0
        
        self.mouse_dragging = False
        self.last_mouse_pos = (0, 0)
        
        self.robot = SoftRobotArm()
        
        self.show_center_line = True
        self.show_mount = True
        self.show_perpendicularity_check = False
    
    def _setup_opengl(self):
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_LINE_SMOOTH)
        glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)
        
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        
        glLightfv(GL_LIGHT0, GL_POSITION, (100, 100, 200, 1))
        glLightfv(GL_LIGHT0, GL_AMBIENT, (0.3, 0.3, 0.3, 1))
        glLightfv(GL_LIGHT0, GL_DIFFUSE, (0.8, 0.8, 0.8, 1))
        
        glClearColor(0.06, 0.06, 0.1, 1.0)
        
        glMatrixMode(GL_PROJECTION)
        gluPerspective(45, WINDOW_WIDTH / WINDOW_HEIGHT, 0.1, 1000.0)
        glMatrixMode(GL_MODELVIEW)
    
    def _update_camera(self):
        glLoadIdentity()
        
        rad_x = math.radians(self.camera_angle_x)
        rad_y = math.radians(self.camera_angle_y)
        
        cam_x = self.camera_distance * math.cos(rad_x) * math.sin(rad_y)
        cam_y = self.camera_distance * math.sin(rad_x)
        cam_z = self.camera_distance * math.cos(rad_x) * math.cos(rad_y)
        
        look_at_height = -25
        gluLookAt(cam_x, cam_z, cam_y,
                  0, look_at_height, 0,
                  0, 1, 0)
    
    def _to_gl_coords(self, v):
        if isinstance(v, np.ndarray):
            return (float(v[0]), float(v[2]), float(v[1]))
        return (v[0], v[2], v[1])
    
    def _draw_mount(self):
        if not self.show_mount:
            return
        
        glColor4f(*COLOR_MOUNT)
        mount_width = TRIANGLE_SIDE * 1.5
        mount_height = 10
        mount_depth = TRIANGLE_SIDE * 1.5
        
        glPushMatrix()
        glTranslatef(0, mount_height/2, 0)
        
        glBegin(GL_QUADS)
        glNormal3f(0, 1, 0)
        glVertex3f(-mount_width/2, mount_height/2, -mount_depth/2)
        glVertex3f(mount_width/2, mount_height/2, -mount_depth/2)
        glVertex3f(mount_width/2, mount_height/2, mount_depth/2)
        glVertex3f(-mount_width/2, mount_height/2, mount_depth/2)
        glNormal3f(0, -1, 0)
        glVertex3f(-mount_width/2, -mount_height/2, mount_depth/2)
        glVertex3f(mount_width/2, -mount_height/2, mount_depth/2)
        glVertex3f(mount_width/2, -mount_height/2, -mount_depth/2)
        glVertex3f(-mount_width/2, -mount_height/2, -mount_depth/2)
        glEnd()
        
        glPopMatrix()
    
    def _draw_grid(self):
        glDisable(GL_LIGHTING)
        glColor4f(*COLOR_GRID)
        glBegin(GL_LINES)
        
        grid_size = 100
        grid_step = 10
        grid_y = -100
        
        for i in range(-grid_size, grid_size + 1, grid_step):
            glVertex3f(i, grid_y, -grid_size)
            glVertex3f(i, grid_y, grid_size)
            glVertex3f(-grid_size, grid_y, i)
            glVertex3f(grid_size, grid_y, i)
        
        glEnd()
        glEnable(GL_LIGHTING)
    
    def _draw_axes(self):
        glDisable(GL_LIGHTING)
        axis_length = 30.0
        
        glLineWidth(2.5)
        glBegin(GL_LINES)
        
        glColor4f(1.0, 0.3, 0.3, 1.0)
        glVertex3f(0, 0, 0)
        glVertex3f(axis_length, 0, 0)
        
        glColor4f(0.3, 1.0, 0.3, 1.0)
        glVertex3f(0, 0, 0)
        glVertex3f(0, 0, axis_length)
        
        glColor4f(0.3, 0.3, 1.0, 1.0)
        glVertex3f(0, 0, 0)
        glVertex3f(0, axis_length, 0)
        
        glEnd()
        glLineWidth(1.0)
        glEnable(GL_LIGHTING)
    
    def _draw_triangular_prism(self, bottom_vertices, top_vertices, color, side_color):
        gl_bottom = [self._to_gl_coords(v) for v in bottom_vertices]
        gl_top = [self._to_gl_coords(v) for v in top_vertices]
        
        glColor4f(*color)
        glBegin(GL_TRIANGLES)
        v0 = np.array(gl_top[0])
        v1 = np.array(gl_top[1])
        v2 = np.array(gl_top[2])
        normal = np.cross(v1 - v0, v2 - v0)
        norm_len = np.linalg.norm(normal)
        if norm_len > 0:
            normal = normal / norm_len
        glNormal3f(*normal)
        for v in gl_top:
            glVertex3f(*v)
        glEnd()
        
        glBegin(GL_TRIANGLES)
        glNormal3f(*(-normal))
        for v in reversed(gl_bottom):
            glVertex3f(*v)
        glEnd()
        
        glColor4f(*side_color)
        for i in range(3):
            j = (i + 1) % 3
            v0 = np.array(gl_bottom[i])
            v1 = np.array(gl_bottom[j])
            v2 = np.array(gl_top[i])
            edge1 = v1 - v0
            edge2 = v2 - v0
            normal = np.cross(edge1, edge2)
            norm_len = np.linalg.norm(normal)
            if norm_len > 0:
                normal = normal / norm_len
            
            glBegin(GL_QUADS)
            glNormal3f(*normal)
            glVertex3f(*gl_bottom[i])
            glVertex3f(*gl_bottom[j])
            glVertex3f(*gl_top[j])
            glVertex3f(*gl_top[i])
            glEnd()
        
        glDisable(GL_LIGHTING)
        glColor4f(1.0, 1.0, 1.0, 0.8)
        glLineWidth(1.5)
        
        glBegin(GL_LINE_LOOP)
        for v in gl_top:
            glVertex3f(*v)
        glEnd()
        
        glBegin(GL_LINE_LOOP)
        for v in gl_bottom:
            glVertex3f(*v)
        glEnd()
        
        glBegin(GL_LINES)
        for i in range(3):
            glVertex3f(*gl_bottom[i])
            glVertex3f(*gl_top[i])
        glEnd()
        
        glLineWidth(1.0)
        glEnable(GL_LIGHTING)
        
        for v in gl_top:
            self._draw_sphere(v, 1.2, (1.0, 1.0, 1.0, 1.0))
        for v in gl_bottom:
            self._draw_sphere(v, 1.2, (0.8, 0.8, 0.8, 1.0))
    
    def _draw_sphere(self, position, radius, color):
        glColor4f(*color)
        glPushMatrix()
        glTranslatef(*position)
        quadric = gluNewQuadric()
        gluSphere(quadric, radius, 16, 16)
        gluDeleteQuadric(quadric)
        glPopMatrix()
    
    def _draw_actuator_tube(self, points, color, radius=1.0):
        gl_points = [self._to_gl_coords(p) for p in points]
        glColor4f(*color)
        
        tube_segments = 8
        
        for i in range(len(gl_points) - 1):
            p1 = np.array(gl_points[i])
            p2 = np.array(gl_points[i + 1])
            
            direction = p2 - p1
            length = np.linalg.norm(direction)
            if length < 0.001:
                continue
            direction = direction / length
            
            if abs(direction[1]) < 0.99:
                up = np.array([0, 1, 0])
            else:
                up = np.array([1, 0, 0])
            
            right = np.cross(direction, up)
            right = right / np.linalg.norm(right)
            up = np.cross(right, direction)
            
            glBegin(GL_QUAD_STRIP)
            for j in range(tube_segments + 1):
                angle = j * 2 * math.pi / tube_segments
                offset = right * math.cos(angle) * radius + up * math.sin(angle) * radius
                normal = offset / radius
                glNormal3f(*normal)
                v1 = p1 + offset
                v2 = p2 + offset
                glVertex3f(*v1)
                glVertex3f(*v2)
            glEnd()
        
        self._draw_sphere(gl_points[0], radius * 1.5, color)
        self._draw_sphere(gl_points[-1], radius * 1.5, color)
    
    def _draw_center_line(self, points, color, line_width=2.0):
        glDisable(GL_LIGHTING)
        gl_points = [self._to_gl_coords(p) for p in points]
        glColor4f(*color)
        glLineWidth(line_width)
        glBegin(GL_LINE_STRIP)
        for p in gl_points:
            glVertex3f(*p)
        glEnd()
        glLineWidth(1.0)
        glEnable(GL_LIGHTING)
    
    def _draw_normal_arrows(self):
        """绘制法向量箭头，验证垂直性"""
        if not self.show_perpendicularity_check:
            return
        
        glDisable(GL_LIGHTING)
        glLineWidth(2.0)
        
        # 固定端法向量（向下）
        glColor4f(1.0, 1.0, 0.0, 1.0)
        glBegin(GL_LINES)
        glVertex3f(0, -TRIANGLE_THICKNESS, 0)
        glVertex3f(0, -TRIANGLE_THICKNESS - 15, 0)
        glEnd()
        
        # 自由端法向量
        end_center = self.robot.pcc_result['end_center']
        end_normal = self.robot.pcc_result['end_normal']
        start = self._to_gl_coords(end_center - np.array([0, 0, TRIANGLE_THICKNESS]))
        end_pt = end_center - np.array([0, 0, TRIANGLE_THICKNESS]) + end_normal * 15
        end_gl = self._to_gl_coords(end_pt)
        
        glColor4f(1.0, 0.5, 0.0, 1.0)
        glBegin(GL_LINES)
        glVertex3f(*start)
        glVertex3f(*end_gl)
        glEnd()
        
        glLineWidth(1.0)
        glEnable(GL_LIGHTING)
    
    def _draw_robot(self):
        self._draw_triangular_prism(
            self.robot.fixed_vertices_bottom,
            self.robot.fixed_vertices_top,
            COLOR_TRIANGLE_FIXED,
            COLOR_TRIANGLE_SIDE
        )
        
        self._draw_triangular_prism(
            self.robot.free_vertices_bottom,
            self.robot.free_vertices_top,
            COLOR_TRIANGLE_FREE,
            COLOR_TRIANGLE_SIDE
        )
        
        colors = [COLOR_ACTUATOR_0, COLOR_ACTUATOR_1, COLOR_ACTUATOR_2]
        for i in range(3):
            self._draw_actuator_tube(self.robot.actuator_arcs[i], colors[i], radius=0.8)
        
        if self.show_center_line:
            self._draw_center_line(self.robot.center_arc, COLOR_CENTER_LINE, line_width=3.0)
        
        self._draw_normal_arrows()
    
    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            
            elif event.type == pygame.KEYDOWN:
                step = math.radians(3)  # 3度步进
                
                if event.key == pygame.K_ESCAPE:
                    return False
                elif event.key == pygame.K_r:
                    self.robot.set_tilt(0, 0)
                    self.robot.set_arc_length(DEFAULT_ARC_LENGTH)
                
                # Tilt X 控制 (Q/A) - 向 X 方向弯曲
                elif event.key == pygame.K_q:
                    self.robot.set_tilt(self.robot.tilt_x + step, self.robot.tilt_y)
                elif event.key == pygame.K_a:
                    self.robot.set_tilt(self.robot.tilt_x - step, self.robot.tilt_y)
                
                # Tilt Y 控制 (W/S) - 向 Y 方向弯曲
                elif event.key == pygame.K_w:
                    self.robot.set_tilt(self.robot.tilt_x, self.robot.tilt_y + step)
                elif event.key == pygame.K_s:
                    self.robot.set_tilt(self.robot.tilt_x, self.robot.tilt_y - step)
                
                # 弧长控制 (UP/DOWN)
                elif event.key == pygame.K_UP:
                    self.robot.set_arc_length(self.robot.arc_length + 2)
                elif event.key == pygame.K_DOWN:
                    self.robot.set_arc_length(self.robot.arc_length - 2)
                
                # 切换显示
                elif event.key == pygame.K_c:
                    self.show_center_line = not self.show_center_line
                elif event.key == pygame.K_m:
                    self.show_mount = not self.show_mount
                elif event.key == pygame.K_p:
                    self.show_perpendicularity_check = not self.show_perpendicularity_check
                    if self.show_perpendicularity_check:
                        # 验证垂直性
                        perp = self.robot.fk.verify_perpendicularity(self.robot.pcc_result)
                        print("\n垂直性验证:")
                        for p in perp:
                            print(f"  Actuator {p['actuator']}: "
                                  f"固定端偏差 {p['angle_at_fixed_end']:.2f}°, "
                                  f"自由端偏差 {p['angle_at_free_end']:.2f}°")
                
                # 预设姿态
                elif event.key == pygame.K_1:
                    self.robot.set_tilt(math.radians(25), 0)
                elif event.key == pygame.K_2:
                    self.robot.set_tilt(math.radians(-25), 0)
                elif event.key == pygame.K_3:
                    self.robot.set_tilt(0, math.radians(25))
                elif event.key == pygame.K_4:
                    self.robot.set_tilt(0, math.radians(-25))
                elif event.key == pygame.K_5:
                    self.robot.set_tilt(math.radians(20), math.radians(20))
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    self.mouse_dragging = True
                    self.last_mouse_pos = event.pos
                elif event.button == 4:
                    self.camera_distance = max(50, self.camera_distance - 10)
                elif event.button == 5:
                    self.camera_distance = min(500, self.camera_distance + 10)
            
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    self.mouse_dragging = False
            
            elif event.type == pygame.MOUSEMOTION:
                if self.mouse_dragging:
                    dx = event.pos[0] - self.last_mouse_pos[0]
                    dy = event.pos[1] - self.last_mouse_pos[1]
                    self.camera_angle_y += dx * 0.5
                    self.camera_angle_x = max(-89, min(89, self.camera_angle_x + dy * 0.5))
                    self.last_mouse_pos = event.pos
            
            elif event.type == pygame.MOUSEWHEEL:
                self.camera_distance = max(50, min(500, self.camera_distance - event.y * 10))
        
        return True
    
    def render(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        self._update_camera()
        self._draw_grid()
        self._draw_axes()
        self._draw_mount()
        self._draw_robot()
        pygame.display.flip()
    
    def run(self):
        clock = pygame.time.Clock()
        running = True
        
        print("\n" + "=" * 70)
        print("  软体机械臂 PCC 模型渲染器 - 倒挂配置 (正确约束)")
        print("  (Piecewise Constant Curvature)")
        print("=" * 70)
        print(f"  三角形边长: {TRIANGLE_SIDE} cm")
        print(f"  三角形厚度: {TRIANGLE_THICKNESS} cm")
        print(f"  弧长范围: {ACTUATOR_MIN} - {ACTUATOR_MAX} cm")
        print(f"  最大弯曲角: {math.degrees(MAX_BEND_ANGLE):.0f}°")
        print("\n  PCC 模型自由度:")
        print("    - φ (phi): 弯曲方向")
        print("    - θ (theta_bend): 弯曲角度")
        print("    - s: 弧长")
        print("    注意: 单段 PCC 无法产生扭转(Yaw)!")
        print("\n  控制方式 (倾斜角度):")
        print("    Q/A: 向 X+/X- 方向弯曲")
        print("    W/S: 向 Y+/Y- 方向弯曲")
        print("    ↑/↓: 弧长 +/-")
        print("\n  预设姿态:")
        print("    1: 向 X+ 弯曲    2: 向 X- 弯曲")
        print("    3: 向 Y+ 弯曲    4: 向 Y- 弯曲")
        print("    5: 对角弯曲")
        print("\n  其他控制:")
        print("    C: 切换中心线   M: 切换支架")
        print("    P: 验证垂直性   R: 重置")
        print("    ESC: 退出")
        print("=" * 70 + "\n")
        
        while running:
            running = self.handle_events()
            self.render()
            
            r = self.robot
            result = r.pcc_result
            print(f"\rφ={math.degrees(r.phi):+6.1f}° "
                  f"θ={math.degrees(r.theta_bend):5.1f}° "
                  f"s={r.arc_length:.1f}cm "
                  f"κ={result['kappa']:.4f} | "
                  f"L=[{result['actuator_lengths'][0]:.1f}, "
                  f"{result['actuator_lengths'][1]:.1f}, "
                  f"{result['actuator_lengths'][2]:.1f}]",
                  end="", flush=True)
            
            clock.tick(60)
        
        print("\n")
        pygame.quit()


# ============================================================
# API 函数
# ============================================================

def forward_kinematics_pcc(phi, theta_bend, arc_length, triangle_side=TRIANGLE_SIDE):
    """
    PCC 正向运动学 - 从 PCC 参数计算机械臂状态
    
    参数：
        phi: 弯曲方向角（弧度），0 = X+ 方向，π/2 = Y+ 方向
        theta_bend: 弯曲角度（弧度）
        arc_length: 中轴弧长 (cm)
        triangle_side: 三角形边长 (cm)
    
    返回：
        dict: PCC 参数和几何信息
    
    示例：
        >>> result = forward_kinematics_pcc(0, math.radians(30), 50)
        >>> print(f"向 X+ 弯曲 30°: 曲率={result['kappa']:.4f}")
        >>> print(f"Actuator 长度: {result['actuator_lengths']}")
    """
    fk = PCCForwardKinematics(triangle_side)
    return fk.forward_kinematics(phi, theta_bend, arc_length)


def forward_kinematics_tilt(tilt_x, tilt_y, arc_length, triangle_side=TRIANGLE_SIDE):
    """
    从倾斜角度计算 PCC 状态
    
    参数：
        tilt_x: X 方向倾斜角（弧度）
        tilt_y: Y 方向倾斜角（弧度）
        arc_length: 中轴弧长 (cm)
    
    说明：
        tilt_x 和 tilt_y 会自动转换为 PCC 参数 (phi, theta_bend)
        theta_bend = sqrt(tilt_x² + tilt_y²)
        phi = atan2(tilt_y, tilt_x)
    """
    fk = PCCForwardKinematics(triangle_side)
    return fk.forward_kinematics_from_tilt(tilt_x, tilt_y, arc_length)


def main():
    renderer = Renderer()
    renderer.run()


if __name__ == "__main__":
    main()
