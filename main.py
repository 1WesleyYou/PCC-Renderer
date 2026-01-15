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


# ============================================================
# 四元数工具函数
# ============================================================

def quaternion_from_euler(roll, pitch, yaw):
    """
    从欧拉角创建四元数 (w, x, y, z)
    roll: 绕 X 轴
    pitch: 绕 Y 轴  
    yaw: 绕 Z 轴
    """
    cr, sr = math.cos(roll/2), math.sin(roll/2)
    cp, sp = math.cos(pitch/2), math.sin(pitch/2)
    cy, sy = math.cos(yaw/2), math.sin(yaw/2)
    
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    
    return np.array([w, x, y, z])


def quaternion_to_rotation_matrix(q):
    """
    四元数转旋转矩阵
    q: (w, x, y, z)
    """
    w, x, y, z = q
    
    # 归一化
    norm = math.sqrt(w*w + x*x + y*y + z*z)
    if norm < 1e-10:
        return np.eye(3)
    w, x, y, z = w/norm, x/norm, y/norm, z/norm
    
    R = np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)]
    ])
    return R


def rotation_matrix_to_quaternion(R):
    """
    旋转矩阵转四元数 (w, x, y, z)
    """
    trace = R[0,0] + R[1,1] + R[2,2]
    
    if trace > 0:
        s = 0.5 / math.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2,1] - R[1,2]) * s
        y = (R[0,2] - R[2,0]) * s
        z = (R[1,0] - R[0,1]) * s
    elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
        s = 2.0 * math.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2])
        w = (R[2,1] - R[1,2]) / s
        x = 0.25 * s
        y = (R[0,1] + R[1,0]) / s
        z = (R[0,2] + R[2,0]) / s
    elif R[1,1] > R[2,2]:
        s = 2.0 * math.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2])
        w = (R[0,2] - R[2,0]) / s
        x = (R[0,1] + R[1,0]) / s
        y = 0.25 * s
        z = (R[1,2] + R[2,1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1])
        w = (R[1,0] - R[0,1]) / s
        x = (R[0,2] + R[2,0]) / s
        y = (R[1,2] + R[2,1]) / s
        z = 0.25 * s
    
    return np.array([w, x, y, z])


def quaternion_multiply(q1, q2):
    """
    四元数乘法 q1 * q2
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    
    return np.array([w, x, y, z])


def quaternion_conjugate(q):
    """四元数共轭"""
    return np.array([q[0], -q[1], -q[2], -q[3]])


def rotation_matrix_to_pcc_params(R_target, R_base):
    """
    从目标旋转矩阵（世界坐标系）计算相对于基座的 PCC 参数
    
    PCC 约束：
    - 末端法向量由弯曲决定
    - 末端 Z 轴（法向量）= R_base @ R_local @ [0,0,-1]
    - R_local 只能是绕某个水平轴旋转（弯曲）
    
    参数:
        R_target: 目标末端旋转矩阵（世界坐标系）
        R_base: 当前节基座的旋转矩阵（世界坐标系）
    
    返回:
        phi: 弯曲方向角
        theta_bend: 弯曲角度
    """
    # 计算相对旋转：R_relative = R_base^T @ R_target
    R_relative = R_base.T @ R_target
    
    # 目标末端法向量（在基座局部坐标系中）
    # 初始法向量是 [0, 0, -1]，旋转后是 R_relative @ [0, 0, -1]
    target_normal_local = R_relative @ np.array([0, 0, -1])
    
    # 在 PCC 模型中，弯曲后的法向量在 XZ 平面内旋转（取决于 phi）
    # 法向量 = [sin(theta)*cos(phi), sin(theta)*sin(phi), -cos(theta)]
    # 其中 theta = theta_bend
    
    # 从目标法向量提取 phi 和 theta_bend
    nx, ny, nz = target_normal_local
    
    # theta_bend = arccos(-nz)，因为 nz = -cos(theta_bend)
    # 限制在有效范围内
    nz_clamped = np.clip(nz, -1, 1)
    theta_bend = math.acos(-nz_clamped)
    
    # phi = atan2(ny, nx)，因为 nx = sin(theta)*cos(phi), ny = sin(theta)*sin(phi)
    if theta_bend > 1e-6:
        phi = math.atan2(ny, nx)
    else:
        phi = 0  # 无弯曲时 phi 无意义
    
    return phi, theta_bend


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
        
        同心圆弧模型：
        - 所有圆弧共享同一个曲率中心
        - 每个 actuator 有不同的弯曲半径
        - 所有圆弧弯曲相同的角度 theta_bend
        - 两端都垂直于板
        """
        kappa = fk_result['kappa']
        phi = fk_result['phi']
        theta_bend = fk_result['theta_bend']
        s = fk_result['s']
        actuator_angle = self.actuator_angles[actuator_index]
        
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
                z = -t * s
                points.append(np.array([P0[0], P0[1], z]))
            return points
        
        # 中心线的弯曲半径
        R_center = s / theta_bend
        
        # 弯曲方向和轴
        bend_dir = np.array([math.cos(phi), math.sin(phi), 0])
        bend_axis = np.array([-math.sin(phi), math.cos(phi), 0])
        
        # 共同的曲率中心（在原点沿 -bend_dir 方向偏移 R_center）
        center = -R_center * bend_dir
        
        # 生成圆弧点
        for i in range(1, num_segments + 1):
            t = i / num_segments
            angle = t * theta_bend
            
            # 旋转矩阵（绕弯曲轴旋转）
            rot = rotation_matrix_axis_angle(bend_axis, angle)
            
            # 起点相对于曲率中心的向量
            v0 = P0 - center
            
            # 旋转后的向量
            v = rot @ v0
            
            # 新的点
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
# PCC 模块类（单节机械臂）
# ============================================================

class PCCModule:
    """
    PCC 模块（单节机械臂）
    
    每个模块有自己的 PCC 参数和几何状态
    可以指定基座的位置和姿态
    """
    
    def __init__(self, module_id=0):
        self.module_id = module_id
        self.fk = PCCForwardKinematics(TRIANGLE_SIDE)
        
        # PCC 参数
        self.phi = 0.0              # 弯曲方向
        self.theta_bend = 0.0       # 弯曲角度
        self.arc_length = DEFAULT_ARC_LENGTH
        
        # 基座变换（用于多节连接）
        self.base_position = np.array([0.0, 0.0, 0.0])
        self.base_rotation = np.eye(3)
        
        # 几何数据
        self.pcc_result = None
        self.fixed_vertices_top = []
        self.fixed_vertices_bottom = []
        self.free_vertices_top = []
        self.free_vertices_bottom = []
        self.actuator_arcs = []
        self.center_arc = []
        self.end_tangent = np.array([0, 0, -1])
        
        self._update_geometry()
    
    def set_pcc_params(self, phi, theta_bend, arc_length):
        """直接设置 PCC 参数"""
        self.phi = phi
        self.theta_bend = max(0, min(MAX_BEND_ANGLE, theta_bend))
        self.arc_length = max(ACTUATOR_MIN, min(ACTUATOR_MAX, arc_length))
        self._update_geometry()
    
    def set_base_transform(self, position, rotation):
        """设置基座变换（用于多节连接）"""
        self.base_position = np.array(position)
        self.base_rotation = np.array(rotation)
        self._update_geometry()
    
    def get_end_transform(self):
        """获取末端变换（用于连接下一节）"""
        if self.pcc_result is None:
            return self.base_position.copy(), self.base_rotation.copy()
        
        # 末端位置：自由端三角形中心
        end_pos = np.mean(self.free_vertices_bottom, axis=0)
        
        # 末端旋转：基座旋转 @ 本节的旋转
        local_rotation = self.pcc_result['end_rotation']
        end_rotation = self.base_rotation @ local_rotation
        
        return end_pos, end_rotation
    
    def _transform_point(self, local_point):
        """将局部坐标转换为全局坐标"""
        return self.base_position + self.base_rotation @ local_point
    
    def _update_geometry(self):
        """更新几何状态"""
        self.pcc_result = self.fk.forward_kinematics(
            self.phi, self.theta_bend, self.arc_length
        )
        
        # 局部坐标系中的固定端三棱柱
        fixed_top_local = get_equilateral_triangle_vertices(
            TRIANGLE_SIDE, center=np.array([0, 0, 0])
        )
        fixed_bottom_local = get_equilateral_triangle_vertices(
            TRIANGLE_SIDE, center=np.array([0, 0, -TRIANGLE_THICKNESS])
        )
        
        # 转换到全局坐标
        self.fixed_vertices_top = [self._transform_point(v) for v in fixed_top_local]
        self.fixed_vertices_bottom = [self._transform_point(v) for v in fixed_bottom_local]
        
        # 计算 Actuator 弧线（局部坐标，从 z=0 开始）
        local_actuator_arcs = []
        for i in range(3):
            arc = self.fk.get_actuator_arc_points(self.pcc_result, i)
            # 不调整，从 z=0 开始（与中心线一致）
            local_actuator_arcs.append(arc)
        
        # 转换到全局坐标
        self.actuator_arcs = []
        for arc in local_actuator_arcs:
            global_arc = [self._transform_point(pt) for pt in arc]
            self.actuator_arcs.append(global_arc)
        
        # 中心线（从固定端顶面开始，即 base_position，与 actuator 一致）
        center_arc_local = self.fk.get_center_arc_points(self.pcc_result)
        self.center_arc = [self._transform_point(pt) for pt in center_arc_local]
        
        # 自由端三角形顶点 = 圆弧终点
        self.free_vertices_top = []
        for i in range(3):
            top_vertex = self.actuator_arcs[i][-1].copy()
            self.free_vertices_top.append(top_vertex)
        
        # 板的法向量
        arc = self.actuator_arcs[0]
        if len(arc) >= 2:
            tangent = np.array(arc[-1]) - np.array(arc[-2])
            norm = np.linalg.norm(tangent)
            if norm > 1e-10:
                self.end_tangent = tangent / norm
            else:
                self.end_tangent = self.base_rotation @ np.array([0, 0, -1])
        else:
            self.end_tangent = self.base_rotation @ np.array([0, 0, -1])
        
        # 自由端底面顶点
        self.free_vertices_bottom = []
        for i in range(3):
            top_vertex = self.free_vertices_top[i]
            bottom_vertex = top_vertex + self.end_tangent * TRIANGLE_THICKNESS
            self.free_vertices_bottom.append(bottom_vertex)
        
        # 重新计算中心线：使用与 actuator 相同的旋转方法
        # 中心线从固定端底面中心开始
        fixed_bottom_center = np.mean(self.fixed_vertices_bottom, axis=0)
        free_bottom_center = np.mean(self.free_vertices_bottom, axis=0)
        
        kappa = self.pcc_result['kappa']
        theta_bend = self.pcc_result['theta_bend']
        phi = self.pcc_result['phi']
        s = self.pcc_result['s']
        
        self.center_arc = []
        num_segments = ARC_SEGMENTS
        
        if kappa < 1e-6:
            # 直线情况：从基座位置到自由端底面中心
            start_pt = self.base_position.copy()
            for i in range(num_segments + 1):
                t = i / num_segments
                pt = start_pt + t * (free_bottom_center - start_pt)
                self.center_arc.append(pt)
        else:
            R_center = s / theta_bend
            
            # 弯曲方向和轴（局部坐标）
            bend_dir = np.array([math.cos(phi), math.sin(phi), 0])
            bend_axis = np.array([-math.sin(phi), math.cos(phi), 0])
            
            # 曲率中心（局部坐标，在原点沿 -bend_dir 方向偏移 R_center）
            local_center = -R_center * bend_dir
            
            # 中心线起点（局部坐标，基座位置=固定端顶面中心）
            local_P0 = np.array([0, 0, 0])
            
            # 中心线弧长（从固定端底面到自由端底面）
            # 包括弧长 s + 两块板的厚度
            total_theta = theta_bend  # 基本弯曲角度
            
            for i in range(num_segments + 1):
                t = i / num_segments
                angle = t * total_theta
                
                # 旋转矩阵
                rot = rotation_matrix_axis_angle(bend_axis, angle)
                
                # 起点相对于曲率中心的向量
                v0 = local_P0 - local_center
                
                # 旋转后的向量
                v = rot @ v0
                
                # 新的点（局部坐标）
                local_pt = local_center + v
                
                # 转换到全局坐标
                global_pt = self._transform_point(local_pt)
                self.center_arc.append(global_pt)
            
            # 平滑插值延伸到自由端底面中心
            if len(self.center_arc) >= 2:
                last_pt = np.array(self.center_arc[-1])
                
                # 计算到自由端底面中心的距离
                dist = np.linalg.norm(free_bottom_center - last_pt)
                
                # 线性插值到 free_bottom_center
                num_extend = max(1, int(dist / 2))  # 每 2cm 一个点
                for i in range(1, num_extend + 1):
                    t = i / num_extend
                    extend_pt = last_pt + t * (free_bottom_center - last_pt)
                    self.center_arc.append(extend_pt)
            


# ============================================================
# 多节软体机械臂类
# ============================================================

class SoftRobotArm:
    """
    多节软体机械臂类（倒挂配置）
    
    由多个 PCCModule 串联组成
    """
    
    def __init__(self, num_modules=2):
        self.num_modules = num_modules
        self.modules = []
        
        for i in range(num_modules):
            module = PCCModule(module_id=i)
            self.modules.append(module)
        
        # 当前选中的模块（用于控制）
        self.active_module = 0
        
        self._update_chain()
    
    def _update_chain(self):
        """更新模块链的变换"""
        for i, module in enumerate(self.modules):
            if i == 0:
                # 第一节：基座在原点
                module.set_base_transform(
                    np.array([0.0, 0.0, 0.0]),
                    np.eye(3)
                )
            else:
                # 后续节：基座连接到前一节的末端
                prev_end_pos, prev_end_rot = self.modules[i-1].get_end_transform()
                module.set_base_transform(prev_end_pos, prev_end_rot)
    
    def set_module_params(self, module_idx, phi, theta_bend, arc_length):
        """设置指定模块的 PCC 参数"""
        if 0 <= module_idx < self.num_modules:
            self.modules[module_idx].set_pcc_params(phi, theta_bend, arc_length)
            # 更新后续模块的变换
            self._update_chain()
    
    def set_pcc_params(self, phi, theta_bend, arc_length):
        """设置当前活动模块的 PCC 参数（兼容旧接口）"""
        self.set_module_params(self.active_module, phi, theta_bend, arc_length)
    
    def set_all_modules_params(self, phi, theta_bend, arc_length):
        """设置所有模块相同的 PCC 参数"""
        for i in range(self.num_modules):
            self.modules[i].phi = phi
            self.modules[i].theta_bend = max(0, min(MAX_BEND_ANGLE, theta_bend))
            self.modules[i].arc_length = max(ACTUATOR_MIN, min(ACTUATOR_MAX, arc_length))
        self._update_chain()
    
    def set_module_world_orientation(self, module_idx, target_quaternion, arc_length=None):
        """
        使用世界坐标系的绝对姿态设置指定模块
        
        参数:
            module_idx: 模块索引
            target_quaternion: 目标四元数 (w, x, y, z)，世界坐标系
            arc_length: 弧长，如果为 None 则保持当前值
        """
        if module_idx < 0 or module_idx >= self.num_modules:
            return
        
        # 首先更新链以获取正确的基座变换
        self._update_chain()
        
        # 获取该模块的基座旋转矩阵
        R_base = self.modules[module_idx].base_rotation
        
        # 目标旋转矩阵（世界坐标系）
        R_target = quaternion_to_rotation_matrix(target_quaternion)
        
        # 计算相对于基座的 PCC 参数
        phi, theta_bend = rotation_matrix_to_pcc_params(R_target, R_base)
        
        # 限制 theta_bend 在有效范围内
        theta_bend = max(0, min(MAX_BEND_ANGLE, theta_bend))
        
        # 设置弧长
        if arc_length is None:
            arc_length = self.modules[module_idx].arc_length
        
        # 设置参数
        self.modules[module_idx].set_pcc_params(phi, theta_bend, arc_length)
        
        # 更新后续模块
        self._update_chain()
    
    def set_module_world_euler(self, module_idx, roll, pitch, yaw, arc_length=None):
        """
        使用世界坐标系的欧拉角设置指定模块
        
        参数:
            module_idx: 模块索引
            roll: 绕 X 轴旋转角度 (弧度)
            pitch: 绕 Y 轴旋转角度 (弧度)
            yaw: 绕 Z 轴旋转角度 (弧度)
            arc_length: 弧长
        """
        q = quaternion_from_euler(roll, pitch, yaw)
        self.set_module_world_orientation(module_idx, q, arc_length)
    
    def get_module_world_orientation(self, module_idx):
        """
        获取指定模块末端的世界坐标系姿态
        
        返回:
            四元数 (w, x, y, z)
        """
        if module_idx < 0 or module_idx >= self.num_modules:
            return np.array([1, 0, 0, 0])
        
        _, end_rot = self.modules[module_idx].get_end_transform()
        return rotation_matrix_to_quaternion(end_rot)
    
    def get_end_effector_world_orientation(self):
        """获取末端执行器的世界坐标系姿态"""
        return self.get_module_world_orientation(self.num_modules - 1)
    
    # 兼容旧接口的属性
    @property
    def pcc_result(self):
        return self.modules[self.active_module].pcc_result
    
    @property
    def phi(self):
        return self.modules[self.active_module].phi
    
    @property
    def theta_bend(self):
        return self.modules[self.active_module].theta_bend
    
    @property
    def arc_length(self):
        return self.modules[self.active_module].arc_length


# ============================================================
# 渲染器
# ============================================================

class Renderer:
    """OpenGL 渲染器"""
    
    def __init__(self):
        pygame.init()
        pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), DOUBLEBUF | OPENGL)
        pygame.display.set_caption("多节软体机械臂 PCC 模型 - 倒挂配置")
        
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
        
        # 第二节世界坐标系姿态控制
        # 存储目标欧拉角（世界坐标系）
        self.module2_world_roll = 0.0    # 绕世界 X 轴
        self.module2_world_pitch = 0.0   # 绕世界 Y 轴
        # 注意：PCC 不能产生 yaw，所以不存储 yaw
    
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
        
        # 为每个模块绘制末端法向量
        for i, module in enumerate(self.robot.modules):
            if module.pcc_result is None:
                continue
            
            # 末端中心
            end_center = np.mean(module.free_vertices_top, axis=0)
            end_normal = module.end_tangent
            
            start = self._to_gl_coords(end_center)
            end_pt = end_center + end_normal * 15
            end_gl = self._to_gl_coords(end_pt)
            
            # 不同模块用不同颜色
            if i == 0:
                glColor4f(1.0, 1.0, 0.0, 1.0)
            else:
                glColor4f(1.0, 0.5, 0.0, 1.0)
            
            glBegin(GL_LINES)
            glVertex3f(*start)
            glVertex3f(*end_gl)
            glEnd()
        
        glLineWidth(1.0)
        glEnable(GL_LIGHTING)
    
    def _draw_module(self, module, is_first=False, is_active=False):
        """绘制单个模块"""
        # 固定端颜色：第一节用蓝色，其他用灰色
        if is_first:
            fixed_color = COLOR_TRIANGLE_FIXED
        else:
            fixed_color = (0.5, 0.5, 0.6, 0.85)  # 中间连接板用灰色
        
        # 自由端颜色：活动模块用橙色，否则用较暗的橙色
        if is_active:
            free_color = COLOR_TRIANGLE_FREE
        else:
            free_color = (0.7, 0.35, 0.2, 0.85)
        
        # 绘制固定端三棱柱（每个模块都要画）
        self._draw_triangular_prism(
            module.fixed_vertices_bottom,
            module.fixed_vertices_top,
            fixed_color,
            COLOR_TRIANGLE_SIDE
        )
        
        # 绘制自由端三棱柱
        self._draw_triangular_prism(
            module.free_vertices_bottom,
            module.free_vertices_top,
            free_color,
            COLOR_TRIANGLE_SIDE
        )
        
        # 绘制 actuator
        colors = [COLOR_ACTUATOR_0, COLOR_ACTUATOR_1, COLOR_ACTUATOR_2]
        for i in range(3):
            self._draw_actuator_tube(module.actuator_arcs[i], colors[i], radius=0.8)
        
        # 绘制中心线
        if self.show_center_line:
            self._draw_center_line(module.center_arc, COLOR_CENTER_LINE, line_width=3.0)
    
    def _draw_robot(self):
        """绘制整个机器人（所有模块）"""
        for i, module in enumerate(self.robot.modules):
            is_first = (i == 0)
            is_active = (i == self.robot.active_module)
            self._draw_module(module, is_first=is_first, is_active=is_active)
        
        self._draw_normal_arrows()
    
    def _get_active_module(self):
        """获取当前活动模块"""
        return self.robot.modules[self.robot.active_module]
    
    def _handle_local_control(self, event, module, step):
        """处理局部坐标系控制（相对于基座）"""
        # Tilt X 控制 (Q/A) - 向 X 方向弯曲
        if event.key == pygame.K_q:
            tilt_x = module.theta_bend * math.cos(module.phi) + step
            tilt_y = module.theta_bend * math.sin(module.phi)
            new_theta = math.sqrt(tilt_x**2 + tilt_y**2)
            new_phi = math.atan2(tilt_y, tilt_x) if new_theta > 1e-6 else 0
            module.set_pcc_params(new_phi, new_theta, module.arc_length)
            self.robot._update_chain()
        elif event.key == pygame.K_a:
            tilt_x = module.theta_bend * math.cos(module.phi) - step
            tilt_y = module.theta_bend * math.sin(module.phi)
            new_theta = math.sqrt(tilt_x**2 + tilt_y**2)
            new_phi = math.atan2(tilt_y, tilt_x) if new_theta > 1e-6 else 0
            module.set_pcc_params(new_phi, new_theta, module.arc_length)
            self.robot._update_chain()
        
        # Tilt Y 控制 (W/S) - 向 Y 方向弯曲
        elif event.key == pygame.K_w:
            tilt_x = module.theta_bend * math.cos(module.phi)
            tilt_y = module.theta_bend * math.sin(module.phi) + step
            new_theta = math.sqrt(tilt_x**2 + tilt_y**2)
            new_phi = math.atan2(tilt_y, tilt_x) if new_theta > 1e-6 else 0
            module.set_pcc_params(new_phi, new_theta, module.arc_length)
            self.robot._update_chain()
        elif event.key == pygame.K_s:
            tilt_x = module.theta_bend * math.cos(module.phi)
            tilt_y = module.theta_bend * math.sin(module.phi) - step
            new_theta = math.sqrt(tilt_x**2 + tilt_y**2)
            new_phi = math.atan2(tilt_y, tilt_x) if new_theta > 1e-6 else 0
            module.set_pcc_params(new_phi, new_theta, module.arc_length)
            self.robot._update_chain()
    
    def _handle_world_control(self, event, step):
        """处理第二节世界坐标系控制"""
        # Q/A: 调整世界坐标系 roll (绕世界 X 轴)
        if event.key == pygame.K_q:
            self.module2_world_roll += step
            self._apply_world_orientation()
            self._print_current_state()
        elif event.key == pygame.K_a:
            self.module2_world_roll -= step
            self._apply_world_orientation()
            self._print_current_state()
        
        # W/S: 调整世界坐标系 pitch (绕世界 Y 轴)
        elif event.key == pygame.K_w:
            self.module2_world_pitch += step
            self._apply_world_orientation()
            self._print_current_state()
        elif event.key == pygame.K_s:
            self.module2_world_pitch -= step
            self._apply_world_orientation()
            self._print_current_state()
    
    def _apply_world_orientation(self):
        """应用第二节的世界坐标系目标姿态"""
        # 从世界坐标系欧拉角创建四元数
        # 注意：PCC 模型不能产生 yaw，所以只用 roll 和 pitch
        q = quaternion_from_euler(self.module2_world_roll, self.module2_world_pitch, 0)
        
        # 设置第二节的世界坐标系姿态
        arc_length = self.robot.modules[1].arc_length
        self.robot.set_module_world_orientation(1, q, arc_length)
    
    def _print_current_state(self):
        """打印当前状态信息"""
        module_idx = self.robot.active_module
        module = self.robot.modules[module_idx]
        
        print(f"  模块 {module_idx + 1}:")
        print(f"    φ = {math.degrees(module.phi):.1f}°")
        print(f"    θ_bend = {math.degrees(module.theta_bend):.1f}°")
        print(f"    弧长 = {module.arc_length:.1f} cm")
        
        if module_idx == 1:
            print(f"  世界坐标系目标:")
            print(f"    Roll = {math.degrees(self.module2_world_roll):.1f}°")
            print(f"    Pitch = {math.degrees(self.module2_world_pitch):.1f}°")
            
            # 显示实际末端姿态
            q = self.robot.get_module_world_orientation(1)
            R = quaternion_to_rotation_matrix(q)
            # 末端法向量（Z 轴）
            normal = R @ np.array([0, 0, -1])
            print(f"  实际末端法向量: [{normal[0]:.3f}, {normal[1]:.3f}, {normal[2]:.3f}]")
    
    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            
            elif event.type == pygame.KEYDOWN:
                step = math.radians(3)  # 3度步进
                module = self._get_active_module()
                module_idx = self.robot.active_module
                
                if event.key == pygame.K_ESCAPE:
                    return False
                elif event.key == pygame.K_r:
                    # 重置所有模块
                    for m in self.robot.modules:
                        m.set_pcc_params(0, 0, DEFAULT_ARC_LENGTH)
                    self.module2_world_roll = 0.0
                    self.module2_world_pitch = 0.0
                    self.robot._update_chain()
                
                # 切换活动模块 (TAB)
                elif event.key == pygame.K_TAB:
                    self.robot.active_module = (self.robot.active_module + 1) % self.robot.num_modules
                    mode_str = "世界坐标系" if self.robot.active_module > 0 else "局部坐标系"
                    print(f"\n当前控制: 模块 {self.robot.active_module + 1} ({mode_str})")
                    self._print_current_state()
                
                # WASD 控制
                # 第一节：局部坐标系（相对于基座）
                # 第二节：世界坐标系（绝对角度）
                elif module_idx == 0:
                    self._handle_local_control(event, module, step)
                else:
                    self._handle_world_control(event, step)
                
                # 弧长控制 (UP/DOWN) - 所有模块通用
                if event.key == pygame.K_UP:
                    module.set_pcc_params(module.phi, module.theta_bend, module.arc_length + 2)
                    self.robot._update_chain()
                elif event.key == pygame.K_DOWN:
                    module.set_pcc_params(module.phi, module.theta_bend, module.arc_length - 2)
                    self.robot._update_chain()
                
                # 切换显示
                elif event.key == pygame.K_c:
                    self.show_center_line = not self.show_center_line
                elif event.key == pygame.K_m:
                    self.show_mount = not self.show_mount
                elif event.key == pygame.K_p:
                    self.show_perpendicularity_check = not self.show_perpendicularity_check
                
                # 预设姿态（应用于当前模块）
                # 第一节：局部坐标系，第二节：世界坐标系
                elif event.key == pygame.K_1:
                    if module_idx == 0:
                        module.set_pcc_params(0, math.radians(25), module.arc_length)
                    else:
                        self.module2_world_roll = math.radians(25)
                        self.module2_world_pitch = 0
                        self._apply_world_orientation()
                    self.robot._update_chain()
                elif event.key == pygame.K_2:
                    if module_idx == 0:
                        module.set_pcc_params(math.pi, math.radians(25), module.arc_length)
                    else:
                        self.module2_world_roll = -math.radians(25)
                        self.module2_world_pitch = 0
                        self._apply_world_orientation()
                    self.robot._update_chain()
                elif event.key == pygame.K_3:
                    if module_idx == 0:
                        module.set_pcc_params(math.pi/2, math.radians(25), module.arc_length)
                    else:
                        self.module2_world_roll = 0
                        self.module2_world_pitch = math.radians(25)
                        self._apply_world_orientation()
                    self.robot._update_chain()
                elif event.key == pygame.K_4:
                    if module_idx == 0:
                        module.set_pcc_params(-math.pi/2, math.radians(25), module.arc_length)
                    else:
                        self.module2_world_roll = 0
                        self.module2_world_pitch = -math.radians(25)
                        self._apply_world_orientation()
                    self.robot._update_chain()
                elif event.key == pygame.K_5:
                    if module_idx == 0:
                        module.set_pcc_params(math.pi/4, math.radians(25), module.arc_length)
                    else:
                        self.module2_world_roll = math.radians(15)
                        self.module2_world_pitch = math.radians(15)
                        self._apply_world_orientation()
                    self.robot._update_chain()
            
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
        print("  多节软体机械臂 PCC 模型渲染器 - 倒挂配置")
        print("  (Piecewise Constant Curvature)")
        print("=" * 70)
        print(f"  模块数量: {self.robot.num_modules}")
        print(f"  三角形边长: {TRIANGLE_SIDE} cm")
        print(f"  三角形厚度: {TRIANGLE_THICKNESS} cm")
        print(f"  弧长范围: {ACTUATOR_MIN} - {ACTUATOR_MAX} cm")
        print(f"  最大弯曲角: {math.degrees(MAX_BEND_ANGLE):.0f}°")
        print("\n  控制方式:")
        print("    TAB: 切换控制的模块")
        print("    Q/A: 向 X+/X- 方向弯曲")
        print("    W/S: 向 Y+/Y- 方向弯曲")
        print("    ↑/↓: 弧长 +/-")
        print("\n  预设姿态 (当前模块):")
        print("    1: 向 X+ 弯曲    2: 向 X- 弯曲")
        print("    3: 向 Y+ 弯曲    4: 向 Y- 弯曲")
        print("    5: 对角弯曲")
        print("\n  其他控制:")
        print("    C: 切换中心线   M: 切换支架")
        print("    R: 重置全部     ESC: 退出")
        print("=" * 70 + "\n")
        
        while running:
            running = self.handle_events()
            self.render()
            
            # 显示当前模块的状态
            m = self._get_active_module()
            result = m.pcc_result
            print(f"\r[模块{self.robot.active_module+1}/{self.robot.num_modules}] "
                  f"φ={math.degrees(m.phi):+6.1f}° "
                  f"θ={math.degrees(m.theta_bend):5.1f}° "
                  f"s={m.arc_length:.1f}cm "
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
