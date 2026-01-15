"""
软体机械臂 PCC 模型渲染器
基于 Piecewise Constant Curvature 模型
两块正三角形板子通过三根气动 origami actuator 连接
Actuator 是等曲率弧形
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

# 窗口设置
WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 800

# 颜色定义 (RGB, 0-1 范围)
COLOR_TRIANGLE_BASE = (0.2, 0.6, 0.9, 0.85)     # 底部三角形 - 蓝色
COLOR_TRIANGLE_TOP = (0.9, 0.4, 0.2, 0.85)      # 顶部三角形 - 橙色
COLOR_TRIANGLE_SIDE = (0.4, 0.4, 0.5, 0.7)      # 三棱柱侧面
COLOR_ACTUATOR_1 = (0.95, 0.25, 0.3, 1.0)       # actuator 1 - 红色
COLOR_ACTUATOR_2 = (0.25, 0.95, 0.35, 1.0)      # actuator 2 - 绿色
COLOR_ACTUATOR_3 = (0.35, 0.25, 0.95, 1.0)      # actuator 3 - 蓝色
COLOR_GRID = (0.3, 0.3, 0.3, 0.5)               # 网格颜色


def get_equilateral_triangle_vertices(side_length, center_z=0):
    """
    获取正三角形顶点坐标
    一个顶点在 x 轴正方向上
    三角形中心在原点
    """
    # 正三角形外接圆半径
    R = side_length / math.sqrt(3)
    
    # 三个顶点，第一个在 x 轴正方向
    vertices = []
    for i in range(3):
        angle = i * 2 * math.pi / 3  # 0°, 120°, 240°
        x = R * math.cos(angle)
        y = R * math.sin(angle)
        z = center_z
        vertices.append((x, y, z))
    
    return vertices


def calculate_arc_points(start, end, arc_length, num_segments=ARC_SEGMENTS):
    """
    计算等曲率弧形的点
    给定起点、终点和弧长，计算弧形上的点
    
    对于等曲率弧（圆弧），弧长 L = R * theta
    其中 R 是曲率半径，theta 是圆心角
    
    弦长 chord = 2 * R * sin(theta/2)
    所以 chord / L = 2 * sin(theta/2) / theta
    """
    start = np.array(start)
    end = np.array(end)
    
    # 计算弦长（起点到终点的直线距离）
    chord_length = np.linalg.norm(end - start)
    
    # 如果弧长约等于弦长，几乎是直线
    if abs(arc_length - chord_length) < 0.01:
        # 返回直线上的点
        points = []
        for i in range(num_segments + 1):
            t = i / num_segments
            point = start + t * (end - start)
            points.append(tuple(point))
        return points
    
    # 对于圆弧：chord = 2 * R * sin(theta/2), arc = R * theta
    # 所以 chord/arc = 2*sin(theta/2)/theta
    # 需要数值求解 theta
    
    ratio = chord_length / arc_length
    
    # 数值求解 theta (圆心角)
    # 使用牛顿法或二分法
    def f(theta):
        if abs(theta) < 0.0001:
            return 1.0 - ratio
        return 2 * math.sin(theta / 2) / theta - ratio
    
    # 二分法求解
    theta_low, theta_high = 0.001, math.pi * 1.99
    for _ in range(50):
        theta_mid = (theta_low + theta_high) / 2
        if f(theta_mid) > 0:
            theta_low = theta_mid
        else:
            theta_high = theta_mid
    
    theta = theta_mid
    R = arc_length / theta  # 曲率半径
    
    # 计算弧的几何参数
    # 弧在起点和终点之间弯曲，需要确定弯曲方向
    
    # 起点到终点的方向向量
    direction = end - start
    direction_norm = direction / chord_length
    
    # 中点
    midpoint = (start + end) / 2
    
    # 需要确定垂直于弦的方向（弧弯曲的方向）
    # 对于 3D，我们选择一个合理的垂直方向
    # 默认让弧向外弯曲（远离中心轴）
    
    # 计算一个垂直于 direction 的向量
    # 优先选择使弧向外凸出的方向
    if abs(direction_norm[2]) < 0.99:
        # 使用 z 轴作为参考
        up = np.array([0, 0, 1])
    else:
        # 如果 direction 接近 z 轴，使用 x 轴
        up = np.array([1, 0, 0])
    
    # 计算垂直于弦的方向
    perpendicular = np.cross(direction_norm, up)
    perp_norm = np.linalg.norm(perpendicular)
    if perp_norm > 0.001:
        perpendicular = perpendicular / perp_norm
    else:
        perpendicular = np.array([1, 0, 0])
    
    # 圆心到弦中点的距离
    h = R * math.cos(theta / 2)
    
    # 弧的凸出高度
    sagitta = R - h
    
    # 圆心位置 (在弦的垂直方向上，距离中点 h)
    # 让弧向外凸出（远离原点方向）
    midpoint_from_origin = midpoint[:2]  # xy 平面上的位置
    mid_dist = np.linalg.norm(midpoint_from_origin)
    
    if mid_dist > 0.001:
        # 让弧向外凸出
        outward = np.array([midpoint_from_origin[0]/mid_dist, 
                           midpoint_from_origin[1]/mid_dist, 0])
        # 计算 perpendicular 在 outward 方向的分量
        dot_product = np.dot(perpendicular, outward)
        if dot_product < 0:
            perpendicular = -perpendicular
    
    center = midpoint - perpendicular * h
    
    # 生成弧上的点
    points = []
    
    # 从起点到终点绕圆心旋转
    start_vec = start - center
    
    # 旋转轴：垂直于弧所在平面的方向
    rotation_axis = np.cross(start - center, end - center)
    rot_axis_norm = np.linalg.norm(rotation_axis)
    if rot_axis_norm > 0.001:
        rotation_axis = rotation_axis / rot_axis_norm
    else:
        rotation_axis = np.array([0, 0, 1])
    
    for i in range(num_segments + 1):
        t = i / num_segments
        angle = t * theta
        
        # 使用罗德里格斯旋转公式
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        
        v = start_vec
        k = rotation_axis
        
        rotated = v * cos_a + np.cross(k, v) * sin_a + k * np.dot(k, v) * (1 - cos_a)
        point = center + rotated
        points.append(tuple(point))
    
    return points


class SoftRobotArm:
    """软体机械臂类"""
    
    def __init__(self):
        # 初始 actuator 长度 (都设为中间值)
        self.actuator_lengths = [50.0, 50.0, 50.0]  # cm
        
        # 底部三角形的 z 位置（考虑厚度）
        # 底部三棱柱的底面在 z=0，顶面在 z=TRIANGLE_THICKNESS
        self.base_z_bottom = 0.0
        self.base_z_top = TRIANGLE_THICKNESS
        
        # 底部三角形顶点
        self.base_vertices_bottom = get_equilateral_triangle_vertices(
            TRIANGLE_SIDE, center_z=self.base_z_bottom)
        self.base_vertices_top = get_equilateral_triangle_vertices(
            TRIANGLE_SIDE, center_z=self.base_z_top)
        
        # 顶部三角形顶点 (会根据 actuator 长度计算)
        self._calculate_top_vertices()
    
    def _calculate_top_vertices(self):
        """
        根据 actuator 长度计算顶部三角形位置
        简化模型：顶部三角形保持平行，每个顶点的 z 坐标由对应 actuator 决定
        """
        # 简化模型：顶部三角形每个顶点直接在底部顶点正上方
        # z 坐标 = 底部顶面 z + actuator 长度
        
        self.top_vertices_bottom = []
        self.top_vertices_top = []
        
        for i, base_v in enumerate(self.base_vertices_top):
            # 顶部三棱柱底面
            x = base_v[0]
            y = base_v[1]
            z_bottom = self.base_z_top + self.actuator_lengths[i]
            z_top = z_bottom + TRIANGLE_THICKNESS
            
            self.top_vertices_bottom.append((x, y, z_bottom))
            self.top_vertices_top.append((x, y, z_top))
    
    def set_actuator_length(self, index, length):
        """设置 actuator 长度"""
        length = max(ACTUATOR_MIN, min(ACTUATOR_MAX, length))
        self.actuator_lengths[index] = length
        self._calculate_top_vertices()
    
    def get_actuator_arc_points(self, index):
        """获取 actuator 弧形的点"""
        # actuator 连接底部三棱柱顶面顶点到顶部三棱柱底面顶点
        start = self.base_vertices_top[index]
        end = self.top_vertices_bottom[index]
        arc_length = self.actuator_lengths[index]
        
        return calculate_arc_points(start, end, arc_length)


class Renderer:
    """OpenGL 渲染器"""
    
    def __init__(self):
        pygame.init()
        pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), DOUBLEBUF | OPENGL)
        pygame.display.set_caption("软体机械臂 PCC 模型渲染器")
        
        # 设置 OpenGL
        self._setup_opengl()
        
        # 相机参数
        self.camera_distance = 180.0
        self.camera_angle_x = 25.0  # 俯仰角
        self.camera_angle_y = 45.0  # 偏航角
        
        # 鼠标控制
        self.mouse_dragging = False
        self.last_mouse_pos = (0, 0)
        
        # 机械臂模型
        self.robot = SoftRobotArm()
    
    def _setup_opengl(self):
        """设置 OpenGL 参数"""
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_LINE_SMOOTH)
        glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)
        
        # 启用光照
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        
        # 设置光源
        glLightfv(GL_LIGHT0, GL_POSITION, (100, 200, 100, 1))
        glLightfv(GL_LIGHT0, GL_AMBIENT, (0.3, 0.3, 0.3, 1))
        glLightfv(GL_LIGHT0, GL_DIFFUSE, (0.8, 0.8, 0.8, 1))
        
        # 设置背景颜色 - 深色主题
        glClearColor(0.06, 0.06, 0.1, 1.0)
        
        # 设置投影
        glMatrixMode(GL_PROJECTION)
        gluPerspective(45, WINDOW_WIDTH / WINDOW_HEIGHT, 0.1, 1000.0)
        glMatrixMode(GL_MODELVIEW)
    
    def _update_camera(self):
        """更新相机视角"""
        glLoadIdentity()
        
        # 计算相机位置
        rad_x = math.radians(self.camera_angle_x)
        rad_y = math.radians(self.camera_angle_y)
        
        cam_x = self.camera_distance * math.cos(rad_x) * math.sin(rad_y)
        cam_y = self.camera_distance * math.sin(rad_x)
        cam_z = self.camera_distance * math.cos(rad_x) * math.cos(rad_y)
        
        # 看向场景中心（稍微向上偏移）
        look_at_height = 30
        gluLookAt(cam_x, cam_z, cam_y,
                  0, look_at_height, 0,
                  0, 1, 0)
    
    def _draw_grid(self):
        """绘制地面网格"""
        glDisable(GL_LIGHTING)
        glColor4f(*COLOR_GRID)
        glBegin(GL_LINES)
        
        grid_size = 100
        grid_step = 10
        
        for i in range(-grid_size, grid_size + 1, grid_step):
            glVertex3f(i, 0, -grid_size)
            glVertex3f(i, 0, grid_size)
            glVertex3f(-grid_size, 0, i)
            glVertex3f(grid_size, 0, i)
        
        glEnd()
        glEnable(GL_LIGHTING)
    
    def _draw_axes(self):
        """绘制坐标轴"""
        glDisable(GL_LIGHTING)
        axis_length = 30.0
        
        glLineWidth(2.5)
        glBegin(GL_LINES)
        
        # X 轴 - 红色
        glColor4f(1.0, 0.3, 0.3, 1.0)
        glVertex3f(0, 0, 0)
        glVertex3f(axis_length, 0, 0)
        
        # Y 轴 - 绿色 (模型的 Y -> OpenGL 的 Z)
        glColor4f(0.3, 1.0, 0.3, 1.0)
        glVertex3f(0, 0, 0)
        glVertex3f(0, 0, axis_length)
        
        # Z 轴 - 蓝色 (模型的 Z -> OpenGL 的 Y, 向上)
        glColor4f(0.3, 0.3, 1.0, 1.0)
        glVertex3f(0, 0, 0)
        glVertex3f(0, axis_length, 0)
        
        glEnd()
        glLineWidth(1.0)
        glEnable(GL_LIGHTING)
    
    def _to_gl_coords(self, v):
        """将模型坐标 (x, y, z) 转换为 OpenGL 坐标 (x, z, y)"""
        return (v[0], v[2], v[1])
    
    def _draw_triangular_prism(self, bottom_vertices, top_vertices, 
                                top_color, side_color):
        """绘制三棱柱"""
        # 转换坐标
        gl_bottom = [self._to_gl_coords(v) for v in bottom_vertices]
        gl_top = [self._to_gl_coords(v) for v in top_vertices]
        
        # 绘制顶面
        glColor4f(*top_color)
        glBegin(GL_TRIANGLES)
        # 计算法向量
        v0 = np.array(gl_top[0])
        v1 = np.array(gl_top[1])
        v2 = np.array(gl_top[2])
        normal = np.cross(v1 - v0, v2 - v0)
        normal = normal / np.linalg.norm(normal)
        glNormal3f(*normal)
        for v in gl_top:
            glVertex3f(*v)
        glEnd()
        
        # 绘制底面
        glBegin(GL_TRIANGLES)
        normal = -normal
        glNormal3f(*normal)
        for v in reversed(gl_bottom):
            glVertex3f(*v)
        glEnd()
        
        # 绘制三个侧面
        glColor4f(*side_color)
        for i in range(3):
            j = (i + 1) % 3
            
            # 计算侧面法向量
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
        
        # 绘制边框
        glDisable(GL_LIGHTING)
        glColor4f(1.0, 1.0, 1.0, 0.8)
        glLineWidth(1.5)
        
        # 顶面边框
        glBegin(GL_LINE_LOOP)
        for v in gl_top:
            glVertex3f(*v)
        glEnd()
        
        # 底面边框
        glBegin(GL_LINE_LOOP)
        for v in gl_bottom:
            glVertex3f(*v)
        glEnd()
        
        # 垂直边
        glBegin(GL_LINES)
        for i in range(3):
            glVertex3f(*gl_bottom[i])
            glVertex3f(*gl_top[i])
        glEnd()
        
        glLineWidth(1.0)
        glEnable(GL_LIGHTING)
        
        # 绘制顶点球
        for v in gl_top:
            self._draw_sphere(v, 1.2, (1.0, 1.0, 1.0, 1.0))
        for v in gl_bottom:
            self._draw_sphere(v, 1.2, (0.8, 0.8, 0.8, 1.0))
    
    def _draw_sphere(self, position, radius, color):
        """绘制球体"""
        glColor4f(*color)
        glPushMatrix()
        glTranslatef(*position)
        
        quadric = gluNewQuadric()
        gluSphere(quadric, radius, 16, 16)
        gluDeleteQuadric(quadric)
        
        glPopMatrix()
    
    def _draw_arc(self, points, color, line_width=4.0):
        """绘制弧形"""
        glDisable(GL_LIGHTING)
        
        # 转换坐标
        gl_points = [self._to_gl_coords(p) for p in points]
        
        # 绘制弧形线条
        glColor4f(*color)
        glLineWidth(line_width)
        
        glBegin(GL_LINE_STRIP)
        for p in gl_points:
            glVertex3f(*p)
        glEnd()
        
        glLineWidth(1.0)
        
        # 在弧的中点绘制小球
        mid_idx = len(gl_points) // 2
        if mid_idx < len(gl_points):
            self._draw_sphere(gl_points[mid_idx], 1.5, color)
        
        glEnable(GL_LIGHTING)
    
    def _draw_actuator_tube(self, points, color, radius=1.0):
        """绘制圆管形状的 actuator（更好的 3D 效果）"""
        gl_points = [self._to_gl_coords(p) for p in points]
        
        # 绘制圆管
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
            
            # 找一个垂直于 direction 的向量
            if abs(direction[1]) < 0.99:
                up = np.array([0, 1, 0])
            else:
                up = np.array([1, 0, 0])
            
            right = np.cross(direction, up)
            right = right / np.linalg.norm(right)
            up = np.cross(right, direction)
            
            # 绘制圆管段
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
        
        # 在端点绘制球体
        self._draw_sphere(gl_points[0], radius * 1.5, color)
        self._draw_sphere(gl_points[-1], radius * 1.5, color)
    
    def _draw_robot(self):
        """绘制机械臂"""
        # 绘制底部三棱柱
        self._draw_triangular_prism(
            self.robot.base_vertices_bottom,
            self.robot.base_vertices_top,
            COLOR_TRIANGLE_BASE,
            COLOR_TRIANGLE_SIDE
        )
        
        # 绘制顶部三棱柱
        self._draw_triangular_prism(
            self.robot.top_vertices_bottom,
            self.robot.top_vertices_top,
            COLOR_TRIANGLE_TOP,
            COLOR_TRIANGLE_SIDE
        )
        
        # 绘制 actuators（等曲率弧形）
        colors = [COLOR_ACTUATOR_1, COLOR_ACTUATOR_2, COLOR_ACTUATOR_3]
        
        for i in range(3):
            arc_points = self.robot.get_actuator_arc_points(i)
            # 使用圆管绘制更好的 3D 效果
            self._draw_actuator_tube(arc_points, colors[i], radius=0.8)
    
    def handle_events(self):
        """处理输入事件"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                elif event.key == pygame.K_r:
                    # 重置
                    self.robot.actuator_lengths = [50.0, 50.0, 50.0]
                    self.robot._calculate_top_vertices()
                elif event.key == pygame.K_1:
                    self.robot.set_actuator_length(0, self.robot.actuator_lengths[0] + 2)
                elif event.key == pygame.K_2:
                    self.robot.set_actuator_length(0, self.robot.actuator_lengths[0] - 2)
                elif event.key == pygame.K_3:
                    self.robot.set_actuator_length(1, self.robot.actuator_lengths[1] + 2)
                elif event.key == pygame.K_4:
                    self.robot.set_actuator_length(1, self.robot.actuator_lengths[1] - 2)
                elif event.key == pygame.K_5:
                    self.robot.set_actuator_length(2, self.robot.actuator_lengths[2] + 2)
                elif event.key == pygame.K_6:
                    self.robot.set_actuator_length(2, self.robot.actuator_lengths[2] - 2)
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # 左键
                    self.mouse_dragging = True
                    self.last_mouse_pos = event.pos
                elif event.button == 4:  # 滚轮上
                    self.camera_distance = max(50, self.camera_distance - 10)
                elif event.button == 5:  # 滚轮下
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
        """渲染一帧"""
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        self._update_camera()
        self._draw_grid()
        self._draw_axes()
        self._draw_robot()
        
        pygame.display.flip()
    
    def run(self):
        """主循环"""
        clock = pygame.time.Clock()
        running = True
        
        print("\n" + "=" * 55)
        print("  软体机械臂 PCC 模型渲染器")
        print("  (Piecewise Constant Curvature)")
        print("=" * 55)
        print(f"  三角形边长: {TRIANGLE_SIDE} cm")
        print(f"  三角形厚度: {TRIANGLE_THICKNESS} cm")
        print(f"  Actuator 范围: {ACTUATOR_MIN} - {ACTUATOR_MAX} cm")
        print(f"  Actuator 形状: 等曲率弧形")
        print("\n  控制说明:")
        print("    鼠标拖拽: 旋转视角")
        print("    滚轮: 缩放")
        print("    1/2: Actuator 1 (红) +/-")
        print("    3/4: Actuator 2 (绿) +/-")
        print("    5/6: Actuator 3 (蓝) +/-")
        print("    R: 重置所有 actuator")
        print("    ESC: 退出")
        print("=" * 55 + "\n")
        
        while running:
            running = self.handle_events()
            self.render()
            clock.tick(60)
        
        pygame.quit()


def main():
    renderer = Renderer()
    renderer.run()


if __name__ == "__main__":
    main()
