"""
软体机械臂 PCC 模型渲染器
基于 Piecewise Constant Curvature 模型
两块正三角形板子通过三根气动 origami actuator 连接
"""

import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
import math

# 常量定义
TRIANGLE_SIDE = 40.0  # 正三角形边长 (cm)
ACTUATOR_MIN = 35.0   # actuator 最短长度 (cm)
ACTUATOR_MAX = 65.0   # actuator 最长长度 (cm)

# 窗口设置
WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 800

# 颜色定义 (RGB, 0-1 范围)
COLOR_TRIANGLE_BASE = (0.2, 0.6, 0.9, 0.8)      # 底部三角形 - 蓝色
COLOR_TRIANGLE_TOP = (0.9, 0.4, 0.2, 0.8)       # 顶部三角形 - 橙色
COLOR_ACTUATOR_1 = (0.9, 0.2, 0.3, 1.0)         # actuator 1 - 红色
COLOR_ACTUATOR_2 = (0.2, 0.9, 0.3, 1.0)         # actuator 2 - 绿色
COLOR_ACTUATOR_3 = (0.3, 0.2, 0.9, 1.0)         # actuator 3 - 蓝色
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


class SoftRobotArm:
    """软体机械臂类"""
    
    def __init__(self):
        # 初始 actuator 长度 (都设为中间值)
        self.actuator_lengths = [50.0, 50.0, 50.0]  # cm
        
        # 底部三角形顶点 (固定在 z=0)
        self.base_vertices = get_equilateral_triangle_vertices(TRIANGLE_SIDE, center_z=0)
        
        # 顶部三角形顶点 (会根据 actuator 长度计算)
        self.top_vertices = self._calculate_top_vertices()
    
    def _calculate_top_vertices(self):
        """
        根据 actuator 长度计算顶部三角形位置
        这里使用简化模型：假设三角形平行上移
        实际的 PCC 模型会更复杂
        """
        # 简化模型：顶部三角形平行于底部，高度为 actuator 平均长度
        avg_length = sum(self.actuator_lengths) / 3
        
        # 计算顶部三角形的中心位置和旋转
        # 使用 actuator 长度差异来计算倾斜
        l1, l2, l3 = self.actuator_lengths
        
        # 简化：假设顶部三角形中心在 z = avg_length 处
        # 并根据三根 actuator 的长度差异计算倾斜
        
        # 基础高度
        base_height = avg_length
        
        # 计算每个顶点的 z 偏移（基于对应 actuator 与平均值的差异）
        top_vertices = []
        for i, base_v in enumerate(self.base_vertices):
            length_diff = self.actuator_lengths[i] - avg_length
            # 顶点位置 = 底部顶点 + (0, 0, actuator_length)
            # 简化模型：直接垂直上移
            x = base_v[0]
            y = base_v[1]
            z = self.actuator_lengths[i]
            top_vertices.append((x, y, z))
        
        return top_vertices
    
    def set_actuator_length(self, index, length):
        """设置 actuator 长度"""
        length = max(ACTUATOR_MIN, min(ACTUATOR_MAX, length))
        self.actuator_lengths[index] = length
        self.top_vertices = self._calculate_top_vertices()
    
    def get_actuator_endpoints(self):
        """获取所有 actuator 的端点对"""
        endpoints = []
        for i in range(3):
            endpoints.append((self.base_vertices[i], self.top_vertices[i]))
        return endpoints


class Renderer:
    """OpenGL 渲染器"""
    
    def __init__(self):
        pygame.init()
        pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), DOUBLEBUF | OPENGL)
        pygame.display.set_caption("软体机械臂 PCC 模型渲染器")
        
        # 设置 OpenGL
        self._setup_opengl()
        
        # 相机参数
        self.camera_distance = 150.0
        self.camera_angle_x = 30.0  # 俯仰角
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
        
        # 设置背景颜色 - 深色主题
        glClearColor(0.08, 0.08, 0.12, 1.0)
        
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
        
        # 看向原点，上方向为 Y 轴（调整以便 Z 轴向上）
        gluLookAt(cam_x, cam_z, cam_y,  # 相机位置
                  0, 30, 0,              # 看向的点（稍微向上偏移）
                  0, 1, 0)               # 上方向
    
    def _draw_grid(self):
        """绘制地面网格"""
        glColor4f(*COLOR_GRID)
        glBegin(GL_LINES)
        
        grid_size = 100
        grid_step = 10
        
        for i in range(-grid_size, grid_size + 1, grid_step):
            # X 方向线
            glVertex3f(i, 0, -grid_size)
            glVertex3f(i, 0, grid_size)
            # Z 方向线
            glVertex3f(-grid_size, 0, i)
            glVertex3f(grid_size, 0, i)
        
        glEnd()
    
    def _draw_axes(self):
        """绘制坐标轴"""
        axis_length = 30.0
        
        glLineWidth(2.0)
        glBegin(GL_LINES)
        
        # X 轴 - 红色
        glColor4f(1.0, 0.3, 0.3, 1.0)
        glVertex3f(0, 0, 0)
        glVertex3f(axis_length, 0, 0)
        
        # Y 轴 - 绿色 (在 OpenGL 中作为 Z 轴显示)
        glColor4f(0.3, 1.0, 0.3, 1.0)
        glVertex3f(0, 0, 0)
        glVertex3f(0, 0, axis_length)
        
        # Z 轴 - 蓝色 (在 OpenGL 中作为 Y 轴显示，向上)
        glColor4f(0.3, 0.3, 1.0, 1.0)
        glVertex3f(0, 0, 0)
        glVertex3f(0, axis_length, 0)
        
        glEnd()
        glLineWidth(1.0)
    
    def _draw_triangle(self, vertices, color, fill=True):
        """绘制三角形"""
        # 坐标转换：模型的 (x, y, z) -> OpenGL 的 (x, z, y)
        gl_vertices = [(v[0], v[2], v[1]) for v in vertices]
        
        if fill:
            # 填充三角形
            glColor4f(*color)
            glBegin(GL_TRIANGLES)
            for v in gl_vertices:
                glVertex3f(*v)
            glEnd()
        
        # 绘制边框
        glColor4f(1.0, 1.0, 1.0, 1.0)
        glLineWidth(2.0)
        glBegin(GL_LINE_LOOP)
        for v in gl_vertices:
            glVertex3f(*v)
        glEnd()
        glLineWidth(1.0)
        
        # 绘制顶点球
        for i, v in enumerate(gl_vertices):
            self._draw_sphere(v, 1.5, (1.0, 1.0, 1.0, 1.0))
    
    def _draw_sphere(self, position, radius, color):
        """绘制球体（用于标记顶点）"""
        glColor4f(*color)
        glPushMatrix()
        glTranslatef(*position)
        
        quadric = gluNewQuadric()
        gluSphere(quadric, radius, 16, 16)
        gluDeleteQuadric(quadric)
        
        glPopMatrix()
    
    def _draw_actuator(self, start, end, color, index):
        """绘制 actuator（圆柱体）"""
        # 坐标转换
        gl_start = (start[0], start[2], start[1])
        gl_end = (end[0], end[2], end[1])
        
        # 计算 actuator 长度
        length = math.sqrt(sum((e - s) ** 2 for s, e in zip(gl_start, gl_end)))
        
        # 绘制圆柱
        glColor4f(*color)
        
        # 使用线条表示 actuator
        glLineWidth(4.0)
        glBegin(GL_LINES)
        glVertex3f(*gl_start)
        glVertex3f(*gl_end)
        glEnd()
        glLineWidth(1.0)
        
        # 在中点绘制标签球
        mid_point = tuple((s + e) / 2 for s, e in zip(gl_start, gl_end))
        self._draw_sphere(mid_point, 2.0, color)
    
    def _draw_robot(self):
        """绘制机械臂"""
        # 绘制底部三角形
        self._draw_triangle(self.robot.base_vertices, COLOR_TRIANGLE_BASE)
        
        # 绘制顶部三角形
        self._draw_triangle(self.robot.top_vertices, COLOR_TRIANGLE_TOP)
        
        # 绘制 actuators
        colors = [COLOR_ACTUATOR_1, COLOR_ACTUATOR_2, COLOR_ACTUATOR_3]
        endpoints = self.robot.get_actuator_endpoints()
        
        for i, (start, end) in enumerate(endpoints):
            self._draw_actuator(start, end, colors[i], i)
    
    def _draw_info_text(self):
        """绘制信息文本（使用 Pygame 2D 渲染）"""
        # 暂时保存 OpenGL 状态
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, WINDOW_WIDTH, WINDOW_HEIGHT, 0, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()
        
        glDisable(GL_DEPTH_TEST)
        
        # 创建 Pygame surface 用于文本
        font = pygame.font.SysFont('Arial', 18)
        
        # 信息文本
        texts = [
            "软体机械臂 PCC 模型渲染器",
            f"三角形边长: {TRIANGLE_SIDE} cm",
            f"Actuator 1 (红): {self.robot.actuator_lengths[0]:.1f} cm",
            f"Actuator 2 (绿): {self.robot.actuator_lengths[1]:.1f} cm",
            f"Actuator 3 (蓝): {self.robot.actuator_lengths[2]:.1f} cm",
            "",
            "控制:",
            "鼠标拖拽: 旋转视角",
            "滚轮: 缩放",
            "1/2: Actuator 1 +/-",
            "3/4: Actuator 2 +/-",
            "5/6: Actuator 3 +/-",
            "R: 重置",
            "ESC: 退出"
        ]
        
        # 恢复 OpenGL 状态
        glEnable(GL_DEPTH_TEST)
        
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()
    
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
                    self.robot.top_vertices = self.robot._calculate_top_vertices()
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
        
        print("\n" + "=" * 50)
        print("软体机械臂 PCC 模型渲染器")
        print("=" * 50)
        print(f"三角形边长: {TRIANGLE_SIDE} cm")
        print(f"Actuator 范围: {ACTUATOR_MIN} - {ACTUATOR_MAX} cm")
        print("\n控制说明:")
        print("  鼠标拖拽: 旋转视角")
        print("  滚轮: 缩放")
        print("  1/2: Actuator 1 增加/减少")
        print("  3/4: Actuator 2 增加/减少")
        print("  5/6: Actuator 3 增加/减少")
        print("  R: 重置所有 actuator")
        print("  ESC: 退出")
        print("=" * 50 + "\n")
        
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
