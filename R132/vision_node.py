#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""视觉识别节点（性能 + 分块识别版）：
  订阅 /camera/image，识别黄/蓝方块（+红框），数个数、画框标注。

  优化点：
    1. 检测在降采样图（1/scale）上做，坐标乘回原图；大幅提速，避免回调节点积压。
    2. 订阅队列深度=1：处理不过来就丢旧帧、只处理最新帧，不会越积越延迟。
    3. 用「距离变换」把并列/贴在一起的同色物块拆成一个个独立物块，
       而不是糊成一个连通块。

  发布：
    /task/yellow_count  (std_msgs/Int32)  黄块数量
    /task/blue_count    (std_msgs/Int32)  蓝块数量
    /camera/annotated   (sensor_msgs/Image, bgr8)  画了框的标注图

  用法：python3 vision_node.py
  前提：camera_bridge 在跑（发布 /camera/image）
"""
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Int32
import numpy as np
import cv2


def img_to_msg(img_bgr, stamp, frame_id):
    """numpy BGR 图 -> sensor_msgs/Image (bgr8)"""
    h, w = img_bgr.shape[:2]
    msg = Image()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.height = h
    msg.width = w
    msg.encoding = 'bgr8'
    msg.is_bigendian = 0
    msg.step = w * 3
    msg.data = img_bgr.tobytes()
    return msg


class VisionNode(Node):
    def __init__(self):
        super().__init__('vision_node')

        # 只处理最新帧：深度 1 + keep_last，慢时丢旧帧不积延迟
        qos_sub = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
        )
        self.sub = self.create_subscription(Image, '/camera/image', self.cb, qos_sub)
        self.pub_annotated = self.create_publisher(Image, '/camera/annotated', 10)
        self.pub_yellow = self.create_publisher(Int32, '/task/yellow_count', 10)
        self.pub_blue = self.create_publisher(Int32, '/task/blue_count', 10)

        # ---- HSV 阈值（按实测微调：真蓝 H≈97 S≈194 V≈167；阴影 S≤113 要滤掉）----
        self.yellow_lo = np.array([18, 100, 120])
        self.yellow_hi = np.array([38, 255, 255])
        self.blue_lo = np.array([90, 130, 100])
        self.blue_hi = np.array([110, 255, 255])
        self.red_lo1 = np.array([0, 70, 40])
        self.red_hi1 = np.array([15, 255, 255])
        self.red_lo2 = np.array([165, 70, 40])
        self.red_hi2 = np.array([180, 255, 255])

        # ---- 可调参数 ----
        self.scale = 2              # 检测时缩小倍数（原图 800x600 -> 400x300）
        self.min_area = 60          # 降采样后「核心」小于该面积当噪声（可调）
        self.dist_ratio = 0.4       # 距离变换取峰值比例，越小核心越容易拆开（可调）

    def cb(self, msg):
        h, w = msg.height, msg.width
        img = np.frombuffer(msg.data, dtype=np.uint8).reshape(h, w, 3)  # BGR

        # ---- 检测在降采样图上做（快 4 倍），坐标乘回原图 ----
        s = self.scale
        small = cv2.resize(img, (w // s, h // s), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)

        yellow = cv2.inRange(hsv, self.yellow_lo, self.yellow_hi)
        blue = cv2.inRange(hsv, self.blue_lo, self.blue_hi)
        red = cv2.bitwise_or(cv2.inRange(hsv, self.red_lo1, self.red_hi1),
                             cv2.inRange(hsv, self.red_lo2, self.red_hi2))

        yellow_cnt, yellow_boxes = self.find_blocks(yellow, s)
        blue_cnt, blue_boxes = self.find_blocks(blue, s)
        red_cnt, red_boxes = self.find_blocks(red, s)

        self.pub_yellow.publish(Int32(data=yellow_cnt))
        self.pub_blue.publish(Int32(data=blue_cnt))

        # ---- 在原分辨率图上画标注 ----
        annotated = img.copy()
        for (x, y, bw, bh) in yellow_boxes:
            cv2.rectangle(annotated, (x, y), (x + bw, y + bh), (0, 255, 255), 2)
            cv2.putText(annotated, 'Y', (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        for (x, y, bw, bh) in blue_boxes:
            cv2.rectangle(annotated, (x, y), (x + bw, y + bh), (255, 0, 0), 2)
            cv2.putText(annotated, 'B', (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        for (x, y, bw, bh) in red_boxes:
            cv2.rectangle(annotated, (x, y), (x + bw, y + bh), (0, 0, 255), 2)
            cv2.putText(annotated, 'R', (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.putText(annotated, 'Y:%d B:%d R:%d' % (yellow_cnt, blue_cnt, red_cnt),
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        self.pub_annotated.publish(img_to_msg(annotated, msg.header.stamp, 'camera'))

        self.get_logger().info('yellow=%d blue=%d red=%d' % (yellow_cnt, blue_cnt, red_cnt),
                               throttle_duration_sec=1.0)

    def find_blocks(self, mask, scale):
        """把二值 mask 里「贴在一起的同色物块」拆成一个个独立物块。

        原理：距离变换后，每个物块中心距离值最大；多个物块贴在一起会形成多个「峰」，
        对距离图按峰值比例取阈值，每个峰就变成一个独立连通域（= 一个物块核心）。
        返回 (个数, [(x,y,w,h)...])，坐标已乘回原图分辨率。
        """
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)   # 去小噪声

        # 距离变换：每个前景像素到最近背景的距离
        dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        peak = float(dist.max())
        if peak <= 0:
            return 0, []

        # 阈值掐出每个物块的核心；贴在一起的物块在接触处距离≈0，会被切开
        _, cores = cv2.threshold(dist, peak * self.dist_ratio, 255, cv2.THRESH_BINARY)
        cores = cores.astype(np.uint8)

        num, _, stats, _ = cv2.connectedComponentsWithStats(cores, 8)
        boxes = []
        pad = int(peak * 0.5)   # 核心比真实物块小一圈，往外扩一点让框贴近物块边缘
        for i in range(1, num):
            x, y, bw, bh, area = stats[i]
            if area < self.min_area:
                continue
            x0 = max(0, x - pad)
            y0 = max(0, y - pad)
            x1 = x + bw + pad
            y1 = y + bh + pad
            boxes.append((x0 * scale, y0 * scale,
                          (x1 - x0) * scale, (y1 - y0) * scale))
        return len(boxes), boxes


def main():
    rclpy.init()
    node = VisionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
