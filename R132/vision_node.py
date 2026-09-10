#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""视觉识别节点：
  订阅 /camera/image，识别黄/蓝方块（+红框），数个数、算中心、画框标注。
  发布：
    /task/yellow_count  (std_msgs/Int32)  黄块数量
    /task/blue_count    (std_msgs/Int32)  蓝块数量
    /camera/annotated   (sensor_msgs/Image, bgr8)  画了框的标注图

用法：python3 vision_node.py
前提：camera_bridge 在跑（发布 /camera/image）
"""
import rclpy
from rclpy.node import Node
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
        self.sub = self.create_subscription(Image, '/camera/image', self.cb, 10)
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

        self.min_area = 200   # 小于这个面积的色块当噪声，可调

    def cb(self, msg):
        h, w = msg.height, msg.width
        img = np.frombuffer(msg.data, dtype=np.uint8).reshape(h, w, 3)  # BGR
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        yellow = cv2.inRange(hsv, self.yellow_lo, self.yellow_hi)
        blue = cv2.inRange(hsv, self.blue_lo, self.blue_hi)
        red = cv2.bitwise_or(cv2.inRange(hsv, self.red_lo1, self.red_hi1),
                             cv2.inRange(hsv, self.red_lo2, self.red_hi2))

        yellow_cnt, yellow_boxes = self.find_blocks(yellow)
        blue_cnt, blue_boxes = self.find_blocks(blue)
        red_cnt, red_boxes = self.find_blocks(red)

        self.pub_yellow.publish(Int32(data=yellow_cnt))
        self.pub_blue.publish(Int32(data=blue_cnt))

        # ---- 画标注图 ----
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

    def find_blocks(self, mask):
        """在二值 mask 里找独立色块，返回 (个数, [(x,y,w,h)...])"""
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)   # 去小噪声
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for c in contours:
            if cv2.contourArea(c) < self.min_area:
                continue
            x, y, bw, bh = cv2.boundingRect(c)
            boxes.append((x, y, bw, bh))
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
