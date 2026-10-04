# from time import sleep
"""
订阅检测结果：
dnn 检测到的方块信息
彩色图像
深度图像

计算：
相机坐标系下的方块位置
转换
工具坐标系下的方块位置
方块颜色

服务返回：
工具坐标系下的方块位置
方块颜色
"""

import rclpy
from rclpy.node import Node
from ai_msgs.msg import PerceptionTargets, Target, Roi
from sensor_msgs.msg import Image, CompressedImage

import numpy as np
from cv_bridge import CvBridge

from tools_demo.modules.tools import (
    get_color,
    pixel_to_camera,
    create_handeye_matrix,
    camera_to_tool,
    Params,
)

from std_srvs.srv import Trigger
import json
import math

import cv2


class DetectionTFBroadcaster(Node):
    # —— 深度拆分参数（解决「多个并列被合并成一个」和「侧面被误当成两个」）——
    BLOCK_SIZE_MM = 30.0    # 物块边长 3cm（立方体）
    TOP_FACE_TOL_M = 0.002  # 顶面深度容差 4mm：侧面是「从顶面深度往下斜的斜坡」，斜拍时
                            # 侧面只比顶面深 10~15mm，12mm 会把大半侧面误并入顶面、把顶面
                            # bbox 撑大，导致两个并列块被切成 4 个。收紧到 4mm 只留顶面。
    MIN_TOP_PIXELS = 10     # 顶面有效像素过少就退回整框中心（旧逻辑）

    def __init__(self):
        super().__init__("block_pose_get")
        # 订阅检测结果
        self.subscription = self.create_subscription(
            PerceptionTargets, "/hobot_dnn_detection", self.detection_callback, 10
        )

        # 深度图像
        self.depth_subscription = self.create_subscription(
            Image,
            "/camera/aligned_depth_to_color/image_raw",
            self.depth_image_callback,
            10,
        )

        # 彩色图像
        self.color_subscription = self.create_subscription(
            CompressedImage,
            "/camera/color/image_raw/compressed",
            self.color_image_callback,
            10,
        )

        self.color_image_ = None
        # self.pub = self.create_publisher(String, "/block_pose", 10)

        # 图像参数
        self.depth_image = None
        self.bridge = CvBridge()

        self.T_tool_cam = create_handeye_matrix()

        # 检测状态控制
        self.detected_blocks = []  # 存储检测到的方块信息
        self.detected_blocks_str = ""  # 存储检测到的方块信息的字符串表示
        self.target_count = 0  # 检测的方块数量
        self.detect_seq = 0  # 检测代次：每次 DNN 出新检测 +1，供调用方判断「新画面」

        self.declare_parameter("target_count", self.target_count)
        self.declare_parameter("detected_blocks", "")
        # 识别服务
        self.restart_detection_service = self.create_service(
            Trigger, "/restart_detection", self.restart_detection_callback
        )

    def detection_callback(self, msg: PerceptionTargets):
        # if self.detection_complete:
        #     return  # 如果检测已完成，跳过处理
        self.detected_blocks = []  # 清空之前的检测结果
        self.target_count = 0  # 重置目标计数
        self.detect_seq += 1  # 每收到一次 DNN 检测就 +1

        # 等待深度图像
        if self.depth_image is None:
            self.get_logger().info("Depth image not available yet")
            return

        new_blocks = []  # 本次回调检测到的新方块

        for target in msg.targets:
            target: Target
            for roi in target.rois:
                roi: Roi
                if roi.type != "block" or roi.confidence < 0.55:
                    continue

                x_offset = roi.rect.x_offset
                y_offset = roi.rect.y_offset
                width = roi.rect.width
                height = roi.rect.height

                img_w, img_h = Params.color_image_size

                # 深度拆分：一个 DNN ROI 可能含多个并列块（合并），
                # 或含「顶面+侧面」（物块位置偏、相机斜拍）。这里按「顶面深度 + 物块
                # 尺寸」拆成 1..N 个真实物块中心（纯深度、不分颜色），逐个采样颜色
                # 再转 3D 入库。
                for (u, v, depth_value, color) in self.split_block_centers(
                    x_offset, y_offset, width, height
                ):
                    # 颜色已在 split_block_centers 里按该子块「顶面掩膜」多数投票得出，
                    # 不再在中心点取固定小窗（固定小窗在异色并列时会采到相邻块、两个都同色）

                    # 离画面中心的像素距离（「居中」过滤：越近物块越正、抓得越准）
                    dist_to_center = math.hypot(u - img_w / 2.0, v - img_h / 2.0)

                    # 转换为3D坐标
                    point_3d = pixel_to_camera(u, v, depth_value)

                    # 将相机坐标系下的3D点转换为工具坐标系下的3D点
                    point_3d_tool = camera_to_tool(
                        point_3d[0], point_3d[1], point_3d[2], self.T_tool_cam
                    )

                    # 检查是否已检测到类似位置的方块（避免重复）
                    if not self.is_new_block(point_3d_tool):
                        continue

                    # 创建方块信息
                    block_info = {
                        "id": len(self.detected_blocks) + len(new_blocks),
                        "tool_pos": [point_3d_tool[0], point_3d_tool[1], point_3d_tool[2]],
                        "color": color,
                        "dist_to_center": dist_to_center,
                    }

                    new_blocks.append(block_info)
                    self.target_count += 1  # 增加目标计数

        # 添加新检测到的方块
        self.detected_blocks.extend(new_blocks)

    def depth_image_callback(self, msg):
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="mono16")
        except Exception as e:
            self.get_logger().error(f"Error converting depth image: {e}")

    def color_image_callback(self, msg):
        try:
            self.color_image_ = self.bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f"Error converting color image: {e}")

    def get_color_(self, roi) -> str:
        """根据ROI获取颜色"""
        if self.color_image_ is None:
            return "no_image"
        color = get_color(self.color_image_, roi)
        # === 调试：打印 ROI 位置、蓝/黄像素占比、中心像素的真实颜色 ===
        #     用来判断是「采错位置」还是「HSV 判错」。定位完可以删掉这段。
        try:
            from tools_demo.modules.tools import Params

            x, y, w, h = [int(v) for v in roi]
            sub = self.color_image_[y : y + h, x : x + w]
            if sub.size:
                hsv = cv2.cvtColor(sub, cv2.COLOR_BGR2HSV)
                mb = cv2.countNonZero(cv2.inRange(hsv, Params.lower_blue, Params.upper_blue))
                my = cv2.countNonZero(
                    cv2.inRange(hsv, Params.lower_yellow, Params.upper_yellow)
                )
                tot = w * h
                cy, cx = h // 2, w // 2
                self.get_logger().info(
                    f"[颜色调试] rect=({x},{y},{w},{h}) 图像={self.color_image_.shape} "
                    f"判定={color} blue占比={mb / tot:.2f} yellow占比={my / tot:.2f} "
                    f"中心BGR={sub[cy, cx].tolist()} 中心HSV={hsv[cy, cx].tolist()}"
                )
        except Exception as e:
            self.get_logger().error(f"颜色调试失败: {e}")
        return color

    def save_to_parameter_server(self):
        """将检测结果保存到参数服务器"""
        # 保存目标数量
        target_count_param = rclpy.Parameter(
            "target_count", rclpy.Parameter.Type.INTEGER, self.target_count
        )

        # 保存方块信息（序列化为JSON字符串）
        blocks_param = []
        for block in self.detected_blocks:
            block_data = {
                "id": block["id"],
                "tool_pos": block["tool_pos"],
                "color": block["color"],
                "dist_to_center": block.get("dist_to_center", 1e9),
            }
            blocks_param.append(block_data)

        self.detected_blocks_str = json.dumps(blocks_param)
        detected_blocks_param = rclpy.Parameter(
            "detected_blocks",
            rclpy.Parameter.Type.STRING,
            self.detected_blocks_str,
        )

        # 将所有参数放入一个列表中
        parameters_to_set = [
            target_count_param,
            detected_blocks_param,
        ]

        # 设置参数
        results = self.set_parameters(parameters_to_set)

        # 检查每个参数设置的结果
        for result in results:
            if not result.successful:
                self.get_logger().error(f"Failed to set parameter: {result.reason}")

        self.get_logger().info(
            f"Saved {len(self.detected_blocks)} blocks to parameter server."
        )
        return self.detected_blocks_str

    def is_new_block(self, position, min_distance=0.02):
        """检查是否是新方块（避免重复检测）。

        min_distance 从 0.03 降到 0.02：两个并列同色块的间距 ≈ 3cm，用 0.03 判重
        时测量噪声会误把相邻块当成重复、丢掉一个；0.02 仍能去掉「同一块被重复检出」
        （间隔 <2cm），又不会吃掉间距 3cm 的相邻块。
        """
        for block in self.detected_blocks:
            existing_pos = block["tool_pos"]
            distance = np.sqrt(
                (position[0] - existing_pos[0]) ** 2
                + (position[1] - existing_pos[1]) ** 2
                + (position[2] - existing_pos[2]) ** 2
            )
            if distance < min_distance:
                return False
        return True

    def get_depth_value(self, x, y):
        """从深度图像获取深度值"""
        if self.depth_image is None:
            return None

        height, width = self.depth_image.shape
        if 0 <= x < width and 0 <= y < height:
            return self.depth_image[int(y), int(x)] / 1000.0  # 毫米转米
        return None

    def sample_block_color(self, u, v, r=6):
        """在子块中心 (u, v) 附近取一小块，判断该块颜色（异色并列时各判各的）。

        返回 "blue" / "yellow" / "other"。r 取小块半径（像素），中心一定落在
        该子块顶面内部，不会采到相邻块。
        """
        if self.color_image_ is None:
            return "other"
        img_h, img_w = self.color_image_.shape[:2]
        x0, y0 = max(0, int(u) - r), max(0, int(v) - r)
        x1, y1 = min(img_w, int(u) + r), min(img_h, int(v) + r)
        if x1 - x0 < 3 or y1 - y0 < 3:
            return "other"
        return get_color(self.color_image_, [x0, y0, x1 - x0, y1 - y0])

    def color_from_mask(self, mask, ox, oy):
        """对顶面掩膜 mask 内的彩色像素做多数投票，返回 blue/yellow/other。

        异色并列时，固定小窗(如 13×13)若采到相邻块就会两个都判成同色；这里改成
        用「该子块自己的顶面掩膜外接框」去采色，外接框只落在自己块上，天然不会
        混进相邻块的颜色。mask 是 (h,w) bool，ox,oy 是它在彩色图里的左上角像素坐标。
        """
        if self.color_image_ is None:
            return "other"
        ys, xs = np.nonzero(mask)
        if len(xs) < self.MIN_TOP_PIXELS:
            return "other"
        x0, x1 = int(xs.min()) + ox, int(xs.max()) + ox + 1
        y0, y1 = int(ys.min()) + oy, int(ys.max()) + oy + 1
        img_h, img_w = self.color_image_.shape[:2]
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(img_w, x1), min(img_h, y1)
        if x1 - x0 < 3 or y1 - y0 < 3:
            return "other"
        color = get_color(self.color_image_, [x0, y0, x1 - x0, y1 - y0])
        # 诊断：打印采色框和蓝/黄占比，判断是「采错位置」还是「HSV 判错」
        try:
            sub = self.color_image_[y0:y1, x0:x1]
            hsv = cv2.cvtColor(sub, cv2.COLOR_BGR2HSV)
            mb = cv2.countNonZero(cv2.inRange(hsv, Params.lower_blue, Params.upper_blue))
            my = cv2.countNonZero(cv2.inRange(hsv, Params.lower_yellow, Params.upper_yellow))
            tot = sub.shape[0] * sub.shape[1]
            self.get_logger().info(
                f"[颜色诊断] bbox=({x0},{y0},{x1 - x0},{y1 - y0}) 判定={color} "
                f"blue={mb / tot:.2f} yellow={my / tot:.2f}"
            )
        except Exception as e:
            self.get_logger().error(f"颜色诊断失败: {e}")
        return color

    def split_block_centers(self, x, y, w, h):
        """把一个 DNN ROI 按「顶面深度 + 物块尺寸」拆成 1..N 个物块中心。

        纯深度拆分、不按颜色预过滤，所以同色/异色并列都能拆开；颜色由调用方对
        每个子块单独采样。解决两类问题：
          1. 多个物块并列 → DNN 把多个合并成一个 ROI，这里按 3cm 边长把顶面
             切成 N 份，返回 N 个中心；
          2. 物块位置偏、相机拍到侧面 → 顶面+侧面同色但深度不同（侧面比顶面
             深约 3cm），这里只保留「顶面」（最近的深度平面），避免把侧面误
             当成第二个物块。

        参数:
            x, y, w, h: ROI 在彩色/深度图中的像素框
        返回:
            [(u, v, depth, color), ...]  每个子块一个中心（颜色图像素坐标 + 深度米 + 颜色）
        """
        if self.depth_image is None:
            # 深度没就绪，退回整框中心（和旧逻辑一致）
            d = self.get_depth_value(x + w / 2, y + h / 2) or 0.0
            return [(x + w / 2, y + h / 2, d, self.sample_block_color(x + w / 2, y + h / 2))]

        img_h, img_w = self.depth_image.shape[:2]
        x0, y0 = max(0, int(x)), max(0, int(y))
        x1, y1 = min(img_w, int(x + w)), min(img_h, int(y + h))
        if x1 - x0 < 2 or y1 - y0 < 2:
            return [(x + w / 2, y + h / 2, 0.0, self.sample_block_color(x + w / 2, y + h / 2))]

        # 1) 深度子图（mm → 米）。顶面是最靠前的平面，比桌面/侧面都近，
        #    单靠深度就能和背景、侧面分开，不需要颜色掩膜。
        roi_depth = self.depth_image[y0:y1, x0:x1].astype(np.float32) / 1000.0
        valid = roi_depth > 0.05
        if int(valid.sum()) < self.MIN_TOP_PIXELS:
            return [(x + w / 2, y + h / 2, 0.0, self.sample_block_color(x + w / 2, y + h / 2))]

        # 2) 顶面深度 = 最近 5% 分位（相机朝下，顶面最靠前=深度最小）
        depths = roi_depth[valid]
        d_top = float(np.percentile(depths, 5))
        top_mask = valid & (roi_depth <= d_top + self.TOP_FACE_TOL_M)

        ys, xs = np.nonzero(top_mask)
        if len(xs) < self.MIN_TOP_PIXELS:
            return [(x + w / 2, y + h / 2, 0.0, self.sample_block_color(x + w / 2, y + h / 2))]

        # 3) 顶面在 d_top 深度处的物理尺寸：1 像素 ≈ d_top / fx 米
        pixel_mm = d_top * 1000.0 / Params.camera_matrix[0, 0]
        u_min, u_max = int(xs.min()), int(xs.max())
        v_min, v_max = int(ys.min()), int(ys.max())
        n_w = max(1, int(round((u_max - u_min + 1) * pixel_mm / self.BLOCK_SIZE_MM)))
        n_h = max(1, int(round((v_max - v_min + 1) * pixel_mm / self.BLOCK_SIZE_MM)))
        # 诊断：打印顶面 bbox 的物理尺寸和切分结果，用来核对「侧面撑大bbox→切成4个」这类问题
        self.get_logger().info(
            f"[拆分诊断] ROI=({x:.0f},{y:.0f},{w:.0f},{h:.0f}) d_top={d_top*1000:.0f}mm "
            f"顶面bbox={(u_max-u_min+1)*pixel_mm:.0f}x{(v_max-v_min+1)*pixel_mm:.0f}mm "
            f"→ 切成 {n_w}x{n_h}"
        )

        # 4) 单个物块：顶面质心（比整框中心准，天然避开了侧面）
        if n_w * n_h <= 1:
            d = float(np.median(roi_depth[top_mask]))
            color = self.color_from_mask(top_mask, x0, y0)
            return [(float(x0 + xs.mean()), float(y0 + ys.mean()), d, color)]

        # 5) 多个物块：按 n_w×n_h 网格切分顶面，每格取该格顶面像素的质心
        u_step = (u_max - u_min + 1) / n_w
        v_step = (v_max - v_min + 1) / n_h
        centers = []
        for i in range(n_w):
            u_lo = int(u_min + u_step * i)
            u_hi = int(u_min + u_step * (i + 1))
            for j in range(n_h):
                v_lo = int(v_min + v_step * j)
                v_hi = int(v_min + v_step * (j + 1))
                seg = top_mask[v_lo:v_hi + 1, u_lo:u_hi + 1]
                if int(seg.sum()) < self.MIN_TOP_PIXELS:
                    continue
                sy, sx = np.nonzero(seg)
                seg_depth = roi_depth[v_lo:v_hi + 1, u_lo:u_hi + 1][seg]
                color = self.color_from_mask(seg, x0 + u_lo, y0 + v_lo)
                centers.append((float(x0 + u_lo + sx.mean()),
                                float(y0 + v_lo + sy.mean()),
                                float(np.median(seg_depth)),
                                color))

        if not centers:
            return [(x + w / 2, y + h / 2, 0.0, self.sample_block_color(x + w / 2, y + h / 2))]
        return centers

    def restart_detection_callback(self, request, response: Trigger.Response):
        """处理重启检测的请求"""
        """PS:
            ros2 service call /restart_detection std_srvs/srv/Trigger "{}"
        """
        self.get_logger().info("Restarting detection.")
        blocks_str = self.save_to_parameter_server()
        response.success = True
        # 返回 seq（检测代次）+ blocks，调用方据此判断是不是机械臂停稳后的新检测
        response.message = json.dumps({
            "seq": self.detect_seq,
            "blocks": json.loads(blocks_str),
        })
        return response


def main(args=None):
    rclpy.init(args=args)
    node = DetectionTFBroadcaster()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()


if __name__ == "__main__":
    main()
