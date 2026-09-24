"""
接收抓取请求
请求当前画面方块信息：
 - 工具坐标系位置
 - 颜色
执行：
 - 遍历方块信息->
    - 工具坐标系转换基坐标系
    - 颜色分组
 - 按「数量多者先抓」排序，执行抓取->
    - 移动到指定位置
    - 吸取
    - 移动到放置位置（只有一个放置区）
    - 释放
 - 真实抓取时把 任务进度/当前任务/状态 实时刷到面板：
    - 状态 = 运行（抓取中）；当前任务 = 前往资源点 / 抓取资源点 / 前往放置区
"""

import json

import rclpy
from rclpy.node import Node
from custom_msgs.srv import StrMsg
from std_srvs.srv import Trigger
from std_msgs.msg import String

from time import sleep
from tools_demo.modules.jaka import Jaka
from tools_demo.modules.tools import tool_to_base

IO_CABINET = 0  # 控制柜面板IO
IO_TOOL = 1  # 工具IO
IO_EXTEND = 2  # 扩展IO

DO1 = 0  # 工具供电开关
DO2 = 1  # 工具使能开关

# 运动模式
ABS = 0  # 绝对位置
INCR = 1  # 增量位置


class Connecter(Node):
    def __init__(self):
        super().__init__("connecter")
        self.robot = Jaka()  # 返回一个机器人对象
        if not self.robot_is_ready():
            self.get_logger().error("robot is not ready")
            return

        # 放置区（只有一个位置）
        self.back_pose = [-31.20, 374.393, 143.20, 180, 0.0, 0.0]

        # 末端位置，拾取移动距离 单位mm
        self.end_pose_z = 120.0
        self.end_move_z = -8

        self.command = {}
        self.srv_ser = self.create_service(
            StrMsg, "send_command", self.connect_callback_
        )
        # ros2 service call /restart_detection std_srvs/srv/Trigger "{}"
        self.srv_client = self.create_client(Trigger, "restart_detection")
        self.target_count = 0
        self.blocks_data = []
        self.targets = []  # 多目标 [{"color":"yellow","num":2}, ...]，按数量降序

        # 面板动态字段发布（真实抓取时实时刷新 任务进度/当前任务/状态）
        self.pub_progress = self.create_publisher(String, "/task/progress", 10)
        self.pub_current = self.create_publisher(String, "/task/current_task", 10)
        self.pub_status = self.create_publisher(String, "/task/status", 10)

    def robot_is_ready(self):
        ret = False
        if self.robot.login_state:
            self.get_logger().info("登录成功")
            self.robot.home_pose = [-252.7, -20.5, 293.7, 180, 0.0, 0.0]
            ret = self.robot.go_home()
            if ret[0] == 0:
                self.get_logger().info("move to point success")
                ret = True
            else:
                self.get_logger().error(f"some things happend,the errcode is:{ret}")
                ret = False
        else:
            self.get_logger().error("登录失败")
            ret = False
        self.get_logger().info("Connecter service started")
        return ret

    def restart_detection(self, done_cb=None):
        if not self.srv_client.wait_for_service(timeout_sec=10.0):
            self.get_logger().info("service not available, waiting again...")
            return False

        self.get_logger().info("service available")

        request = Trigger.Request()
        future = self.srv_client.call_async(request)

        if done_cb:
            future.add_done_callback(done_cb)
            return future
        else:
            # 非阻塞轮询
            while not future.done():
                rclpy.spin_once(self, timeout_sec=0.1)
            self.get_logger().info("Service call completed")
            return future.result()

    def connect_callback_(self, request: StrMsg.Request, response: StrMsg.Response):
        """接收多目标抓取请求：
        request.data = {"targets":[{"color":"yellow","num":2},{"color":"blue","num":3}],"total":5}
        哪个颜色数量多就先抓哪个。
        """
        self.get_logger().info(f"request received: {request.data}")
        try:
            cmd = json.loads(request.data)
            targets_data = cmd.get("targets")
            if not isinstance(targets_data, list) or not targets_data:
                raise ValueError("缺少 targets 列表")

            targets = []
            for t in targets_data:
                color = t.get("color")
                num = t.get("num")
                if not color or num is None:
                    raise ValueError(f"target 缺少 color/num: {t}")
                targets.append({"color": str(color).lower(), "num": int(num)})

            # 数量多者先抓
            self.targets = sorted(targets, key=lambda t: -t["num"])
            self.get_logger().info(
                "抓取顺序: " + ", ".join(f"{t['color']}:{t['num']}" for t in self.targets)
            )

            # 启动检测，拿到当前画面方块后开始抓
            self.restart_detection(done_cb=self.process_restart_result)
            response.success = True
            response.message = "Command received, restart_detection in progress"
        except Exception as e:
            self.get_logger().error(f"Error processing request: {str(e)}")
            response.success = False
            response.message = str(e)
        return response

    def process_restart_result(self, future):
        result: Trigger.Response = future.result()
        self.get_logger().info("Restart detection result received")

        try:
            self.blocks_data = json.loads(result.message)
        except Exception:
            self.get_logger().error(f"解析方块结果失败: {result.message}")
            self.publish_panel(status="待命")
            return

        self.get_logger().info(
            f"收到 {len(self.blocks_data)} 个方块: "
            f"{[(b.get('color'), b.get('tool_pos')) for b in self.blocks_data]}"
        )

        current_pose = self.robot.get_tools_pos()
        if current_pose == -1:
            self.get_logger().error("Failed to get current tool position")
            self.publish_panel(status="待命")
            return

        # 转换坐标系并按颜色分组
        color_blocks = {}
        for item in self.blocks_data:
            pos = tool_to_base(item["tool_pos"], current_pose)
            color = item["color"]
            self.get_logger().info(f"item_pos: {item['tool_pos']}, pos: {pos}, color: {color}")
            color_blocks.setdefault(color, []).append(pos)

        # 按「数量多者先抓」的目标顺序，构建抓取序列
        pick_list = []  # [(base_pos, color), ...]
        for t in self.targets:
            available = color_blocks.get(t["color"], [])
            pick_list.extend((pos, t["color"]) for pos in available[: t["num"]])

        if not pick_list:
            self.get_logger().warn("没有可抓的方块")
            self.publish_panel(status="待命")
            return

        self.get_logger().info(f"待抓取序列共 {len(pick_list)} 个: {[c for _, c in pick_list]}")

        # 执行抓取操作（真实抓取时把进度/状态实时刷到面板）
        self.robot.pick_init()
        for i, (pos, color) in enumerate(pick_list, 1):
            self.get_logger().info(f"抓取第{i}个：{color}")
            try:
                self.pick(pos[0], pos[1], pos[2], i)
            except Exception as e:
                self.get_logger().error(f"抓取第{i}个({color})异常：{e}")
        self.robot.pick_end()
        self.robot.go_home()
        # 全部完成，回待命
        self.publish_panel(status="待命", current="")

    def move_to_point(self, x, y, z):
        """
        先移动到指定位置,单位 mm
        """
        self.get_logger().info(f"move to point: {x}, {y}, {z}")
        ret = self.robot.go_point([x, y, z])
        if ret[0] == 0:
            self.get_logger().info("move to point success")
            return True
        else:
            self.get_logger().error(f"some things happend,the errcode is:{ret}")
            return False

    def publish_panel(self, progress=None, current=None, status=None):
        """实时刷新面板动态字段（只发传入的非空字段）。"""
        if progress is not None:
            self.pub_progress.publish(String(data=progress))
        if current is not None:
            self.pub_current.publish(String(data=current))
        if status is not None:
            self.pub_status.publish(String(data=status))

    def pick(self, x, y, z, index):
        """
        抓取一个方块，面板按三态刷新：前往资源点 -> 抓取资源点 -> 前往放置区，
        状态全程为「运行」。
        """
        # 1. 前往资源点
        self.publish_panel(progress=f"第{index}个", current="前往资源点", status="运行")
        offset_x = 0
        offset_y = -5 if y > 0 else -6
        if not self.move_to_point(
            x * 1000 + offset_x, y * 1000 + offset_y, self.end_pose_z
        ):
            return False
        sleep(1)

        # 2. 抓取资源点（吸取）
        self.publish_panel(current="抓取资源点", status="运行")
        self.robot.do_pick_on(self.end_move_z)
        sleep(1)

        # 3. 前往放置区（只有一个放置区）
        self.publish_panel(current="前往放置区", status="运行")
        if not self.robot.go_pose(self.back_pose):
            return False
        self.robot.pick_off()
        return True


def main(args=None):
    rclpy.init(args=args)
    connecter = Connecter()
    try:
        rclpy.spin(connecter)
    except KeyboardInterrupt:
        pass
    connecter.get_logger().info("Shutdown")
    connecter.destroy_node()


if __name__ == "__main__":
    main()
