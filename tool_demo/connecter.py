"""
接收抓取请求，按「颜色外层 × 扇区内层」扫描抓取物块。

解决「相机视野有限、边缘物块被斜拍导致抓偏」的问题：
  - 单个位姿下相机只能正对中间，四周物块被斜拍（侧面进画面），抓取偏。
  - 扫描位姿由人工示教得到（见 SCAN_POSES）：把机械臂拖到「相机正对该区域、
    物块在画面中心」的位置，读出 6 自由度位姿填进去。逐扇区扫描，只抓离画面
    中心近的物块（dist_to_center < CENTER_RADIUS），保证每个都正对、抓得准。
  - 顺序：先抓完数量多的颜色（外层），再抓少的；每个颜色内部逐扇区找（内层）。
  - 兜底 B：某颜色所有扇区扫完仍不够，再逐扇区抓离中心最近的边缘块。
  - 每抓一个物块 → 回到扫描位姿重扫一次再抓下一个（不一次抓空一个扇区，
    避免旧检测算出的过期坐标抓偏）。
  - 抓取在后台线程跑：一轮结束后回待命、解除占用，点「开始任务」可再派单，
    不用重启服务。

面板动态字段（真实抓取时实时刷新）：
  状态 = 执行；当前任务 = 前往资源点 / 抓取资源 / 前往放置区；放完回 待命。
"""

import json
import math
import threading

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSHistoryPolicy
from custom_msgs.srv import StrMsg
from std_srvs.srv import Trigger
from std_msgs.msg import String

from time import sleep, time
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
    # 手眼标定残差校正：绕基座Z轴旋转 θ 度 + 平移 (mm)
    # 左右分开（以 y=0 为中线）：y<0 左边用 L 组，y>=0 右边用 R 组。
    # 原因：单一组刚体(旋转+平移)只能校正「全局刚体」残差，而手眼残差随位置变化、
    # 左右符号相反，单一一组只能折中。左右各 3 个自由度能各自吸收自己的残差。
    # 初值都取原来的全局值，标定后分别填入 calib_correct.py 跑左/右两组的输出。
    CORRECT_THETA_L = 2.406   # 度（左边，y<0）
    CORRECT_TX_L = 3.34 #4.34      # mm
    CORRECT_TY_L = -5.01      # mm

    CORRECT_THETA_R = 2.406   # 度（右边，y>=0）
    CORRECT_TX_R = -10.34      # mm
    CORRECT_TY_R = -2.01      # mm

    # 扫描位姿（示教得到）：手动把机械臂拖到「相机正对该区域、物块在画面中心」的位置，
    # 用 read_pose.py 读出 [x,y,z,rx,ry,rz](mm,deg) 填到下面。顺序即扫描顺序，中间先放。
    # 第一个建议接近 home（逆解种子最稳），后面的按离前一个由近到远排。
    SCAN_POSES = [

        [-10.3, -230.9, 254.7, -177.6, 2.7, 84.7],
        [-138.1, -185.4, 254.6, -177.6, 2.7, 50.6],
        [-212.4, -81.3, 254.7, -177.6, 2.7, 20.5],
        [-218.1, 36.6, 254.7, -177.6, 2.7, -22.1],
        [-218.1, 130.6, 254.7, -177.6, 2.7, -22.1],
        [-146.8, 178.6, 254.7, -177.6, 2.7, -53.3],
        [-23.8, 204.0, 219.1, 179.4, -0.9, -88.2],

        
        [-218.1, 150.6, 254.7, -177.6, 2.7, -22.1],
        [-212.4, -91.3, 254.7, -177.6, 2.7, 20.5]
        
    ]

    # 「居中」过滤半径（像素）：dist_to_center < 此值才抓
    CENTER_RADIUS = 200  #160.0

    # 过滤区（放置区）：物块落在这个矩形内就不抓，避免重复抓取已放置的块。
    # 单位 mm（基座坐标系）：x ∈ [FILTER_X_MIN, FILTER_X_MAX] 且 y ∈ [FILTER_Y_MIN, FILTER_Y_MAX]
    FILTER_X_MIN = -150.0
    FILTER_X_MAX = 0.0
    FILTER_Y_MIN = 320.0
    FILTER_Y_MAX = 520.0
    # 精确过滤：距任一放置点(back_poses)小于该半径(mm)的块视为「已放置」，跳过。
    # 比矩形过滤更稳：放置点坐标就是机械臂放块的坐标，不受「y_min 只差 5mm」这种
    # 边界太紧的影响。资源区离放置区远，不会误伤。
    PLACE_FILTER_RADIUS_MM = 30.0

    def __init__(self):
        super().__init__("connecter")
        self.robot = Jaka()  # 返回一个机器人对象
        if not self.robot_is_ready():
            self.get_logger().error("robot is not ready")
            return

        # 放置区（只有一个位置）
        self.back_poses = [
            [-135.20, 375.0, 143.20, 180, 0.0, 0.0],
            [-70.20, 385.0, 143.20, 180, 0.0, 0.0],
            [-135.20, 325.393, 143.20, 180, 0.0, 0.0],
            [-70.20, 325.0, 143.20, 180, 0.0, 0.0],
            [-25.20, 325.393, 143.20, 180, 0.0, 0.0]
          ]

        # 末端位置，拾取移动距离 单位mm
        self.end_pose_z = 120.0
        self.end_move_z = -10      # 抓取时向下距离（mm，负=向下）
        self.place_down_z = -23    # 放置区向下距离（比抓取多10mm）
        self.place_up_z = 23      # 放置区释放后抬起距离（和向下高度相同）

        # 扫描位姿直接取上面示教好的 SCAN_POSES（6 自由度，任意姿态都行，转换时读实际位姿）
        self.scan_poses = [list(p) for p in self.SCAN_POSES]
        self.grab_index = 0  # 全局物块序号（决定放置区）

        self.command = {}
        self.srv_ser = self.create_service(
            StrMsg, "send_command", self.connect_callback_
        )
        # ros2 service call /restart_detection std_srvs/srv/Trigger "{}"
        self.srv_client = self.create_client(Trigger, "restart_detection")
        self.target_count = 0
        self.blocks_data = []
        self.targets = []  # 多目标 [{"color":"yellow","num":2}, ...]，按数量降序
        self.is_busy = False  # 抓取进行中标记：上一轮没结束就再次派单会被单线程执行器吞掉，这里直接拒绝

        # 面板动态字段发布（真实抓取时实时刷新 任务进度/当前任务/状态）
        # 面板订阅读的是 TRANSIENT_LOCAL，这里必须一致，否则 QoS 不兼容、面板收不到任何消息
        panel_qos = QoSProfile(
            depth=1,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.pub_progress = self.create_publisher(String, "/task/progress", panel_qos)
        self.pub_current = self.create_publisher(String, "/task/current_task", panel_qos)
        self.pub_status = self.create_publisher(String, "/task/status", panel_qos)

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

    def restart_detection(self, timeout_sec=10.0):
        """调用 /restart_detection 服务并等结果返回 response；失败/超时返回 None。

        供后台抓取线程调用：不能再 spin_once（会跟主 executor 打架、把本轮卡死成
        「只能抓一次」），而是轮询 future.done() 阻塞等待，靠主 executor 完成 future。
        （本版 rclpy 的 Future.result() 不带 timeout 参数，所以手动轮询做超时。）
        """
        if not self.srv_client.wait_for_service(timeout_sec=10.0):
            self.get_logger().warn("restart_detection 服务不可用")
            return None
        request = Trigger.Request()
        future = self.srv_client.call_async(request)
        deadline = time() + timeout_sec
        while not future.done():
            if time() >= deadline:
                self.get_logger().error("restart_detection 调用超时")
                return None
            sleep(0.05)
        try:
            return future.result()
        except Exception as e:
            self.get_logger().error(f"restart_detection 调用失败: {e}")
            return None

    def connect_callback_(self, request: StrMsg.Request, response: StrMsg.Response):
        """接收多目标抓取请求：
        request.data = {"targets":[{"color":"yellow","num":2},{"color":"blue","num":3}],"total":5}
        哪个颜色数量多就先抓哪个。
        """
        self.get_logger().info(f"request received: {request.data}")
        # 上一轮抓取还在跑时直接拒绝，避免请求在单线程执行器里排队、被当成「没反应」
        if self.is_busy:
            self.get_logger().warn("上一轮抓取尚未结束（is_busy），拒绝本次派单")
            response.success = False
            response.message = "busy: 上一轮抓取尚未结束，请等待完成后再派单"
            return response
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
        except Exception as e:
            self.get_logger().error(f"Error processing request: {str(e)}")
            response.success = False
            response.message = str(e)
            return response

        # 后台线程跑抓取，不阻塞 executor（否则本轮结束后收不到下一轮派单、只能重启服务）
        self.is_busy = True
        self.publish_panel(status="执行")  # 派单即切「执行」，进度/当前仍「—」直到第一个块开抓
        threading.Thread(target=self._grab_task, daemon=True).start()
        response.success = True
        response.message = "Command received, grab in progress"
        return response

    def _grab_task(self):
        """后台抓取线程入口：跑完整轮抓取，结束后无论成败都解除占用、回待命。

        抓取跑在独立线程，主 executor 只负责处理服务请求/响应，不再被长流程阻塞，
        所以本轮结束后能立刻接收下一轮派单（不用重启服务）。
        """
        try:
            self.process_restart_result()
        except Exception as e:
            self.get_logger().error(f"抓取流程异常终止: {e}")
        finally:
            self.is_busy = False
            try:
                self.robot.pick_end()
                self.robot.go_home()
            except Exception as e:
                self.get_logger().error(f"收尾回 home 失败: {e}")
            # 三格一起复位：状态=待命、进度=—、当前=—（「—」不匹配任何规则，面板显示 fallback「—」）
            self.publish_panel(progress="—", current="—", status="待命")
            self.get_logger().info("本轮任务结束，回到待命，可再次派单")

    def process_restart_result(self):
        """抓取主流程：颜色外层 × 扇区内层，每抓一个物块就回到扫描位姿重扫。

        关键改动：不再一次性把「一个扇区里所有该颜色块」抓完——那样第 2 个起用的
        是旧检测算出的过期坐标（抓完第一个机械臂已去了放置区，位姿变了会算错/抓偏）。
        改成每次只抓最居中那一个，抓完回扫描位姿重扫，保证每个块都用最新画面算坐标。
        """
        self.get_logger().info("开始抓取（每抓一个回扫一次）")
        self.robot.pick_init()
        self.grab_index = 0  # 全局物块序号（决定放置区，超过 5 个循环）

        for t in self.targets:
            color = t["color"]
            remaining = t["num"]

            # ---- 第一遍：逐扇区抓居中块，抓一个重扫一次 ----
            for scan_pose in self.scan_poses:
                while remaining > 0:
                    if not self.move_to_scan(scan_pose):
                        break
                    grabbable = self.current_grabbable(color, centered_only=True)
                    if not grabbable:
                        break  # 本扇区没有该颜色居中块了，换下一扇区
                    pos, b = grabbable[0]  # 最居中
                    if self.grab_one(pos, b, color):
                        remaining -= 1
                    else:
                        break  # 抓失败，换下一扇区

            # ---- 兜底 B：还差就逐扇区抓离中心最近的块（不限 R） ----
            if remaining > 0:
                self.get_logger().warn(
                    f"{color} 居中块抓完还差 {remaining} 个，走兜底抓边缘块"
                )
                for scan_pose in self.scan_poses:
                    while remaining > 0:
                        if not self.move_to_scan(scan_pose):
                            break
                        grabbable = self.current_grabbable(color, centered_only=False)
                        if not grabbable:
                            break
                        pos, b = grabbable[0]
                        if self.grab_one(pos, b, color):
                            remaining -= 1
                        else:
                            break

            if remaining > 0:
                self.get_logger().warn(f"{color} 最终还差 {remaining} 个没抓到")

    def move_to_scan(self, scan_pose):
        """移动到示教好的扫描位姿（go_pose，6 自由度）。到位后等机械臂停稳。"""
        ret = self.robot.go_pose(scan_pose)
        if isinstance(ret, tuple) and ret[0] != 0:
            self.get_logger().error(f"移动到扫描位姿失败: {scan_pose} -> {ret}")
            return False
        sleep(2.0)  # 等停稳；「新画面」由 detect_blocks_sync 的 seq 等待保证
        return True

    def detect_blocks_sync(self, timeout=6.0):
        """等机械臂停稳后，拿到「新画面」的检测结果。

        机械臂移动过程中 DNN 会持续吐检测（多为 0 块），直接取第一次结果拿到的
        很可能是移动残留的旧画面——这就是之前「每个扇区都 0 个块」的原因。
        这里先读一次 seq，再等到 seq 变化（说明停稳后又出了新检测），返回那一刻的块。
        """
        deadline = time() + timeout
        first_seq = None
        last_blocks = None
        while time() < deadline:
            result = self.restart_detection()
            if not result:
                sleep(0.5)
                continue
            try:
                payload = json.loads(result.message)
            except Exception as e:
                self.get_logger().error(f"解析方块结果失败: {result.message} ({e})")
                sleep(0.5)
                continue
            if not isinstance(payload, dict):
                self.get_logger().error(
                    f"检测返回格式异常（需新版 pose_get）: {str(result.message)[:80]}"
                )
                sleep(0.5)
                continue
            seq = payload.get("seq")
            blocks = payload.get("blocks")
            if not isinstance(blocks, list):
                blocks = []
            last_blocks = blocks
            if first_seq is None:
                first_seq = seq  # 第一次读到的代次（可能是移动残留）
                sleep(0.5)
                continue
            if seq != first_seq:
                self.get_logger().info(f"本扇区检测到 {len(blocks)} 个块 (seq={seq})")
                return blocks
            sleep(0.5)
        self.get_logger().warn("等待新鲜检测超时")
        return last_blocks

    def current_grabbable(self, color, centered_only):
        """当前扫描位姿下，返回该颜色可抓块列表 [(base_pos, block), ...]。

        - 只取该颜色块（可选：只取「居中」块 dist_to_center < CENTER_RADIUS）；
        - 统一转基座坐标并过滤放置区（blocks_to_base 内做）；
        - 按离画面中心由近到远排序，调用方抓第一个。
        每次只抓一个后回扫，所以这里只用「当前这一刻」的位姿算一次，坐标不会过期。
        """
        blocks = self.detect_blocks_sync()
        if not blocks:
            return []
        current_pose = self.robot.get_tools_pos()
        if current_pose == -1:
            self.get_logger().error("获取工具位姿失败")
            return []
        cand = [b for b in blocks if b.get("color") == color]
        if centered_only:
            cand = [b for b in cand if b.get("dist_to_center", 1e9) < self.CENTER_RADIUS]
        base_list = self.blocks_to_base(cand, current_pose)  # 已过滤放置区
        base_list.sort(key=lambda pb: pb[1].get("dist_to_center", 1e9))
        return base_list

    def grab_one(self, pos, b, color):
        """抓取单个块（基座坐标已算好），返回是否成功。"""
        self.grab_index += 1
        self.get_logger().info(
            f"抓取第{self.grab_index}个：{color}（dist={b.get('dist_to_center')}）"
        )
        try:
            ok = self.pick(pos[0], pos[1], pos[2], self.grab_index)
        except Exception as e:
            self.get_logger().error(
                f"抓取第{self.grab_index}个({color})异常：{e}"
            )
            ok = False
        if not ok:
            self.get_logger().error(f"抓取第{self.grab_index}个({color})失败")
        return ok

    def blocks_to_base(self, blocks, current_pose):
        """把一组块（tool 坐标）转成基座坐标，返回 [(base_pos, block), ...]。
        落在过滤区（放置区）的块直接丢弃，不参与抓取；转换失败的也跳过。"""
        out = []
        for b in blocks:
            tool_pos = b.get("tool_pos")
            if not tool_pos or len(tool_pos) < 3:
                continue
            pos = self.correct_pos(tool_to_base(tool_pos, current_pose))
            # 过滤放置区：x∈[-200,0] 且 y∈[320,520]（mm）内的块跳过
            x_mm = pos[0] * 1000.0
            y_mm = pos[1] * 1000.0
            color = b.get("color", "?")
            in_rect = (self.FILTER_X_MIN <= x_mm <= self.FILTER_X_MAX
                       and self.FILTER_Y_MIN <= y_mm <= self.FILTER_Y_MAX)
            near_place = self._is_in_placement(x_mm, y_mm)
            # 诊断：打印每个块的校正后基座坐标 + 过滤判定，核对过滤区边界是否包住放置区
            if in_rect or near_place:
                self.get_logger().info(
                    f"[过滤诊断] color={color} base=({x_mm:.1f}, {y_mm:.1f})mm "
                    f"矩形内={in_rect} 近放置点={near_place} → 跳过"
                )
                continue
            self.get_logger().info(
                f"[过滤诊断] color={color} base=({x_mm:.1f}, {y_mm:.1f})mm → 参与抓取"
            )
            out.append((pos, b))
        return out

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

    def correct_pos(self, pos):
        """对计算出的基座坐标施加旋转+平移校正，补偿手眼标定残差。
        pos 单位米，返回校正后的 [x, y, z]（米）。
        左右分开：按校正前的 y 正负选 L/R 组参数（物块都在 |y|≈250，离 y=0 远，分类稳定）。
        """
        x = pos[0] * 1000.0
        y = pos[1] * 1000.0
        if y < 0:
            th = math.radians(self.CORRECT_THETA_L)
            tx, ty = self.CORRECT_TX_L, self.CORRECT_TY_L
        else:
            th = math.radians(self.CORRECT_THETA_R)
            tx, ty = self.CORRECT_TX_R, self.CORRECT_TY_R
        xr = x * math.cos(th) - y * math.sin(th)
        yr = x * math.sin(th) + y * math.cos(th)
        return [
            (xr + tx) / 1000.0,
            (yr + ty) / 1000.0,
            pos[2],
        ]

    def _is_in_placement(self, x_mm, y_mm):
        """判断校正后的基座坐标 (x,y) 是否落在任一放置点附近（已放置的块）。

        用「到 back_poses 的平面距离」判断，比固定矩形稳：放置点就是机械臂放块的
        坐标，误差再大也不会差出 30mm；而资源区离放置区远，不会误伤。
        """
        for bp in self.back_poses:
            if math.hypot(x_mm - bp[0], y_mm - bp[1]) < self.PLACE_FILTER_RADIUS_MM:
                return True
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
        self.publish_panel(progress=f"第{index}个", current="前往资源点", status="执行")
        offset_x = 0
        offset_y = 0 if y > 0 else -6    #40 -30左边偏右下角
        if not self.move_to_point(
            x * 1000 + offset_x, y * 1000 + offset_y, self.end_pose_z
        ):
            return False
        sleep(1)

        # 2. 抓取资源点（吸取）
        self.publish_panel(current="抓取资源", status="执行")
        self.get_logger().info(f"第{index}个：开始吸取（下压 {self.end_move_z}mm）")
        self.robot.do_pick_on(self.end_move_z)
        self.get_logger().info(f"第{index}个：吸取完成，抬起")
        sleep(1)

        # 3. 前往放置区（第 index 个物块 → 第 index 个放置区，超过 5 个循环）
        #    无论成功失败都必须释放吸盘：绝不吸着物块去下一个物块
        self.publish_panel(current="前往放置区", status="执行")
        back_pose = self.back_poses[(index - 1) % len(self.back_poses)]
        ret = self.robot.go_pose(back_pose)
        if isinstance(ret, tuple) and ret[0] != 0:
            self.get_logger().error(f"前往放置区失败: {ret}")

        # 到达放置区后，先向下移动 place_down_z（比抓取多10mm），再放
        self.robot.robot.linear_move(
            [0, 0, self.place_down_z, 0, 0, 0], INCR, True, 12
        )
        sleep(0.5)

        # 不管上面成没成，先把吸盘松开（避免带着物块跑）
        self.robot.pick_off()
        # 释放后抬起来：抬起高度 = 放置区下降高度 place_up_z
        self.robot.robot.linear_move([0, 0, self.place_up_z, 0, 0, 0], INCR, True, 30)

        # ★ 关键：必须返回布尔值。放置区到达成功才算这个物块抓成功。
        #   漏了 return 会隐式返回 None，调用方把 None 当「失败」，永远停在第一个。
        return not (isinstance(ret, tuple) and ret[0] != 0)
        
        

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
