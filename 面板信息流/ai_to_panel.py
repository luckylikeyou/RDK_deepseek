#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 AI 解题结果实时发布到 ROS，供 coStudio 面板显示。

对应任务规范「四、面板集成」。结构上严格拆成三层（纯函数与副作用分离）：

  1. parse_ai_answer(text)         纯函数：噪声文本 -> {yellow, blue}（见 parse_ai_answer.py）
  2. build_panel_state(counts)     纯函数：{yellow,blue} -> 标签 / 抓取顺序 / 方块列表
  3. publish_state(node, state)    ROS 发布（副作用层）：数量 / 先抓后抓 / 进度 / 3D 方块

流式安全（方案 A）：AI 回答是流式吐字时，不解析半截 JSON——等「回答结束」信号
（命令行模式即整条参数；--watch 模式即文件 300ms 不再变化）再调用 parse。
解析器本身也只匹配「完整闭合的 {color,num}」，半截对象（如 {color:yell）天然不匹配。

发布主题（供 coStudio 面板）：
  /task/yellow_target、/task/blue_target  std_msgs/Int32              -> 黄色/蓝色数量
  /task/grab_first、/task/grab_second     std_msgs/String             -> 先抓取/后抓取
  /task/progress                          std_msgs/String "第N个"     -> 任务进度
  /task/status                            std_msgs/String             -> 状态
  /task/plan                              std_msgs/String(JSON)       -> 完整抓取顺序
  /task/blocks                            visualization_msgs/MarkerArray -> 3D 面板按数量渲染方块

订阅主题（供开始按钮）：
  /task/start                             std_msgs/String             -> "basic"/"challenge"（按下即选任务并出题）

用法：
  python3 ai_to_panel.py "{color:yellow,num:4},{color:blue,num:1}"
  python3 ai_to_panel.py "{color:yellow,num:4},{color:blue,num:1}" --simulate   # 2s/步演示进度
  python3 ai_to_panel.py --watch /tmp/ai_answer.txt   # 监听文件（deepseek 写完即解析）
  python3 ai_to_panel.py --watch /tmp/ai_answer.txt --question /tmp/question.txt   # 按钮出题 + 题目上屏
"""
import argparse
import json
import os
import subprocess
import sys
import threading
import time

from parse_ai_answer import parse_ai_answer

CN = {'yellow': '黄色', 'blue': '蓝色'}

# 方块颜色唯一来源：按 parse 结果里的 color 字段查表，不硬编码十六进制色值
BLOCK_COLOR = {
    'yellow': (1.00, 0.85, 0.15, 1.0),   # 黄
    'blue':   (0.17, 0.35, 0.93, 1.0),   # 蓝
}

PROBLEM_TOPIC = "/task/problem"


def read_question_file(path):
    """读题目文件：GBK 解码、去「题目: 」前缀，返回干净题目文本。

    题目文件由 `./TMSCQtest_arm.bin > /tmp/question.txt` 生成（GBK 字节、带「题目: 」前缀）。
    """
    with open(path, "rb") as f:
        raw = f.read()
    text = None
    for enc in ("gbk", "utf-8"):
        try:
            text = raw.decode(enc).strip()
            break
        except UnicodeDecodeError:
            continue
    if text is None:
        text = raw.decode("utf-8", errors="replace").strip()
    text = text.strip().lstrip("\ufeff")
    for prefix in ("题目:", "题目："):
        if text.startswith(prefix):
            text = text[len(prefix):]
            break
    return text.strip()


def build_result_text(counts):
    """{yellow, blue}（值可能为 None）→ "结果：黄色N个、蓝色M个"；全空返回空串。"""
    parts = []
    for c in ("yellow", "blue"):
        n = counts.get(c)
        if n is not None:
            parts.append(f"{CN[c]}{n}个")
    if not parts:
        return ""
    return "结果：" + "、".join(parts)


def apply_task_rule(mode, counts):
    """按任务类型把「解题结果」换算成「最终抓取数量」。

    challenge：解出几个抓几个（counts 原样）。
    basic：固定抓 3 个——数量多的颜色抓 2、少的抓 1；相等按黄2蓝1。
    某颜色原本没出现(None)则保持不抓，避免去抓不存在的颜色。
    """
    if mode != "basic":
        return dict(counts)
    y = counts.get("yellow") or 0
    b = counts.get("blue") or 0
    out = {"yellow": 2, "blue": 1} if y >= b else {"yellow": 1, "blue": 2}
    if counts.get("yellow") is None:
        out["yellow"] = None
    if counts.get("blue") is None:
        out["blue"] = None
    return out


QUESTION_GEN_CMD_DEFAULT = "./TMSCQtest_arm.bin > /tmp/question.txt"


def run_generator():
    """运行题目生成器，把题目写进 /tmp/question.txt，返回是否成功。

    命令用环境变量 QUESTION_GEN_CMD 覆盖（默认 ./TMSCQtest_arm.bin > /tmp/question.txt）。
    生成器阻塞（最长 ~15s），调用方务必放到子线程，别卡住 ROS executor。
    """
    cmd = os.environ.get("QUESTION_GEN_CMD", QUESTION_GEN_CMD_DEFAULT)
    try:
        subprocess.check_call(cmd, shell=True, timeout=20)
        return True
    except Exception as e:
        print(f"[出题] 运行生成器失败：{e}", file=sys.stderr)
        return False


def build_panel_state(counts):
    """纯函数：{yellow, blue} -> 面板状态。

    counts 形如 parse_ai_answer 的返回值 {'yellow': 4, 'blue': 1}，None 表示该颜色没出现。
    抓取顺序：数量多者先抓、少者后抓；完整序列 seq 用于进度「第N个」。
    """
    yellow = counts.get('yellow')
    blue = counts.get('blue')

    present = [c for c in ('yellow', 'blue') if counts.get(c) is not None]
    order = sorted(present, key=lambda c: -counts[c])   # 数量降序：多的先抓
    if len(order) < 2:
        first = order[0] if order else 'yellow'
        second = 'blue' if first == 'yellow' else 'yellow'
    else:
        first, second = order[0], order[1]

    seq = []                                   # 完整抓取序列：先抓完多的再抓少的
    for c in order:
        seq += [c] * counts[c]

    labels = {
        '黄色数量': str(yellow) if yellow is not None else '—',
        '蓝色数量': str(blue) if blue is not None else '—',
        '先抓取': CN.get(first, first),
        '后抓取': CN.get(second, second),
        '任务进度': '第1个',
        # 动态字段（任务进度/当前任务/状态）：结果一上屏即进入「执行」，
        #   随后由 connecter 接管（前往资源点 / 抓取资源点 / 前往放置区），放完回 待命。
        #   面板状态词只有 待命/执行 两个，没有「已完成」。
        '当前任务': '',
        '状态': '执行',
    }
    return {
        'labels': labels,
        'order': order,
        'seq': seq,
        'blocks': [{'color': c} for c in seq],
        'counts': counts,
    }


def render_blocks(blocks):
    """方块列表 -> visualization_msgs/MarkerArray（横向排成一排）。

    颜色由 block['color'] 查 BLOCK_COLOR 得到。无 viz_msgs 时返回 None。
    """
    try:
        from visualization_msgs.msg import Marker, MarkerArray
    except ImportError:
        print("⚠️ 未安装 visualization_msgs，跳过 /task/blocks 方块渲染。"
              "（数量仍会发布到 /task/yellow_target 等）", file=sys.stderr)
        return None

    arr = MarkerArray()
    size = 0.06
    gap = 0.08
    for i, b in enumerate(blocks):
        r, g, bl, a = BLOCK_COLOR.get(b['color'], (0.5, 0.5, 0.5, 1.0))
        m = Marker()
        m.header.frame_id = 'map'
        m.ns = 'task_blocks'
        m.id = i
        m.type = Marker.CUBE
        m.action = Marker.ADD
        m.pose.position.x = float(i * gap)
        m.pose.position.y = 0.0
        m.pose.position.z = 0.0
        m.pose.orientation.w = 1.0
        m.scale.x = m.scale.y = m.scale.z = size
        m.color.r, m.color.g, m.color.b, m.color.a = r, g, bl, a
        arr.markers.append(m)
    return arr


def make_publishers(node):
    """创建全部面板主题的发布器。

    用 transient_local QoS（KEEP_LAST + depth=1 + TRANSIENT_LOCAL）：DDS 保留
    最后一次发布的值，任何后订阅/重连的面板一订阅就立刻收到最新值，不用靠定时
    重发来补。这解决「同一批里黄色先变、蓝色滞后」的订阅时机不一致问题。
    """
    from std_msgs.msg import Int32, String
    from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy, QoSDurabilityPolicy
    qos = QoSProfile(
        history=QoSHistoryPolicy.KEEP_LAST,
        depth=1,
        reliability=QoSReliabilityPolicy.RELIABLE,
        durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
    )
    pubs = {
        'yellow':   node.create_publisher(Int32, '/task/yellow_target', qos),
        'blue':     node.create_publisher(Int32, '/task/blue_target', qos),
        'first':    node.create_publisher(String, '/task/grab_first', qos),
        'second':   node.create_publisher(String, '/task/grab_second', qos),
        'progress': node.create_publisher(String, '/task/progress', qos),
        'current':  node.create_publisher(String, '/task/current_task', qos),
        'status':   node.create_publisher(String, '/task/status', qos),
        'plan':     node.create_publisher(String, '/task/plan', qos),
        'problem':  node.create_publisher(String, PROBLEM_TOPIC, qos),
    }
    
    try:
        from visualization_msgs.msg import MarkerArray
        pubs['blocks'] = node.create_publisher(MarkerArray, '/task/blocks', qos)
    except ImportError:
        pubs['blocks'] = None
    return pubs


def publish_state(node, state, pubs=None):
    """把面板状态发布到 ROS（副作用层），返回发布出去的标签。

    pubs 可选：常驻重发时传入 make_publishers 建好的发布器，避免每次重建。
    """
    from std_msgs.msg import Int32, String
    if pubs is None:
        pubs = make_publishers(node)

    labels = state['labels']
    counts = state['counts']
    if counts.get('yellow') is not None:
        pubs['yellow'].publish(Int32(data=counts['yellow']))
    if counts.get('blue') is not None:
        pubs['blue'].publish(Int32(data=counts['blue']))
    pubs['first'].publish(String(data=labels['先抓取']))
    pubs['second'].publish(String(data=labels['后抓取']))
    pubs['plan'].publish(String(data=json.dumps({
        'order': state['order'],
        'seq': state['seq'],
        'counts': {c: counts[c] for c in counts if counts[c] is not None},
    }, ensure_ascii=False)))

    if pubs.get('blocks') is not None:
        arr = render_blocks(state['blocks'])
        if arr is not None:
            pubs['blocks'].publish(arr)

    return labels


def publish_running_dynamic(pubs):
    """新答案到来时把动态字段（任务进度/当前任务/状态）置为「执行中」初始态。

    结果一上屏即表示任务已下发、机械臂即将开始抓取，所以状态写「执行」而非「待命」。
    真实抓取时这三个字段由 connecter 节点接管（前往资源点 / 抓取资源点 / 前往放置区），
    这里只在新答案到来时置一次初始态，之后不再重发，防止把 connecter 的实时进度覆盖掉。
    """
    from std_msgs.msg import String
    pubs['progress'].publish(String(data="第1个"))
    pubs['current'].publish(String(data=""))
    pubs['status'].publish(String(data="执行"))


def dispatch_to_arm(node, rclpy, counts):
    """把解析结果转成 /send_command 的 targets JSON，派单给机械臂抓取。

    一次新答案只派一次（由调用方保证）。counts = {'yellow': N, 'blue': M}，
    None 表示该颜色没出现。机械臂侧 connecter 收到后按「数量多者先抓」执行，
    并实时刷面板的 任务进度/当前任务/状态（前往资源点/抓取资源点/前往放置区）。
    """
    try:
        from custom_msgs.srv import StrMsg
    except ImportError:
        print("⚠️ 未找到 custom_msgs，跳过机械臂派单（需 source ~/code_ws/install）",
              file=sys.stderr)
        return False

    targets, total = [], 0
    for color in ('yellow', 'blue'):
        num = counts.get(color)
        if num is not None and num > 0:
            targets.append({"color": color, "num": int(num)})
            total += int(num)
    if not targets:
        print("⚠️ 黄/蓝数量都为空，跳过机械臂派单", file=sys.stderr)
        return False

    targets.sort(key=lambda t: -t["num"])   # 数量多者先抓
    data = json.dumps({"targets": targets, "total": total}, ensure_ascii=False)

    client = node.create_client(StrMsg, "/send_command")
    if not client.wait_for_service(timeout_sec=3.0):
        print("⚠️ /send_command 服务未就绪（connecter 未启动或机械臂未登录），跳过派单",
              file=sys.stderr)
        return False

    req = StrMsg.Request()
    req.data = data
    future = client.call_async(req)
    rclpy.spin_until_future_complete(node, future, timeout_sec=5.0)
    print(f"→ 已派单给机械臂：{data}")
    return True


def _simulate(node, state, pubs=None):
    """用已建好的 node 模拟抓取进度，演示「任务进度/当前任务/状态」实时变化。

    状态机（按面板规则）：
      状态 = 执行（全程），全部放完回到 待命；
      每个资源依次：前往资源点 → 抓取资源 → 前往放置区。
    """
    from std_msgs.msg import String
    if pubs is None:
        pubs = make_publishers(node)
    seq = state['seq']
    print(f"模拟抓取 {len(seq)} 个资源：{'、'.join(CN[c] for c in seq)}")
    try:
        pubs['status'].publish(String(data="执行"))
        for i, c in enumerate(seq, 1):
            pubs['progress'].publish(String(data=f"第{i}个"))
            pubs['current'].publish(String(data="前往资源点"))
            print(f"  第{i}个（{CN[c]}）：前往资源点")
            time.sleep(2)
            pubs['current'].publish(String(data="抓取资源"))
            print(f"  第{i}个（{CN[c]}）：抓取资源")
            time.sleep(2)
            pubs['current'].publish(String(data="前往放置区"))
            print(f"  第{i}个（{CN[c]}）：前往放置区")
            time.sleep(2)
        pubs['status'].publish(String(data="待命"))
        print("  完成 → 待命")
    except KeyboardInterrupt:
        pass


def _watch(rclpy, path, question_path=None):
    """监听答案文件（+ 可选题目文件），内容变化后解析发布；每 0.5s 重发兜底。

    question_path：题目文件（如 /tmp/question.txt），变化即把题目发到 /task/problem；
    答案变化时把「结果」追加到题目后面一起发。

    新增「开始按钮」：订阅 /task/start（std_msgs/String，值为 basic/challenge），
    按下即 (1) 选任务类型 (2) 后台跑题目生成器（默认 ./TMSCQtest_arm.bin > /tmp/question.txt，
    可用 QUESTION_GEN_CMD 覆盖）。生成器出题后由 deepseek_client 解题写回答案文件，
    本节点再把「解题结果」按任务类型换算后派单抓取。
    """
    from rclpy.node import Node
    from std_msgs.msg import String
    last_mtime = None
    last_state = None
    last_pub = 0.0

    # 题目显示状态
    problem_text = ""          # 当前题目原文
    problem_message = ""       # 发到 /task/problem 的完整内容（题目 + 结果）
    last_q_mtime = None

    # 当前任务类型：默认挑战任务（和加按钮前「解几个抓几个」行为一致）
    mode = "challenge"

    node = Node('ai_to_panel')
    pubs = make_publishers(node)

    # ---- 开始按钮：Publish 面板点一下发一条 String("basic"/"challenge") 到 /task/start ----
    def on_start(msg):
        nonlocal mode
        task = (msg.data or "").strip().lower()
        if task not in ("basic", "challenge"):
            print(f"⚠️ 未知任务类型 {msg.data!r}（应为 basic 或 challenge），忽略", file=sys.stderr)
            return
        mode = task
        print(f"→ 收到开始按钮：{task}，启动出题程序…")
        threading.Thread(target=_run_generator, args=(task,), daemon=True).start()

    def _run_generator(task):
        if run_generator():
            print(f"[{task}] 题目已生成，等待 DeepSeek 解题…")
        else:
            print(f"[{task}] 出题失败，任务未开始", file=sys.stderr)

    node.create_subscription(String, "/task/start", on_start, 10)

    if question_path:
        print(f"监听题目 {question_path} ...")
    print(f"监听答案 {path} ...（每 0.5s 重发兜底，Ctrl+C 停止）")
    try:
        while True:
            # ---- 题目文件：mtime 变了就重新上屏 ----
            if question_path and os.path.exists(question_path):
                q_mtime = os.path.getmtime(question_path)
                if q_mtime != last_q_mtime:
                    last_q_mtime = q_mtime
                    time.sleep(0.05)   # 等生成器写完整（> 会先截断再写）
                    problem_text = read_question_file(question_path)
                    if not problem_text:
                        print("⚠️ 题目文件为空（可能生成器还没写完），跳过", file=sys.stderr)
                    else:
                        problem_message = problem_text   # 新题清掉旧结果
                        pubs['problem'].publish(String(data=problem_message))
                        print(f"已发布题目到 {PROBLEM_TOPIC}：{problem_text}")

            # ---- 答案文件：mtime 变了就解析 + 按任务类型换算 + 派单 ----
            if os.path.exists(path):
                mtime = os.path.getmtime(path)
                if mtime != last_mtime:
                    last_mtime = mtime
                    time.sleep(0.05)
                    with open(path, 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                    counts = parse_ai_answer(content)
                    if counts['yellow'] is None and counts['blue'] is None:
                        print(f"⚠️ 未解析到答案：{content[:80]!r}", file=sys.stderr)
                    else:
                        final = apply_task_rule(mode, counts)
                        if final != counts:
                            print(f"[{mode}] 解题 黄={counts.get('yellow')} 蓝={counts.get('blue')}"
                                  f" → 抓取 黄={final.get('yellow')} 蓝={final.get('blue')}")
                        last_state = build_panel_state(final)
                        labels = publish_state(node, last_state, pubs)
                        publish_running_dynamic(pubs)
                        print("已发布：", ", ".join(f"{k}={v}" for k, v in labels.items()))
                        # 先让面板把「映射结果」（黄/蓝数量、先抓/后抓、3D 方块）收到并渲染出来，
                        # 再派单让机械臂动。否则发布和派单几乎同时，面板还没上屏机械臂就动了。
                        try:
                            map_delay = float(os.environ.get("PANEL_MAP_DELAY", "2.0"))
                        except ValueError:
                            map_delay = 2.0
                        if map_delay > 0:
                            rclpy.spin_once(node, timeout_sec=0.5)  # 泵一次把发布刷出去
                            time.sleep(map_delay)                    # 等 coStudio 渲染映射
                        dispatch_to_arm(node, rclpy, final)
                        # 结果追加到题目后面，一起发到「题目」面板
                        result = build_result_text(final)
                        if result and problem_text:
                            problem_message = f"{problem_text}\n{result}"
                            pubs['problem'].publish(String(data=problem_message))
                            print(f"已追加结果到题目：{result}")

            # ---- 兜底重发 ----
            if time.time() - last_pub >= 0.5:
                if last_state is not None:
                    publish_state(node, last_state, pubs)
                if problem_message:
                    pubs['problem'].publish(String(data=problem_message))
                last_pub = time.time()

            # 泵一次 executor：既当循环节拍，又处理 /task/start 按钮回调
            rclpy.spin_once(node, timeout_sec=0.05)
    except KeyboardInterrupt:
        pass


def main():
    ap = argparse.ArgumentParser(description='把 AI 解题结果实时发布到 ROS / coStudio')
    ap.add_argument('result', nargs='?', help='AI 解题结果，如 "{color:yellow,num:4},{color:blue,num:1}"')
    ap.add_argument('--simulate', action='store_true', help='2s/步模拟抓取进度')
    ap.add_argument('--watch', help='监听答案文件，内容变化后解析发布')
    ap.add_argument('--question', help='监听题目文件（如 /tmp/question.txt），变化即发 /task/problem')
    args = ap.parse_args()

    import rclpy
    rclpy.init(args=sys.argv[1:1])

    if args.watch:
        _watch(rclpy, args.watch, args.question)
        rclpy.shutdown()
        return

    if not args.result:
        print("用法：python3 ai_to_panel.py \"{color:yellow,num:4},{color:blue,num:1}\" [--simulate]")
        print("      python3 ai_to_panel.py --watch /tmp/ai_answer.txt")
        print("      python3 ai_to_panel.py --watch /tmp/ai_answer.txt --question /tmp/question.txt")
        rclpy.shutdown()
        sys.exit(1)

    from rclpy.node import Node
    node = Node('ai_to_panel')

    counts = parse_ai_answer(args.result)
    if counts['yellow'] is None and counts['blue'] is None:
        print(f"⚠️ 未解析到有效答案（黄/蓝都为空），不发布：{args.result[:80]!r}", file=sys.stderr)
        rclpy.shutdown()
        sys.exit(1)

    state = build_panel_state(counts)
    pubs = make_publishers(node)
    labels = publish_state(node, state, pubs)
    publish_running_dynamic(pubs)
    print("已发布：", ", ".join(f"{k}={v}" for k, v in labels.items()))

    if args.simulate:
        _simulate(node, state, pubs)
    else:
        # 常驻发布：每 2s 重发一次。否则只发一次就退出，cobridge 可能还没
        # 发现/订阅这些话题，面板就收不到值（跟 ros2 topic pub -1 一样的坑）。
        node.create_timer(2.0, lambda: publish_state(node, state, pubs))
        print("持续发布中（每 2s 重发一次），Ctrl+C 停止。")
        try:
            rclpy.spin(node)
        except KeyboardInterrupt:
            pass

    rclpy.shutdown()


if __name__ == '__main__':
    main()
