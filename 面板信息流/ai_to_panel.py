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

用法：
  python3 ai_to_panel.py "{color:yellow,num:4},{color:blue,num:1}"
  python3 ai_to_panel.py "{color:yellow,num:4},{color:blue,num:1}" --simulate   # 2s/步演示进度
  python3 ai_to_panel.py --watch /tmp/ai_answer.txt   # 监听文件（deepseek 写完即解析）
"""
import argparse
import json
import os
import sys
import time

from parse_ai_answer import parse_ai_answer

CN = {'yellow': '黄色', 'blue': '蓝色'}

# 方块颜色唯一来源：按 parse 结果里的 color 字段查表，不硬编码十六进制色值
BLOCK_COLOR = {
    'yellow': (1.00, 0.85, 0.15, 1.0),   # 黄
    'blue':   (0.17, 0.35, 0.93, 1.0),   # 蓝
}


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
        # 动态字段（任务进度/当前任务/状态）在真实抓取时由 connecter 接管：
        #   状态 = 前往抓取（吸盘空）/ 前往放置（吸盘有物块）；全部放完回 待命
        '当前任务': '前往抓取',
        '状态': '待命',
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


def publish_idle_dynamic(pubs):
    """把动态字段（任务进度/当前任务/状态）复位回「待命」。

    真实抓取时这三个字段由 connecter 节点接管（前往抓取 / 前往放置），这里只在
    新答案到来时把面板复位一次，避免残留上一轮抓取的状态。复位后不再重发动态
    字段，防止把 connecter 的实时进度覆盖掉。
    """
    from std_msgs.msg import String
    pubs['progress'].publish(String(data="第1个"))
    pubs['current'].publish(String(data="前往抓取"))
    pubs['status'].publish(String(data="待命"))


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


def _watch(rclpy, path):
    """监听文件：内容变化后稳定 50ms 才解析发布；并每 0.5s 重发最近结果兜底。

    为什么常驻重发：std_msgs 主题是易失的（不缓存历史），coStudio 一旦重连 / 重新导入
    布局，之前发过的消息就收不到了。发布器已用 transient_local QoS 保留最后值，这里
    再每 0.5s 重发一遍双保险，保证面板任何时候都能在 0.5s 内同步回最新值。
    """
    from rclpy.node import Node
    last_mtime = None
    last_content = None
    last_state = None
    last_pub = 0.0
    node = Node('ai_to_panel')
    pubs = make_publishers(node)   # 提前建好，让 /task/* 主题一开跑就存在
    print(f"监听 {path} ...（每 0.5s 重发最近结果兜底，Ctrl+C 停止）")
    try:
        while True:
            if os.path.exists(path):
                mtime = os.path.getmtime(path)
                if mtime != last_mtime:
                    last_mtime = mtime
                    time.sleep(0.05)   # 等 50ms（deepseek 单次 write+close，几乎无半截风险）
                    with open(path, 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                    if content != last_content:
                        last_content = content
                        counts = parse_ai_answer(content)
                        if counts['yellow'] is None and counts['blue'] is None:
                            print(f"⚠️ 未解析到答案：{content[:80]!r}", file=sys.stderr)
                        else:
                            last_state = build_panel_state(counts)
                            labels = publish_state(node, last_state, pubs)
                            publish_idle_dynamic(pubs)
                            print("已发布：", ", ".join(f"{k}={v}" for k, v in labels.items()))
                    else:
                        print(f"文件内容与上次相同（答案没变），跳过：{content[:40]!r}")
            # 常驻重发兜底：transient_local 之外再每 0.5s 重发一次，双保险
            if last_state is not None and time.time() - last_pub >= 0.5:
                publish_state(node, last_state, pubs)
                last_pub = time.time()
            time.sleep(0.05)
    except KeyboardInterrupt:
        pass


def main():
    ap = argparse.ArgumentParser(description='把 AI 解题结果实时发布到 ROS / coStudio')
    ap.add_argument('result', nargs='?', help='AI 解题结果，如 "{color:yellow,num:4},{color:blue,num:1}"')
    ap.add_argument('--simulate', action='store_true', help='2s/步模拟抓取进度')
    ap.add_argument('--watch', help='监听文件，内容稳定 300ms 后解析发布')
    args = ap.parse_args()

    import rclpy
    rclpy.init(args=sys.argv[1:1])

    if args.watch:
        _watch(rclpy, args.watch)
        rclpy.shutdown()
        return

    if not args.result:
        print("用法：python3 ai_to_panel.py \"{color:yellow,num:4},{color:blue,num:1}\" [--simulate]")
        print("      python3 ai_to_panel.py --watch /tmp/ai_answer.txt")
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
    publish_idle_dynamic(pubs)
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
