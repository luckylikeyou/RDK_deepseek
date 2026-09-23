#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DeepSeek API 客户端 —— 可直接在 RDK X5 (Ubuntu aarch64) 上运行。

用法:
    # 单次提问
    python3 deepseek_client.py "介绍一下RDK X5"

    # 交互式聊天（多轮对话，保留上下文）
    python3 deepseek_client.py -i

    # 自动：运行题目生成器出题，自动解题 -> 上板 -> 派单给机械臂抓取
    python3 deepseek_client.py --auto

    # 指定模型 / 自定义系统提示词
    python3 deepseek_client.py "你好" --model deepseek-chat --system "你是机器人助手"

    # 环境变量里放 Key（推荐，避免写死在代码里）
    export DEEPSEEK_API_KEY=sk-xxxx
    python3 deepseek_client.py -i

依赖:
    pip3 install openai
    机械臂派单需 source ROS 工作区（custom_msgs），并已启动 connecter 节点（提供 /send_command）。
"""

import os
import sys
import re
import json
import math
import time
import subprocess
import argparse

# 默认模型
DEFAULT_MODEL = "deepseek-chat"
BASE_URL = "https://api.deepseek.com"

DEFAULT_SYSTEM = """你负责从应用题中提取数量，绝对不要做任何计算、求和、除法、取整。

【题目模式】
题目里有两种物品，第一种对应"黄色"，第二种对应"蓝色"。
题目会列出每个人领取/需要的数量。

【规则】
1. 只提取"每个人领取/需要的数量"，逐个放进数组。
2. 绝对不要提取"每个XX有N个/件"里的N，那是每个容器的容量，不是领取数量。
3. 没有提到的物品不要写，不要把0写进去。
4. 输出里不能出现任何计算、求和、解释。

【输出格式】
只输出一行 JSON，不要任何其他文字、不要换行：
{"yellow":[第一种物品的每个数量],"blue":[第二种物品的每个数量]}

【示例】
题目：每个盒子有5个球;有红球、蓝球两种盒子;小明拿了2个红球,3个蓝球;小红拿了4个红球;小刚拿了5个蓝球。
输出：{"yellow":[2,4],"blue":[3,5]}

题目：每个宠物店有5只宠物;有猫、狗两种宠物店;小爱领养2只猫,5只狗;小宠、小动都领养2只猫;小乐领养3只猫,5只狗;小欢领养4只狗。
输出：{"yellow":[2,2,2,3],"blue":[5,5,4]}
"""


def get_api_key() -> str:
    key = os.environ.get("DEEPSEEK_API_KEY", "")
    if key:
        return key
    key = "sk-在这里填你的KEY"
    if key.startswith("sk-") and "在这里填" not in key:
        return key
    print("[错误] 未设置 API Key。请任选其一：")
    print("  1. 环境变量:  export DEEPSEEK_API_KEY=sk-xxxx")
    print("  2. 直接改本文件 get_api_key() 里的 key 变量")
    sys.exit(1)


def make_client():
    from openai import OpenAI
    return OpenAI(api_key=get_api_key(), base_url=BASE_URL)


def ask(client, messages, model, stream=True):
    """调用 DeepSeek，返回完整回复文本。不再边收边打印。"""
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=stream,
        temperature=0.0,
        max_tokens=1024,
    )
    if stream:
        full = ""
        for chunk in resp:
            if chunk.choices and chunk.choices[0].delta.content:
                full += chunk.choices[0].delta.content
        return full
    else:
        return resp.choices[0].message.content


def run_generator(gen_cmd="./TMSCQtest_arm.bin"):
    """运行题目生成器，返回打印出来的题目文本（自动剥掉 '题目:' 前缀）。

    生成器命令可用环境变量 QUESTION_GEN 覆盖。运行失败/无输出返回 None。
    """
    cmd = os.environ.get("QUESTION_GEN", gen_cmd)
    try:
        out = subprocess.check_output(cmd, shell=True, timeout=15)
    except Exception as e:
        print(f"[出题] 运行 {cmd} 失败：{e}", file=sys.stderr)
        return None
    q = out.decode('utf-8', errors='replace').strip()
    if not q:
        print("[出题] 生成器没有输出", file=sys.stderr)
        return None
    # 剥掉 '题目:' / '题目：' 前缀
    q = re.sub(r'^\s*题目\s*[:：]\s*', '', q)
    return q


def format_answer(reply: str, per_shop: int = 5, max_reasonable: int = 30) -> str:
    """
    从模型回复中提取最后一组 JSON：
    {"yellow":[...], "blue":[...]}
    然后在本地求和、向上取整，输出固定格式。
    """
    matches = re.findall(r'\{[^{}]*"yellow"[^{}]*\}', reply)
    if not matches:
        raise ValueError(f"模型未按要求输出 JSON，原始回复：{reply}")

    data = json.loads(matches[-1])
    yellow_list = data.get("yellow", [])
    blue_list = data.get("blue", [])

    try:
        yellow_total = sum(int(v) for v in yellow_list)
        blue_total = sum(int(v) for v in blue_list)
    except (TypeError, ValueError):
        raise ValueError(f"JSON 中数值格式不对：{data}")

    # 合理性校验：明显过大说明模型把"每店5只"之类混进去了
    if yellow_total > max_reasonable or blue_total > max_reasonable:
        raise ValueError(
            f"提取的数字明显偏大，可能把容器容量也提取了：{data} "
            f"(yellow_total={yellow_total}, blue_total={blue_total})"
        )
    if not yellow_list or not blue_list:
        raise ValueError(f"某个数组为空：{data}")

    x = math.ceil(yellow_total / per_shop)
    y = math.ceil(blue_total / per_shop)
    return f"{{color:yellow,num:{x}}},{{color:blue,num:{y}}}"


def push_answer_to_panel(answer: str):
    """把解题结果写入 ai_to_panel.py 监听的文件，由它发布到 ROS /task/* 面板。

    ai_to_panel.py 用 `--watch <文件>` 盯这个文件（内容稳定 300ms 后解析发布）。
    路径用环境变量 AI_ANSWER_FILE 指定，默认 /tmp/ai_answer.txt。
    写文件失败不影响答题（比如文件系统只读时）。
    """
    path = os.environ.get("AI_ANSWER_FILE", "/tmp/ai_answer.txt")
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(answer + "\n")
        print(f"[面板] 结果已写入 {path}")
        print(f"[面板] 由 ai_to_panel.py --watch {path} 发布到 coStudio 面板")
        return path
    except Exception as e:
        # 面板更新失败不影响答题流程
        print(f"[面板] 写入失败（忽略）：{e}")
        return None


def answer_to_targets(answer: str):
    """把 "{color:yellow,num:X},{color:blue,num:Y}" 转成 send_command 的 payload。

    返回 {"targets":[{"color":"yellow","num":X},...],"total":X+Y}；解析失败返回 None。
    """
    pairs = re.findall(
        r'color\s*:\s*(yellow|blue)\s*,\s*num\s*:\s*(\d+)', answer, re.IGNORECASE
    )
    if not pairs:
        print(f"[机械臂] 解析不到 color/num 对：{answer!r}", file=sys.stderr)
        return None
    targets = [{"color": c.lower(), "num": int(n)} for c, n in pairs]
    return {"targets": targets, "total": sum(t["num"] for t in targets)}


# 机械臂派单用的 ROS 客户端（惰性初始化一次，复用同一个 node/client）
_arm_node = None
_arm_cli = None


def _get_arm_client():
    """惰性初始化 rclpy 与 /send_command 客户端；ROS 不可用时返回 (None, None)。"""
    global _arm_node, _arm_cli
    if _arm_node is not None:
        return _arm_node, _arm_cli
    try:
        import rclpy
        from rclpy.node import Node
        from custom_msgs.srv import StrMsg
    except Exception as e:
        print(f"[机械臂] 无法导入 rclpy/custom_msgs（source 工作区了吗？）：{e}",
              file=sys.stderr)
        return None, None
    if not rclpy.ok():
        rclpy.init(args=[])
    _arm_node = Node('deepseek_arm_dispatch')
    _arm_cli = _arm_node.create_client(StrMsg, 'send_command')
    return _arm_node, _arm_cli


def push_answer_to_arm(answer: str):
    """把解题结果派给机械臂：调 /send_command 服务，按颜色抓取到放置区。

    connecter 节点（tools_demo）提供 /send_command (custom_msgs/srv/StrMsg)，
    request.data 是 {"targets":[{"color":"yellow","num":X},...],"total":N} 的 JSON。
    机械臂未就绪 / ROS 没 source 时只打印警告，不影响答题流程。

    先等面板把结果完整映射完（push_answer_to_panel 写文件后，ai_to_panel 会解析并
    发布到 /task/*），再让机械臂开工，确保「结果上板 → 机械臂开始」的顺序。
    等待时长用环境变量 PANEL_TO_ARM_DELAY（秒，默认 1.0）控制。
    """
    try:
        panel_delay = float(os.environ.get("PANEL_TO_ARM_DELAY", "1.0"))
    except ValueError:
        panel_delay = 1.0
    if panel_delay > 0:
        time.sleep(panel_delay)

    payload = answer_to_targets(answer)
    if payload is None:
        return None

    node, cli = _get_arm_client()
    if node is None or cli is None:
        return None

    import rclpy
    try:
        if not cli.service_is_ready():
            if not cli.wait_for_service(timeout_sec=3.0):
                print("[机械臂] /send_command 服务未就绪（connecter 没启动？），跳过派单",
                      file=sys.stderr)
                return None
        req = cli.srv_type.Request()
        req.data = json.dumps(payload, ensure_ascii=False)
        future = cli.call_async(req)
        # 同步等返回：抓取本身是阻塞动作，这里阻塞等是合理的
        while rclpy.ok() and not future.done():
            rclpy.spin_once(node, timeout_sec=0.1)
        res = future.result()
        print(f"[机械臂] 已派单 {payload} -> success={res.success}, message={res.message}")
        return res
    except Exception as e:
        print(f"[机械臂] 派单失败（忽略）：{e}", file=sys.stderr)
        return None


def interactive(client, model, system):
    print("DeepSeek 交互模式（输入 exit / quit 退出）")
    while True:
        try:
            line = input("\n你 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见")
            break
        if not line:
            continue
        if line.lower() in ("exit", "quit"):
            break

        # 每道题独立处理，不带历史，避免上一题的答案干扰
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": line},
        ]

        try:
            reply = ask(client, messages, model, stream=True)
            final = format_answer(reply)
            print(final)
            push_answer_to_panel(final)
            push_answer_to_arm(final)
        except Exception as e:
            print(f"\n[错误] {e}")
            continue


def auto_loop(client, model, system):
    """自动循环：出题 → 解题 → 上板，直到 Ctrl+C 手动退出。

    每轮跑一次题目生成器拿到题目，同一道题直接喂给 DeepSeek，题目与答案
    一一对应、不会错位。每轮间隔用环境变量 AI_LOOP_INTERVAL 控制（秒，默认 3）。
    """
    try:
        interval = float(os.environ.get("AI_LOOP_INTERVAL", "3"))
    except ValueError:
        interval = 3.0
    print(f"自动出题+解题循环（每轮间隔 {interval}s，Ctrl+C 手动退出）")
    last_q = None
    while True:
        try:
            q = run_generator()
            if not q:
                print("[出题] 失败，2 秒后重试...", file=sys.stderr)
                time.sleep(2)
                continue
            if q == last_q:
                # 生成器题库小、随机种子没变时会连抽同一道题，跳过重抽
                print(f"[跳过] 生成器又抽到同一道题，{interval}s 后再抽...")
                time.sleep(interval)
                continue
            last_q = q
            print("=" * 60)
            print(f"题目: {q}")
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": q},
            ]
            reply = ask(client, messages, model, stream=True)
            final = format_answer(reply)
            print(final)
            push_answer_to_panel(final)
            push_answer_to_arm(final)
            print("=" * 60)
            time.sleep(interval)
        except KeyboardInterrupt:
            print("\n手动退出")
            break
        except Exception as e:
            print(f"[错误] {e}")
            time.sleep(2)


def _read_text(path):
    """读文件文本并 strip；文件不存在或读失败返回 None。"""
    try:
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            return f.read().strip()
    except OSError:
        return None


def _clear_text(path):
    """清空文件内容（作为「这道题已消费」的信号），失败忽略。"""
    try:
        open(path, 'w', encoding='utf-8').close()
    except OSError:
        pass


def watch_question(client, model, system, path):
    """被动接题：题目文件里出现完整新题就解一次，解完清空文件，再等下一道。

    出题侧每次跑生成器把题写进 path：
        ./TMSCQtest_arm.bin > /tmp/question.txt
    本进程读到非空且稳定（连读两次内容一致）的题目就解一次、上板，然后把文件
    清空作为「已消费」信号。这样「跑一次生成器 = 解一次」，不多解、不漏解。

    前提：先启动本进程、再跑生成器；并等上一次「已解题」打印出来再跑下一次，
    避免生成器把上一道还没解的题覆盖掉。
    """
    print(f"监听题目文件 {path} ...（先启动本进程，再跑 ./TMSCQtest_arm.bin > {path}；Ctrl+C 退出）")
    while True:
        try:
            if not os.path.exists(path):
                time.sleep(0.3)
                continue
            # 稳定读：连读两次内容一致，才认为题目写完整了（防读到半截）
            r1 = _read_text(path)
            time.sleep(0.3)
            r2 = _read_text(path)
            if r1 is None or r2 is None or r1 != r2:
                time.sleep(0.3)
                continue
            q = re.sub(r'^\s*题目\s*[:：]\s*', '', r1).strip()
            if not q:
                time.sleep(0.3)
                continue
            print("=" * 60)
            print(f"题目: {q}")
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": q},
            ]
            try:
                reply = ask(client, messages, model, stream=True)
                final = format_answer(reply)
                print(final)
                push_answer_to_panel(final)
                push_answer_to_arm(final)
            except Exception as e:
                print(f"[错误] {e}")
            finally:
                _clear_text(path)   # 无论成败都清空，避免同一道题被反复解
            print("=" * 60)
            print("已解题，继续等待下一道题...")
        except KeyboardInterrupt:
            print("\n手动退出")
            break
        except Exception as e:
            print(f"[错误] {e}")
            time.sleep(1)


def main():
    parser = argparse.ArgumentParser(description="DeepSeek API 客户端")
    parser.add_argument("question", nargs="?", help="单次提问内容")
    parser.add_argument("-i", "--interactive", action="store_true", help="进入交互模式")
    parser.add_argument("--auto", action="store_true", help="自动：运行题目生成器出题，自动解题并上板")
    parser.add_argument("--watch-question", metavar="FILE",
                        help="监听题目文件：出现新题就解一次，解完继续等下一道")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"模型 ID（默认 {DEFAULT_MODEL}）")
    parser.add_argument("--system", default=DEFAULT_SYSTEM, help="系统提示词")
    parser.add_argument("--no-stream", action="store_true", help="关闭流式输出")
    args = parser.parse_args()

    client = make_client()

    if args.interactive:
        interactive(client, args.model, args.system)
    elif args.auto:
        auto_loop(client, args.model, args.system)
    elif args.watch_question:
        watch_question(client, args.model, args.system, args.watch_question)
    elif args.question:
        messages = [
            {"role": "system", "content": args.system},
            {"role": "user", "content": args.question},
        ]
        try:
            reply = ask(client, messages, args.model, stream=not args.no_stream)
            final = format_answer(reply)
            print(final)
            push_answer_to_panel(final)
            push_answer_to_arm(final)
        except Exception as e:
            print(f"[错误] {e}")
            sys.exit(1)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()