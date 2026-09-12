#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DeepSeek API 客户端 —— 可直接在 RDK X5 (Ubuntu aarch64) 上运行。

用法:
    # 单次提问
    python3 deepseek_client.py "介绍一下RDK X5"

    # 交互式聊天（多轮对话，保留上下文）
    python3 deepseek_client.py -i

    # 指定模型 / 自定义系统提示词
    python3 deepseek_client.py "你好" --model deepseek-chat --system "你是机器人助手"

    # 环境变量里放 Key（推荐，避免写死在代码里）
    export DEEPSEEK_API_KEY=sk-xxxx
    python3 deepseek_client.py -i

依赖:
    pip3 install openai
"""

import os
import sys
import re
import json
import math
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
    """把解题结果写进 coStudio 布局 JSON（改面板标签，副作用，失败不影响答题）。

    面板脚本/布局 JSON 的目录用环境变量 COSTUDIO_LAYOUT_DIR 指定，
    默认 Windows 上的 D:\\比赛\\任务挑战赛；目录/脚本/模板任一缺失就静默跳过
    （比如在 RDK 上跑、没放布局文件时）。
    """
    panel_dir = os.environ.get("COSTUDIO_LAYOUT_DIR", r"D:\比赛\任务挑战赛")
    try:
        if not os.path.isdir(panel_dir):
            return None
        sys.path.insert(0, panel_dir)
        import gen_layout
        template = os.path.join(panel_dir, "2026", "可视化软件", "可视化布局参考.json")
        output = os.path.join(panel_dir, "2026", "可视化软件", "可视化布局.json")
        labels = gen_layout.update_layout(answer, template, output)
        if labels:
            print(f"[面板] 黄色数量={labels['黄色数量']} 蓝色数量={labels['蓝色数量']}"
                  f" 先抓={labels['先抓取']} 后抓={labels['后抓取']}")
            print(f"[面板] 已写入 → {output}（在 coStudio 里导入这个文件）")
        return labels
    except Exception as e:
        # 面板更新失败不影响答题流程
        print(f"[面板] 更新失败（忽略）：{e}")
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
        except Exception as e:
            print(f"\n[错误] {e}")
            continue


def main():
    parser = argparse.ArgumentParser(description="DeepSeek API 客户端")
    parser.add_argument("question", nargs="?", help="单次提问内容")
    parser.add_argument("-i", "--interactive", action="store_true", help="进入交互模式")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"模型 ID（默认 {DEFAULT_MODEL}）")
    parser.add_argument("--system", default=DEFAULT_SYSTEM, help="系统提示词")
    parser.add_argument("--no-stream", action="store_true", help="关闭流式输出")
    args = parser.parse_args()

    client = make_client()

    if args.interactive:
        interactive(client, args.model, args.system)
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
        except Exception as e:
            print(f"[错误] {e}")
            sys.exit(1)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()