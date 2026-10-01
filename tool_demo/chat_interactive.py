import requests
import json
import sys

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

PROBLEM_TOPIC = "/task/problem"


class ProblemPublisher(Node):
    """把题目文本发到 coStudio「题目」面板（RawMessages 读 /task/problem）"""

    def __init__(self):
        super().__init__("problem_publisher")
        # transient_local：coStudio 后连上来也能拿到最新一条题目
        qos = rclpy.qos.QoSProfile(
            reliability=rclpy.qos.ReliabilityPolicy.RELIABLE,
            durability=rclpy.qos.DurabilityPolicy.TRANSIENT_LOCAL,
            depth=1,
        )
        self.pub = self.create_publisher(String, PROBLEM_TOPIC, qos)

    def publish(self, text):
        msg = String()
        msg.data = text
        self.pub.publish(msg)


def strip_problem_prefix(text):
    """去掉题目生成器带的「题目：」前缀，面板只显示题目本身"""
    t = text.strip()
    for prefix in ("题目：", "题目:"):
        if t.startswith(prefix):
            return t[len(prefix):].strip()
    return t

    def __init__(self, base_url="http://localhost:8080"):
        self.base_url = base_url
        self.chat_history = []
        self.system_prompt = "你将接收动态变化的应用题，题型规则：每种仓库固定存放5个物品，分为两类仓库，变量x代表黄色仓库数量，变量y代表蓝色仓库数量。先求解出题目中的x、y数值。输出严格遵循格式：{color:yellow,num:x的计算结果}，{color:blue,num:y的计算结果}。不要提出其他字眼，禁止输出任何分析、过程、多余文字，仅输出指定格式内容。"

    def send_message(self, user_input):
        # 构建消息历史
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(self.chat_history)
        messages.append({"role": "user", "content": user_input})

        data = {
            "model": "qwen2.5-coder",
            "messages": messages,
            "max_tokens": 512,
            "temperature": 0.7,
            "stream": False,
        }

        try:
            response = requests.post(
                f"{self.base_url}/v1/chat/completions",
                json=data,
                headers={"Content-Type": "application/json"},
            )

            if response.status_code == 200:
                result = response.json()
                assistant_reply = result["choices"][0]["message"]["content"]

                # 更新对话历史
                self.chat_history.append({"role": "user", "content": user_input})
                self.chat_history.append(
                    {"role": "assistant", "content": assistant_reply}
                )

                return assistant_reply
            else:
                return f"错误: {response.status_code}, {response.text}"

        except Exception as e:
            return f"连接错误: {str(e)}"

    def process_response(self, response, option):
        """处理回复的不同选项"""
        if option.upper() == "N":
            print("不处理，继续对话...")
            return response
        elif option.upper() == "R":
            print("默认处理...")
            # 这里可以添加默认处理逻辑
            # 发布StrMsg到服务 send_command
            processed = f"[已处理] {response}"
            return processed
        else:
            print("未知选项，使用默认处理...")
            return response


def main():
    rclpy.init()
    pub = ProblemPublisher()
    chat = LlamaChat()

    print("=== Llama 交互式聊天程序 ===")
    print("输入 'quit' 或 'exit' 退出程序")
    print("每次回复后可选择处理方式:")
    print("  N  - 不处理")
    print("  R  - 默认处理")
    print("-" * 40)

    while True:
        try:
            # 获取用户输入
            user_input = input("\n> ").strip()

            if user_input.lower() in ["quit", "exit"]:
                print("再见！")
                break

            if not user_input:
                continue
                
            # 发题目给 llama 的同时，把题目上屏
            pub.publish(strip_problem_prefix(user_input))

            # 发送消息并获取回复
            print("正在思考...")
            response = chat.send_message(user_input)
            print(f"\n助手: {response}")

            # 询问处理选项
            while True:
                option = input("\n选择处理方式 [N/R]: ").strip()
                if option.upper() in ["N", "R"]:
                    processed_response = chat.process_response(response, option)
                    if processed_response != response:
                        print(f"处理结果: {processed_response}")
                    break
                else:
                    print("请输入有效选项: N, R")

        except KeyboardInterrupt:
            print("\n\n程序被中断,再见！")
            break
        except Exception as e:
            print(f"发生错误: {str(e)}")


if __name__ == "__main__":
    main()
