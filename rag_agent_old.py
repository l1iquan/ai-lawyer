from dataclasses import dataclass
from langchain.tools import tool
import os
import requests
import json
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver

# 配置参数
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "sk-8c526fe03364421fbf8b4c47cf3e25c7")
MODEL_NAME = os.getenv("MODEL_NAME", "qwen-flash")
API_BASE_URL = os.getenv("API_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
LOCAL_KB_URL = "http://169.254.96.144:8000/qa"  # 你的本地接口

os.environ["DASHSCOPE_API_KEY"] = DASHSCOPE_API_KEY

@dataclass
class Context:
    user_id: str

# 改进的工具函数，添加详细调试信息
@tool
def get_legal_advice(case_summary: str) -> str:
    """调用本地知识库接口，基于案件摘要生成法律建议"""
    try:
        print(f"[DEBUG] 尝试调用本地接口: {LOCAL_KB_URL}")
        
        # 提取纯文本案件摘要（去掉"案件摘要："前缀）
        if "案件摘要：" in case_summary:
            clean_summary = case_summary.split("案件摘要：")[1].strip()
        else:
            clean_summary = case_summary
            
        print(f"[DEBUG] 发送的查询内容: {clean_summary}")
        
        # 调用本地知识库接口
        response = requests.post(
            LOCAL_KB_URL,
            json={"query": clean_summary},
            timeout=10
        )
        
        print(f"[DEBUG] 接口响应状态码: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"[DEBUG] 接口返回完整数据: {json.dumps(result, ensure_ascii=False)}")
            answer = result.get('answer', '未找到相关法律建议')
            return answer
        else:
            error_msg = f"知识库接口错误: {response.status_code} - {response.text}"
            print(f"[DEBUG] {error_msg}")
            return error_msg
            
    except requests.exceptions.ConnectionError as e:
        error_msg = f"无法连接到本地知识库接口: {str(e)}"
        print(f"[DEBUG] {error_msg}")
        return error_msg
    except requests.exceptions.Timeout as e:
        error_msg = f"连接本地知识库超时: {str(e)}"
        print(f"[DEBUG] {error_msg}")
        return error_msg
    except Exception as e:
        error_msg = f"调用知识库失败: {str(e)}"
        print(f"[DEBUG] {error_msg}")
        return error_msg

# 初始化主agent模型
main_model = init_chat_model(
    model="qwen-plus-latest",
    model_provider="openai",
    base_url=API_BASE_URL,
    api_key=DASHSCOPE_API_KEY,
    temperature=0.5
)

# 简化系统提示词，让Agent更专注于追问和生成摘要
SIMPLIFIED_SYSTEM_PROMPT = """
你是一名专业的AI律师助手，负责通过对话了解用户的法律问题并生成案件摘要。

你的工作流程：
1. 通过2-5轮对话了解案件关键信息
2. 当信息充分时生成案件摘要，格式为：{案件摘要：具体内容}
3. 生成摘要后，系统会自动调用法律建议工具

追问重点：
- 事件时间、地点、涉及人员
- 争议内容（金额、劳动关系、伤害等）
- 证据情况（合同、录音、聊天记录等）
- 维权情况（是否沟通过、报警等）

沟通风格：生活化、口语化，亲切耐心，不使用表情符号。

注意：每次只问1-2个问题，信息充分后立即生成案件摘要。
"""

# 创建检查点保存器
checkpointer = InMemorySaver()

# 创建主agent（集成建议工具）
main_agent = create_agent(
    model=main_model,
    tools=[get_legal_advice],
    system_prompt=SIMPLIFIED_SYSTEM_PROMPT,
    checkpointer=checkpointer
)

def run_legal_consultation():
    """运行法律咨询对话"""
    print("=== 法律咨询助手 ===")
    print("您好！我是AI律师助手，帮您厘清案件事实并提供法律建议。")
    print("输入'退出'结束对话\n")
    
    config = {"configurable": {"thread_id": "main_consultation"}}
    last_case_summary = ""  # 保存上次的案件摘要
    
    while True:
        user_input = input("\n您: ").strip()
        
        if user_input.lower() in ['退出', 'quit', 'exit']:
            print("感谢使用法律咨询助手！")
            break
        
        # 处理工具调用请求
        if any(phrase in user_input for phrase in ['调用工具', '获取建议', '法律建议', '建议']):
            if last_case_summary:
                print("[DEBUG] 用户要求调用工具获取法律建议")
                try:
                    advice = get_legal_advice.invoke(last_case_summary)
                    print(f"\n助手: {advice}")
                    continue
                except Exception as e:
                    print(f"[DEBUG] 工具调用异常: {e}")
                    print("\n助手: 抱歉，获取法律建议时出现错误。")
                    continue
            else:
                print("\n助手: 请先描述您的问题，我会帮您分析并生成案件摘要。")
                continue
        
        # 处理简单回复
        if any(word in user_input for word in ['谢谢', '感谢', '好的', '知道了', '牛逼']):
            print("\n助手: 不客气，如果还有其他问题可以继续咨询。")
            continue
        
        try:
            # 调用主agent
            response = main_agent.invoke(
                {"messages": [{"role": "user", "content": user_input}]},
                config=config
            )
            
            # 显示助手回复
            assistant_response = response["messages"][-1].content
            print(f"\n助手: {assistant_response}")
            
            # 检测并保存案件摘要，然后自动调用工具
            if "案件摘要" in assistant_response:
                last_case_summary = assistant_response
                print("[DEBUG] 检测到案件摘要，自动调用工具获取建议")
                
                # 自动调用工具获取建议
                try:
                    advice = get_legal_advice.invoke(last_case_summary)
                    print(f"\n法律建议：{advice}")
                except Exception as e:
                    print(f"[DEBUG] 自动调用工具失败: {e}")
                    print("\n助手: 抱歉，暂时无法获取详细法律建议。")
                    
        except Exception as e:
            print(f"[DEBUG] Agent调用失败: {e}")
            # 尝试重置对话状态
            config = {"configurable": {"thread_id": f"main_consultation_{id(time.time())}"}}
            print("\n助手: 抱歉，系统出现了一些问题，请重新描述您的问题。")

if __name__ == "__main__":
    import time
    run_legal_consultation()