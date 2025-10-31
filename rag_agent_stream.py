from dataclasses import dataclass
from langchain.tools import tool
import os
import requests
import json
import sys
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver

# 配置参数（保持不变）
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "sk-8c526fe03364421fbf8b4c47cf3e25c7")
STREAM_API_URL = "http://192.168.19.166:8000/api/stream-text"

os.environ["DASHSCOPE_API_KEY"] = DASHSCOPE_API_KEY

@dataclass
class Context:
    user_id: str

# 工具函数（保持不变）
@tool
def get_legal_advice(case_summary: str) -> str:
    """调用本地知识库流式接口，基于案件摘要生成法律建议"""
    try:
        if "案件摘要：" in case_summary:
            clean_summary = case_summary.split("案件摘要：")[1].strip()
        else:
            clean_summary = case_summary
            
        
        payload = {"query": clean_summary}
        headers = {"Content-Type": "application/json"}
        
        response = requests.post(
            STREAM_API_URL,
            data=json.dumps(payload),
            headers=headers,
            stream=True,
            timeout=30
        )
        
        response.raise_for_status()
        
        full_text = ""
        
        for line in response.iter_lines(chunk_size=1):
            if line:
                line_str = line.decode('utf-8')
                if line_str.startswith('data: '):
                    data_str = line_str[6:]
                    
                    if data_str == '[DONE]':
                        break
                    
                    try:
                        data = json.loads(data_str)
                        if 'content' in data:
                            content = data['content']
                            print(content, end='', flush=True)
                            full_text += content
                    except json.JSONDecodeError:
                        print(data_str, end='', flush=True)
                        full_text += data_str
        
        print()
        return full_text
        
    except Exception as e:
        error_msg = f"调用知识库失败: {str(e)}"
        print(f"[DEBUG] {error_msg}")
        return error_msg

# 初始化主agent模型
main_model = init_chat_model(
    model="qwen-plus-latest",
    model_provider="openai",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=DASHSCOPE_API_KEY,
    temperature=0.5
)

# 修改系统提示词：告诉Agent生成摘要后不要自行提供建议
UPDATED_SYSTEM_PROMPT = """
你是一名专业的AI律师电话助手，擅长处理各种常见的法律咨询，例如：借贷纠纷、劳动争议、婚姻家庭、交通事故、合同纠纷、房屋租赁、侵权纠纷等。
你的任务是通过多轮对话，帮用户厘清案件事实，最后输出案件摘要。
【核心目标】
1. 逐步追问，明确案件关键信息；
2. 当信息清晰后，自动生成案件摘要；
3. 如果用户提出新问题，先回答再回到当前案件；
4. 案件摘要用于律师后续处理，不含法律建议。
【对话阶段说明】
### 一、追问阶段
- 每次只能追问 **1~2个问题**。
- 追问时要针对“事实”而非“法律判断”。
- 追问重点：
  - 事件时间、地点
  - 涉及人员及关系（当事人是谁）
  - 争议内容（如金额、劳动关系、伤害、合同等）
  - 证据情况（是否有合同、录音、聊天记录、目击证人等）
  - 维权情况（是否沟通过、是否报警、是否起诉等）
- 用户若回答清楚，不要重复提问。
- 若用户回答模糊，可温和引导进一步说明。
### 二、处理新问题
- 若用户在追问过程中提出了新的法律问题：
  - 先简要回答该问题；
  - 再继续当前案件的追问或总结；
  - 若当前案件已结束，则启动新案件的追问流程。
### 三、结束与摘要
- 当案件事实清晰（至少包括时间、当事人关系、争议点、证据情况）时，
  主动生成案件摘要，格式为：
  **{案件摘要：……。}**
- 案件摘要应当简明、客观。
- 案件摘要示例：
  {案件摘要：2023年3月，用户在公司工作期间因加班工资未结清，已多次向公司反映但未获解决，有聊天记录为证。}
- 如果用户主动要求“总结”“案件摘要”“案情”，立即生成案件摘要。
【语气与风格】
- 对用户保持尊重与安抚；
- 不使用表情或符号；
- 回答简短、自然、有条理。
【输出类型】
你每次的回复只能属于以下三类之一：
1. **追问问题**：继续了解案件细节；
2. **回答问题并追问**：用户提出新问题时先简答再继续追问；
3. **案件摘要**：当案件信息充分时，用格式 {案件摘要：……。} 输出完整摘要。
【额外要求】
- 每个案件对话控制在2~5轮之间；
- 若用户继续追问其他案件，重新开始新的案件分析流程；
- 不要输出除中文外的文字或符号；
"""

# 创建检查点保存器
checkpointer = InMemorySaver()

# 创建主agent（不集成工具，避免自动调用）
main_agent = create_agent(
    model=main_model,
    tools=[],  # 不集成工具，由外部手动调用
    system_prompt=UPDATED_SYSTEM_PROMPT,
    checkpointer=checkpointer
)

def run_legal_consultation():
    """运行法律咨询对话"""
    print("=== 法律咨询助手 ===")
    print("您好！我是AI律师助手，帮您厘清案件事实并提供法律建议。")
    print("输入'退出'结束对话\n")
    
    config = {"configurable": {"thread_id": "main_consultation"}}
    last_case_summary = ""
    last_advice = ""
    consultation_complete = False  # 标记咨询是否完成
    
    while True:
        user_input = input("\n您: ").strip()
        
        if user_input.lower() in ['退出', 'quit', 'exit']:
            print("感谢使用法律咨询助手！")
            break

        try:
            # 使用流式输出调用主agent
            print("\n助手: ", end="", flush=True)
            assistant_response = ""
            
            # 使用 stream_mode="messages" 来流式输出LLM tokens
            for token, metadata in main_agent.stream(
                {"messages": [{"role": "user", "content": user_input}]},
                config=config,
                stream_mode="messages"
            ):
                # 只处理来自model节点的文本内容
                if metadata.get('langgraph_node') == 'model':
                    for content_block in token.content_blocks:
                        if content_block.get('type') == 'text':
                            text = content_block.get('text', '')
                            print(text, end='', flush=True)
                            assistant_response += text
            
            print()  # 换行
            
            # 检测并保存案件摘要，然后自动调用工具
            if "案件摘要" in assistant_response and not consultation_complete:
                last_case_summary = assistant_response              
                # 自动调用工具获取建议
                try:
                    advice = get_legal_advice.invoke(last_case_summary)
                    last_advice = advice
                    consultation_complete = True
                    
                    
                    # 将法律建议添加到Agent的对话历史中（不显示给用户）
                    _ = main_agent.invoke(
                        {"messages": [
                            {"role": "assistant", "content": f"基于您的案件情况，我为您提供了以下专业的法律建议：\n\n{advice}\n\n如果您对任何建议有疑问，可以随时问我。"}
                        ]},
                        config=config
                    )
                    
                except Exception as e:
                    print(f"[DEBUG] 自动调用工具失败: {e}")
                    print("\n助手: 抱歉，暂时无法获取详细法律建议。")
                    
        except Exception as e:
            print(f"[DEBUG] Agent调用失败: {e}")
            print("\n助手: 抱歉，系统出现了一些问题，请重新描述您的问题。")

if __name__ == "__main__":
    import time
    run_legal_consultation()
