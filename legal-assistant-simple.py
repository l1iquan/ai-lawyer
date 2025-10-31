from dataclasses import dataclass
from langchain.tools import tool
import os
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver

# 配置千问大模型参数
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "sk-8c526fe03364421fbf8b4c47cf3e25c7")
MODEL_NAME = os.getenv("MODEL_NAME", "qwen-flash")
API_BASE_URL = os.getenv("API_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")

# 设置环境变量
os.environ["DASHSCOPE_API_KEY"] = DASHSCOPE_API_KEY

@dataclass
class Context:
    """Custom runtime context schema."""
    user_id: str

# 初始化追问模型（千问）
question_model = init_chat_model(
    model="qwen-plus-latest",
    model_provider="openai",
    base_url=API_BASE_URL,
    api_key=DASHSCOPE_API_KEY,
    temperature=0.4
)

# 初始化建议模型（可以换其他模型）
advice_model = init_chat_model(
    model="qwen-plus-latest",  # 这里可以换成其他模型
    model_provider="openai",
    base_url=API_BASE_URL,
    api_key=DASHSCOPE_API_KEY,
    temperature=0.7
)

SYSTEM_PROMPT = """
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
- 案件摘要应当简明、客观、无建议性语言，不包含“建议起诉”“建议报警”等字样。
- 案件摘要示例：
  {案件摘要：2023年3月，用户在公司工作期间因加班工资未结清，已多次向公司反映但未获解决，有聊天记录为证。}
- 如果用户主动要求“总结”“案件摘要”“案情”，立即生成案件摘要。
【语气与风格】
- 使用生活化、口语化的表达；
- 语气亲切、有耐心；
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
- 不要输出“请咨询律师”“建议起诉”等法律行动建议；
- 不要输出除中文外的文字或符号；
- 重点是“帮用户把事实说清楚”，而不是“帮用户定性案件”。
"""

ADVICE_SYSTEM_PROMPT = """你是一名专业的AI律师助手，以下内容为案件摘要.请遵循以下规范提供法律咨询服务：
1. 专业规范：
- 保持专业、客观、中立的态度
- 不提供具体的法律代理服务
- 不承诺案件结果
- 涉及紧急情况时建议立即报警
- 复杂案件建议咨询专业律师
2. 沟通风格：
- 使用生活化、口语化的语言
- 保持亲切友好的态度
3. 回答要求：
- 提供具体的行动指引
- 避免使用过于专业的法律术语
- 简短有效具有实用性
注意：不要回复表情以及符号如*等，只回复文字就可以。也不要问还要其他帮助吗。"""

@dataclass
class ResponseFormat:
    """响应格式"""
    response: str
    case_summary: str | None = None

@dataclass
class AdviceResponseFormat:
    """建议响应格式"""
    response: str

# 创建检查点保存器
checkpointer = InMemorySaver()

# 创建追问agent
question_agent = create_agent(
    model=question_model,
    tools=[],
    system_prompt=SYSTEM_PROMPT,
    response_format=ResponseFormat,
    checkpointer=checkpointer
)

# 创建建议agent
advice_agent = create_agent(
    model=advice_model,
    tools=[],
    system_prompt=ADVICE_SYSTEM_PROMPT,
    response_format=AdviceResponseFormat,
    checkpointer=checkpointer
)

def run_legal_consultation():
    """运行法律咨询对话"""
    print("=== 法律咨询助手 ===")
    print("您好！我是AI律师助手，帮您厘清案件事实。")
    print("输入'退出'结束对话\n")
    
    config = {"configurable": {"thread_id": "1"}}
    advice_given = False  # 标记是否已给出法律建议
    
    while True:
        # 获取用户输入
        user_input = input("\n您: ").strip()
        
        if user_input.lower() in ['退出', 'quit', 'exit']:
            print("感谢使用法律咨询助手！")
            break
        
        # 调用追问agent，checkpointer会自动管理对话历史
        try:
            response = question_agent.invoke(
                {"messages": [{"role": "user", "content": user_input}]},
                config=config,
                context=Context(user_id="1")
            )
            
            # 显示助手回复
            assistant_response = response['structured_response'].response
            print(f"\n助手: {assistant_response}")
            
            # 简单判断：是否包含"案件摘要"
            if "案件摘要" in assistant_response:
                print("[DEBUG] 检测到案件摘要，调用建议agent")
                # 调用建议agent生成法律建议
                try:
                    advice_response = advice_agent.invoke(
                        {"messages": [{"role": "user", "content": f"请为以下案件提供建议：{assistant_response}"}]},
                        config = {"configurable": {"thread_id": "advice_thread"}},
                        context=Context(user_id="1")
                    )
                    advice_content = advice_response['structured_response'].response
                    print(f"\n法律建议：{advice_content}")
                    advice_given = True
                except Exception as e:
                    print(f"[DEBUG] 建议agent调用失败: {e}")
                    # 直接调用模型作为备选方案
                    direct_response = advice_model.invoke([{"role": "user", "content": f"请为以下案件提供法律建议：{assistant_response}"}])
                    print(f"\n法律建议：{direct_response.content}")
                    advice_given = True
        except Exception as e:
            print(f"[DEBUG] 追问agent调用失败: {e}")
            print("\n助手: 抱歉，系统出现了一些问题，请重新描述您的问题。")

if __name__ == "__main__":
    run_legal_consultation()
