# Voice Agent Flow - 基于状态管理的语音智能体交接系统

Question: Do we need precise control over task oriented conversational AI?
https://mp.weixin.qq.com/s/wAnHOZoH6645vQ3gIp8xDg

Why pydantic: I like the output design.   
For tool invoking, the strategy is : Run LLM again.  
For string output or structured output: Stop generation and return the output.
This is perfect for Conversational AI, which need both agent state management, entity extraction, tool invoking and answer generation.

## 项目概述

Voice Agent Flow 是一个创新的多智能体协作框架，专门为语音交互场景设计。该系统通过 Pydantic 结构化输出和智能转移机制，实现了智能体之间的无缝交接，有效解决了传统语音智能体系统中的状态管理、信息传递和流程控制问题。

## 核心特性

### 1. 结构化状态管理
- **Pydantic Schema驱动**: 每个智能体的任务通过 Pydantic 模型定义，确保信息收集的完整性和准确性
- **强类型约束**: 利用 Python 类型系统，在编译时就能发现潜在的数据结构问题
- **自动验证**: 输入数据自动验证，减少运行时错误

### 2. 智能化转移机制
- **基于状态的转移**: 每个任务类都包含 `transfer()` 方法，根据收集到的信息智能决定下一个智能体
- **条件性路由**: 支持复杂的业务逻辑判断，如时间可用性检查、资格预审等
- **灵活的流程控制**: 可以实现循环、跳转、提前结束等多种流程模式

### 3. 会话连续性保障
- **统一消息历史**: 所有智能体共享同一个消息历史记录，确保上下文连贯
- **信息累积**: 收集到的结构化信息会被保存，供后续智能体使用
- **无缝交接**: 用户感受不到智能体切换，对话体验自然流畅

## 为什么特别适合语音智能体系统？

### 1. 解决语音交互的特殊挑战

#### 🎯 **精准信息提取**
语音交互中，用户的表达往往不够精确，存在口语化、省略、歧义等问题。传统方法难以准确提取和验证关键信息。

```python
class PartySizeResult(BaseModel):
    size: int = Field(..., description='预订聚会的人数')
    
    def transfer(self) -> str:
        print("转移到时间收集器")
        return 'time_collector'
```

**优势**：
- Pydantic 的类型验证确保数据正确性
- 结构化输出避免信息丢失或误解
- 智能体专注于特定信息收集任务，提高成功率

#### 🔄 **动态流程适配**
语音对话的流程往往不是线性的，需要根据用户回答动态调整对话策略。

```python
class TimeResult(BaseModel):
    time: str = Field(..., description='预订时间，格式：YYYY-MM-DD HH')
    
    def transfer(self) -> str:
        if self.check_availability(self.time):
            return 'end'  # 时间可用，流程结束
        else:
            return 'time_collector'  # 时间不可用，重新收集
```

**优势**：
- 基于实际数据的智能路由决策
- 支持复杂的业务逻辑判断
- 避免死循环和无效对话

#### 📞 **电话场景优化**
电话客服场景中，对话需要高效、目标明确，同时要处理用户的偏离话题和不确定回答。

```python
instruction = """
你是汽车金融公司的客服代表，通过电话收集客户信息。
对于是否问题，如果客户没有明确拒绝，应该假设客户同意。
注意你是多智能体系统的一部分，不要添加额外的解释、问候或结束语。
"""
```

**优势**：
- 每个智能体专注单一任务，提高对话效率
- 减少用户等待时间和重复解释
- 智能处理模糊回答和偏离话题

### 2. 系统架构优势

#### 🏗️ **模块化设计**
```python
@dataclass
class AgentNode:
    name: str
    model: OpenAIChatModel
    instruction: str
    example: str
    task_cls: BaseModel  # 关键：每个节点都有明确的任务定义
```

**优势**：
- 每个智能体职责单一，便于调试和优化
- 可以独立测试和改进每个对话环节
- 易于添加新的业务流程或修改现有逻辑

#### 🔗 **状态持久化**
```python
class AgentRunner:
    def __init__(self):
        self.all_messages = []        # 会话历史
        self.collected_information = [] # 收集的结构化信息
        self.current_agent = None     # 当前活跃智能体
```

**优势**：
- 完整保存对话上下文，支持复杂业务场景
- 结构化信息可用于后续业务处理
- 支持对话中断和恢复

#### ⚡ **高效执行**
```python
def run(self, input_text: str):
    res = self.current_agent.run_sync(input_text, message_history=self.all_messages)
    output = res.output
    
    if isinstance(output, BaseModel):
        # 收集到结构化信息，自动转移
        target_agent = output.transfer()
        self.current_agent = self.get_agent(target_agent)
```

**优势**：
- 自动状态转移，减少手动编程复杂度
- 类型安全的数据传递
- 支持同步和异步执行模式

## 实际应用场景

### 1. 餐厅预订系统
```python
# 人数收集 → 时间收集 → 可用性验证 → 预订确认
agents = {
    "party_size_collector": AgentNode(...),
    "time_collector": AgentNode(...),
}
```

### 2. 汽车金融客服
```python
# 客户确认 → 需求询问 → 资质审核 → 业务办理
agents = {
    "customer_name_inquiry": AgentNode(...),
    "financial_support_inquiry": AgentNode(...),
    "vehicle_payment_status": AgentNode(...),
}
```

## 技术栈

- **AI框架**: PydanticAI - 提供类型安全的AI应用开发
- **模型支持**: OpenAI GPT系列、Azure OpenAI
- **数据验证**: Pydantic - 确保结构化输出的正确性
- **语言支持**: Python 3.8+

## 快速开始

### 安装依赖

```bash
pip install pydantic pydantic-ai openai
```

### 定义任务类

```python
from pydantic import BaseModel, Field

class CustomerInfo(BaseModel):
    name: str = Field(..., description='客户姓名')
    phone: str = Field(..., description='联系电话')
    
    def transfer(self) -> str:
        if self.validate_info():
            return 'next_agent'
        return 'current_agent'  # 信息不完整，继续收集
```

### 创建智能体

```python
from voice_agent_flow.node import AgentNode
from voice_agent_flow.runner import AgentRunner

agent = AgentNode(
    name="info_collector",
    model=your_model,
    instruction="收集客户基本信息...",
    example="请提供您的姓名和电话号码",
    task_cls=CustomerInfo
)

runner = AgentRunner(
    agents={"info_collector": agent},
    entry_agent_name="info_collector"
)
```

### 运行对话

```python
response = runner.run("你好，我想咨询业务")
print(response)
```

## 项目结构

```
voice_agent_flow/
├── __init__.py
├── node.py          # 智能体节点定义
├── runner.py        # 运行器和状态管理
├── load_env.py      # 环境配置
└── llms/
    ├── __init__.py
    ├── openai_provider.py     # OpenAI 提供者
    └── pydantic_provider.py   # PydanticAI 集成

applications/
├── resturant_reservation/    # 餐厅预订示例
│   ├── main.py
│   ├── task_cls.py
│   └── test.ipynb
└── auto_finance/            # 汽车金融示例
    ├── main.py
    └── task_cls.py
```

## 贡献

欢迎提交 Issue 和 Pull Request 来完善这个项目。

## 许可证

MIT License

---

**Voice Agent Flow - 让语音智能体协作更简单、更可靠！**