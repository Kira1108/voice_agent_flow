from typing import Optional

from agentic_data.llms import pydantic_openai_like_async
from pydantic import BaseModel, Field

from voice_agent_flow.agents import AgentSession
from voice_agent_flow.agents.agent_node import AgentNode, DoHangUp, HangUpNode
from voice_agent_flow.agents.multi_agent_runner import MultiAgentRunner
from voice_agent_flow.llms import create_pydantic_azure_openai
from voice_agent_flow.tools import create_phone_num_check_tool

class CustomerName(BaseModel):
    
    """Greet the customer with the customer's name in the greeting message. If not explicitly told that the name is wrong or wrong number dialed, assume the name is confirmed.
    Example: 
        Customer Service Representative: "喂，您好，请问是xxx吗?"
        Customer: "/不是/打错了/我不是" -> -> create CustomerName(customer_name=None, name_checked=False)
        Customer: "用户无法接听, 请在语音信箱留言" -> create CustomerName(customer_name = None, name_checked = False)
        Customer: "您好，我是xxx的智能助手，..." -> create CustomerName(customer_name = 'xxx', name_checked = False)
    """
    
    customer_name: Optional[str] = Field(None, description = 'The name of the customer')
    name_checked: Optional[bool] = Field(True, description = 'Whether the customer name is confirmed(not explicitly rejected)')
    
    def transfer(self) -> str:
        
        if self.name_checked is None or self.name_checked:
            print("Transferring to FinancialSupportStatus")
            return 'financial_support_inquiry'
        
        else:
            print("Transferring to HangUp")
            return 'hangup'
    
    

class FinancialSupportStatus(BaseModel):
    
    """Whether the customer need financial support. True is the customer do not refuse/reject financial support.
    As long the the customer asking product details or show insterest, create FinancialSupportInquiry(require_financial_support=True).
    Explicit rejection or refusal should create FinancialSupportStatus(require_financial_support=False).
    Other case, you should continue to ask the customer for clarification.
    """
    require_financial_support: bool = Field(..., description = 'Whether the customer need financial support')
    
    def transfer(self) -> str:
        
        if self.require_financial_support:
            print("Transferring to VehicleNotUnderRepayment")
            return 'vehicle_payment_status'
        
        else:
            print("Transferring to HangUp")
            return 'hangup'
        
    
class PaymentMethod(BaseModel):
    """
    Whether the vehicle is under a auto loan or fully paid off.
    It the vehicle is not bought with a loan, is_not_under_repayment = True.
    If the vehicle is bought with a loan, and there are ongoing installments currently, is_not_under_repayment = False.
    """
    
    is_not_under_repayment: bool = Field(..., description = 'Whether the vehicle is not under repayment')
    
    def transfer(self) -> str:
        
        if self.is_not_under_repayment:
            print("Transferring to VehicleLiscenceUnderControl")
            return 'vehicle_liscence_under_control'
        
        else:
            print("Transferring to HangUp")
            return 'hangup'
        
    
class VehicleLiscenceUnderControl(BaseModel):
    
    """Whether the vehicle liscence(Chinese机动车行驶证， 绿本， 大本) is under the customer's control.
    If the vehicle liscense is under the customer's control, create VehicleLiscenceUnderControl(green_book_available=True).
    If the vehicle liscense is under the customer's spouse or family member's control, create VehicleLiscenceUnderControl(green_book_available=True).
    If the vehicle liscense ir under his/her company's control, create VehicleLiscenceUnderControl(green_book_available=False).
    """
    
    green_book_available: bool = Field(..., description = 'Whether the vehicle liscence is under the customer\'s control')
    
    def transfer(self) -> str:
        
        if self.green_book_available:
            print("Transferring to wechat_account_confirm")
            return "wechat_account_confirm"
        
        else:
            print("Transferring to HangUp")
            return 'hangup'


class WeChatAccount(BaseModel):
    """Wechat Account confirmed by user, only accept phone number based wechat account. 11 digits number string"""
    
    wechat_account:str = Field(..., description = 'The WeChat account provided by the customer, should be 11 digits phone number based wechat account')
    
    def transfer(self) -> str:
        return "wechat_add_request"


def add_wechat_account(account:str):
    """Add customer wechat account, account should be a 11 digits phone number string, confirmed by customer as well."""
    return f"Sent a wechat add request to customer [Success]. [not guaranteed to be accepted by customer], wechat account: {account}."


class WeChatRequestReceived(BaseModel):
    
    """The customer receive the wechat add request. Any positive response like '看到了', '收到了', or other positive response from the customer set received = True"""
    received: bool = Field(..., description = 'Whether the customer receive the wechat add request')
    
    def transfer(self) -> str:
        if self.received:
            print("Transferring to WeChatAccpeted")
            return "wechat_guide"
        else:             
            print("Transferring to HangUp")
            return "hangup" 


class WeChatAccpeted(BaseModel):
    
    """The customer accepted the wechat add request. Any positive response like '接受了', '同意了', or other positive response from the customer set accepted = True"""
    accepted: bool = Field(..., description = 'Whether the customer accepted the wechat add request')
    
    def transfer(self) -> str:
        return "hangup"
    
INSTRUCTION = """
## SYSTEM ROLE & OBJECTIVE
You are a customer service representative for an auto finance company (易鑫集团). Your task is to conduct telephone conversations to collect customer information according to predefined schemas. You speak in Chinese and focus on efficiently gathering required data.

## CORE PRINCIPLES

### Conversation Flow
- Do NOT add greetings, explanations, or closing statements—you are part of a multi-agent system
- Only generate structured output after collecting all required schema fields
- Before that, engage in natural conversation with the customer
- Create schema output as soon as possible to move the conversation forward

### Information Collection
- Output minimal text per response to efficiently collect information
- In each step, you can: (1) collect information for the current step, (2) handle customer questions briefly
- Do NOT ask questions beyond what is specified in the step instruction
- Only ask for information defined in the current step's schema

### Re-ask Policy
- If the customer does NOT answer the current step question, re-ask the same question.
- Each re-ask must use different phrasing and word choice, while keeping the meaning unchanged.
- Re-asks must be brief and simple.
- Do NOT introduce new questions or new schema fields when re-asking.
- After re-asking, return immediately to collecting the current step field.


### Yes/No Question Policy
- If a customer's response is NOT explicitly rejective/refusal, assume agreement
- Create the schema with corresponding fields filled immediately
- Ambiguous responses like "嗯", "呃", "哪里", "你说" = confirmation

## RESPONSE STYLE
1. **Colloquial & Fluent**: Enrich conversations with realistic examples
2. **Action-Oriented**: Prioritize schema creation to advance the flow
3. **Responsive**: Answer customer questions first, then return to step objective
4. **Concise**: Use short sentences, especially when guiding multi-step procedures

## STEP-BY-STEP WORKFLOW
Each step has:
- **Objective**: What schema field(s) to collect
- **Approach**: How to ask and interpret responses
- **Decision Logic**: When to create schema and which agent to transfer to
- **Examples**: Model dialogue patterns
"""
    
def create_agent_session(model:str = "Qwen3-32B-AWQ") -> AgentSession:
    
    if model == "gpt-4o-mini":
        # use gpt-4o-mini
        model = create_pydantic_azure_openai('gpt-4o-mini')
        
    elif model == "Qwen3-32B-AWQ":
        # use Qwen3-32B-AWQ
        model = pydantic_openai_like_async(model_name = model, max_tokens = 24000)  
        
    else:
        raise ValueError(f"Model {model} not supported, please choose from ['gpt-4o-mini', 'Qwen3-32B-AWQ']")



    agents = {
        
        # Complex business rules, you need more prompt, but just in this step.
        "customer_name_inquiry": AgentNode(
            name="customer_name_inquiry",
            model=model,
            instruction=INSTRUCTION,
            task_cls= CustomerName,
            step_instruction=(
                "Confirm the customer's name with a greeting message(customer name included in the message)."
                "Amubiguous response from customer should be treated as confirmation and create the schema. Current Customer Name: 李老三"
                "Any response for the message will be treated as confirmation unless the customer explicitly says he/she is not the person or dialed wrong number."
                "Event a simple ‘嗯’, ‘呃’, '哪里'，‘你说’ indicates a confirmation. You should create the schema immediately"
            ),
                
            examples=["您好，请问是xxx(plug customer name here)吗？"],
            ),
        
        # Complex business rules, you need more prompt, but just in this step.
        "financial_support_inquiry": AgentNode(
            name="financial_support_inquiry",
            model=model,
            instruction=INSTRUCTION,
            task_cls= FinancialSupportStatus,
            step_instruction=(
                "Briefly introduce yourself(您好，这边是易鑫集团的金融顾问) and ask the customer whether he/she needs financial support.（Do COPY the example）. 看到你的申请的资金方案，您最近是有资金需求吗？"
                "If user acknowledged with `是`，‘嗯’，‘有’，‘有的’， call tool to create FinancialSupportStatus(require_financial_support=True)."
                "If user does not explicitly reject or refuse, create FinancialSupportStatus(require_financial_support=True)."
                "If user explicity reject or refuse, ask again, if the reject is truely confirmed, create FinancialSupportStatus(require_financial_support=False)."
            ),
            examples=["您好，这边是易鑫集团的金融顾问，看到你的申请的资金方案，您最近是有资金需求吗？"],
            ),
        
        # simple business rules, you can be direct and concise.
        "vehicle_payment_status": AgentNode(
            name="vehicle_payment_status",
            model=model,
            instruction=INSTRUCTION,
            task_cls= PaymentMethod,
            step_instruction=(
                "Start a question by asking the customer if his/her vehicle is bought on finance or fully paid off.(您的车是全款买的还是按揭买的？)."
                "If the vehicle is fully paid off（全款）, create VehicleNotUnderRepayment(is_not_under_repayment=True). "
                "If the vehicle is bought with a loan(按揭)， follow up by asking '那您的分期现在还完了么'."
                "If the user confirms like 还清了/换完了, create VehicleNotUnderRepayment(is_not_under_repayment=True)."
                "If the vehicle is still under repayment (还没有/还在贷款/贷款呢/还差一点/过几个月/还有几期), create VehicleNotUnderRepayment(is_not_under_repayment=False)."
                "Overall, you if there are ongoing installments, is_not_under_repayment = False, else True"
            ),
            examples=["您名下的车目前是已经还清贷款了吗？"],
            ),
        
        # simple business rules, you can be direct and concise.
        "vehicle_liscence_under_control": AgentNode(
            name="vehicle_liscence_under_control",
            model=model,
            instruction=INSTRUCTION,
            task_cls= VehicleLiscenceUnderControl,
            step_instruction=(
                "Ask the customer whether the vehicle liscence is under the customer's control.（vehicle liscence is also referred to as '绿本', '机动车行驶证' in Chinese）"
                "If the customer says the vehicle liscense is under his/her control like '在手里', '在家里'，'嗯', create VehicleLiscenceUnderControl(green_book_available=True)."
                "If the vehicle liscense is under his/her spouse or family member's control, create VehicleLiscenceUnderControl(green_book_available=True)."
                "If the vehicle liscense ir under his/her company's control, create VehicleLiscenceUnderControl(green_book_available=False)."
            ),
            examples=["那这个绿本现在是在您本人手上吗？"],
            ),
        
        "wechat_account_confirm": AgentNode(
        name = 'wechat_account_confirm',
        model = model,
        instruction = INSTRUCTION,
        task_cls=WeChatAccount,
        step_instruction = (
            "The only task in this step is to ask the customer whether the current taking phone can be used to add wechat account. (方便用您尾号xxxx的手机号加您的微信么？)"
            "If the customer acknowledge with the current talking phone, create WeChatAccount(wechat_account=current_talking_phone)."
            "If the current talking phone is not associated with the customer's wechat account, continue persuade the customer to provide a valid phone number(associated with wechat account)"
            "progressively collect the valid phone number.(Use `check_wechat_account_validity` whenever you receive a new alpha numeric part)"
            "When guiding the customer to provide complete phone number, response in extremely short sentence like: '您继续', '嗯嗯'"
            "Current Talking Phone Number: 15001395923"
        ),
        examples = [
            "Assistant: 方便用您尾号xxxx的手机号加您的微信么？ Customer: 可以 -> create WeChatAccount(wechat_account=current_talking_phone).",
            "Assistant: 方便用您尾号xxxx的手机号加您的微信么？ Customer: 不方便 Assistant: 那您方便提供一个能加微信的手机号吗.",
            "Customer: 150 Assistant: 您继续 Customer:0123 -> Assistant: 嗯嗯 -> Customer:0245 -> (check validity with `check_wechat_account_validity`) -> if True Assistant: 好的，确认一下是，15001230245吗？ Customer: 对的 -> create WeChatAccount(wechat_account=15001230245)", 
            "Customer: 不方便，加微信干嘛？ Assistant: 加微信是后续办理业务方便，咱们在微信上提供一些资料，最快当天就能放款，您请放心"
        ],
        tools = [create_phone_num_check_tool()]
        ),
        
        "wechat_add_request": AgentNode(
            name = "wechat_add_request",
            model = model,
            instruction = INSTRUCTION,
            task_cls = WeChatRequestReceived,
            step_instruction = (
                "Start with a short sentence to inform the customer that you will send a wechat add request. (您先别挂， 我现在加您一下，稍等哈。)"
                "Then use confirmed account(phone number) to send wechat add request to customer."
                "Use `add_wechat_account` tool to send wechat add request to customer. The account should be the one confirmed by customer in previous step."
                "when user received the wechat add request, any positive response like '看到了', '收到了', or other positive response from the customer create WeChatRequestReceived(received=True)."
                "If the user failed to received the request, you can response with '可能是网络有延迟，您下拉刷新看下有没新的消息' to prompt the user to check the message again. If the user still can not receive the request, you can resend the wechat add request by calling `add_wechat_account` tool again."
                "If you tried multiple times, but eventually the user cannot received it. create WeChatRequestReceived(received=False)"        
            ),
            examples = [
                "您先别挂， 我现在加您一下，稍等哈。",
                "Customer: 好的 Assistant: call add_wechat_account(...), then tell user ‘我加您了，是企业微信加的。麻烦您在微信消息列表找一下‘服务通知’，里面应该有个邀请，您看下有没有收到？’",
                "Customer: 收到了 Assistant: create WeChatRequestReceived(received=True)",
                "Customer: 没收到 Assistant: 可能是网络有延迟，您下拉刷新看下有没新的消息",
                "Customer: 还是没收到 Assistant: 那我这边再给你重新发送一次 -> Call add_wechat_account(...) again"
            ],
            tools = [add_wechat_account]
        ),
        
        "wechat_guide": AgentNode(
            name = 'wechat_guide',
            model = model,
            instruction = INSTRUCTION,
            task_cls=WeChatAccpeted,
            step_instruction = (
                "Dynamically guide the user through the 4-step acceptance process.\n"           
                """
                "INTERNAL KNOWLEDGE [The Procedure]:",
                "  Step A: Inside 'Service Notification', click message '企业微信加好友'. -> 看到那个蓝色的‘企业微信加好友’了吗？点进去。'",
                "  Step B: Long-press the QR Code image.(not scan the QA code with a carmera) -> '好，长按这个二维码别松手。'",
                "  Step C: Select '打开对方企业微信名片' (Open Business Card). -> '弹出的菜单里，点那个‘打开对方企业微信名片’。'",
                "  Step D: Click the blue button '添加到通讯录' (Add to Contacts). -> '对，直接点‘添加到通讯录’。'",
                "STRATEGY:",
                "1. Listen to user's current status.",
                "2. Map to [The Procedure].",
                "3. Output instruction ONLY for the IMMEDIATE NEXT STEP.",
                "4. If user jumps ahead, skip previous steps.",
                "5. If the user complains about the complex procedure or seems churn, try to persuade them gently to continue.",
                """
            ),
            examples = [
                "User: '点进来了' -> Agent: '看到那个蓝色的‘企业微信加好友’了吗？点进去。'",
                "User: '看到二维码了' -> Agent: '好，长按这个二维码别松手。'",
                "User: '长按了' -> Agent: '弹出的菜单里，点那个‘打开对方企业微信名片’。'",
                "User: '看到了添加按钮' -> Agent: '对，直接点‘添加到通讯录’。'",
                "User: '太麻烦了/不弄了/不想加了' -> Agent: '马上就完成了呢，你稍微操作几个步骤就好了，很快的。'"
                "User：嗯嗯/哦/ambiguous response -> Agent: [short answer to guide the next step or explain the current step] 看到了吗？ 进去了吗？打开了么？，点了吗？" 
                "User: 你们利率是多少/能贷款多少/... Agent: 您先加上微信，我稍后在微信给您详细介绍好么？"
                ]
        ),
        
        "hangup": HangUpNode(model = model)
    }


    runner = MultiAgentRunner(
        agents=agents, 
        entry_agent_name="customer_name_inquiry",
        ending_message="好的，我们稍后会加您的微信，请你注意在服务通知后查看我们的企业微信请求，再见！"
    )  

    chat = AgentSession(runner)

    return chat