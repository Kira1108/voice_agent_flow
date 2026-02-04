from task_cls import (
    CustomerName,
    FinancialSupportStatus,
    VehicleNotUnderRepayment, 
    VehicleLiscenceUnderControl,
    AgreeToAddWechatAccount,
    WeChatId,
    
)

from voice_agent_flow.node import AgentNode
from voice_agent_flow.llms import create_pydantic_azure_openai
from voice_agent_flow.llms import create_ollama_model

from voice_agent_flow.runner import AgentRunner

model = create_pydantic_azure_openai('gpt-4o-mini')

instruction = """
You are a customer service representative in a auto finance company. You task is to talk with customer via telephone to collection information.
You will be given one task defined by a schema, you should talk with the user until you can fill the schema completely.
You shoule speak in Chinese.
On each text reply, you should output as less text as possible to collect information.

Overall Conversation Policy: For yes or not question, if the customer's response is not explicitly reject or refuse, you should assume the customer agree or accept it.
Then create the schema with the corresponding fields filled.

Note you are part of a multi-agent system, do not add additional explaination, greeting or closing statement, just focus on collecting information to fill the schema.
Do not Generate Structured output until you have collected information defined by the schema. Before that ,you shoule chat with the customer.

"""

agents = {
    
    # Complex business rules, you need more prompt, but just in this step.
    "customer_name_inquiry": AgentNode(
        name="customer_name_inquiry",
        model=model,
        instruction=instruction,
        task_cls= CustomerName,
        step_instruction=(
            "Confirm the customer's name with a greeting message(customer name included in the message)."
            "Amubiguous response from customer should be treated as confirmation and create the schema. Current Customer Name: 李老三"
            "Any response for the message will be treated as confirmation unless the customer explicitly says he/she is not the person or dialed wrong number."
            "Event a simple ‘嗯’, ‘呃’, '哪里'，‘你说’ indicates a confirmation. You shoule create the schema immediately"
        ),
            
        examples=["您好，请问是xxx(plug customer name here)吗？"],
        ),
    
    # Complex business rules, you need more prompt, but just in this step.
    "financial_support_inquiry": AgentNode(
        name="financial_support_inquiry",
        model=model,
        instruction=instruction,
        task_cls= FinancialSupportStatus,
        step_instruction=(
            "Ask the customer whether he/she need financial support."
            "Customer raising a new question in step indicates she/he is interested in the financial support."
            "You should answer the customer query first, after the deviation is handled, create the schema immediately."
        ),
        examples=["您好，李先生，请问您对我们的金融支持服务感兴趣吗？"],
        ),
    
    # simple business rules, you can be direct and concise.
    "vehicle_payment_status": AgentNode(
        name="vehicle_payment_status",
        model=model,
        instruction=instruction,
        task_cls= VehicleNotUnderRepayment,
        step_instruction="Ask the customer whether the vehicle is already fully paid off.",
        examples=["请问您的车辆目前是已经还清贷款了吗？"],
        ),
    
    # simple business rules, you can be direct and concise.
    "vehicle_liscence_under_control": AgentNode(
        name="vehicle_liscence_under_control",
        model=model,
        instruction=instruction,
        task_cls= VehicleLiscenceUnderControl,
        step_instruction="Ask the customer whether the vehicle liscence is under the customer's control.",
        examples=["请问您的车辆行驶证现在是在您本人手上吗？"],
        ),
    
    # simple business rules, you can be direct and concise.
    "agree_wechat_add": AgentNode(
        name="agree_wechat_add",
        model=model,
        instruction=instruction,
        step_instruction="Ask the customer whether he/she agree to add wechat account for further contact.",
        task_cls= AgreeToAddWechatAccount,
        examples=["为了方便后续联系，您是否同意添加我们的微信账号？"],
        ),
    
    # simple business rules, you can be direct and concise.
    "ask_wechat_id": AgentNode(
        name="ask_wechat_id",
        model=model,
        instruction=instruction,
        step_instruction="Ask the customer for his/her wechat id.",
        task_cls= WeChatId,
        examples=["请问您的微信号是多少？"],
        ),
}

runner = AgentRunner(
    agents=agents, 
    entry_agent_name="customer_name_inquiry",
    ending_message="好的，我们稍后会加您的微信，请你注意在服务通知后查看我们的企业微信请求，再见！"
)  
            
if __name__ == "__main__":
    customer_utterances = [
        "喂，你好",
        "呃",
        "再说一次，你说啥？",
        "哦哦，我有需求",
        "还清了",
        "在手上" ,
        "可以",
        "liushaoshan123"
    ]
    
    for idx, utterance in enumerate(customer_utterances):
        print(">>>>>>> Turn", idx+1," <<<<<<<")
        print("Customer:", utterance)
        response = runner.run(utterance)
        print("Agent:", response)
        
    print("-"*30, "Collected Information", "-"*30)
    print(runner.show_information())