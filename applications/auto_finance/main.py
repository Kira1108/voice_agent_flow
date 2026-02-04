from task_cls import (
    CustomerName,
    FinancialSupportStatus,
    VehicleNotUnderRepayment, 
    VehicleLiscenceUnderControl
)

from voice_agent_flow.node import AgentNode
from voice_agent_flow.llms import create_pydantic_azure_openai

from voice_agent_flow.runner import AgentRunner

model = create_pydantic_azure_openai()

instruction = """
You are a customer service representative in a auto finance company. You task is to talk with customer via telephone to collection information.
You will be given one task defined by a schema, you should talk with the user until you can fill the schema completely.
You shoule speak in Chinese.
On each text reply, you should output as less text as possible to collect information.

Overall Conversation Policy: For yes or not question, if the customer's response is not explicitly reject or refuse, you should assume the customer agree or accept it.
Then create the schema with the corresponding fields filled.

Note you are part of a multi-agent system, do not add additional explaination, greeting or closing statement, just focus on collecting information to fill the schema.

Example inquiry/response: {example} [Use this as reference only]
"""

agents = {
    "customer_name_inquiry": AgentNode(
        name="customer_name_inquiry",
        model=model,
        instruction=instruction + f"\nConfirm the customer's name with a greeting message(customer name included in the message)",
        example="您好，请问是xxx(plug customer name here)吗？ # Amubiguous response from customer should be treated as confirmation and create the schema. Current Customer Name: 李老三",
        task_cls= CustomerName
        ),
    
    "financial_support_inquiry": AgentNode(
        name="financial_support_inquiry",
        model=model,
        instruction=instruction,
        example="您好，李先生，请问您对我们的金融支持服务感兴趣吗？",
        task_cls= FinancialSupportStatus
        ),
    
    "vehicle_payment_status": AgentNode(
        name="vehicle_payment_status",
        model=model,
        instruction=instruction,
        example="请问您的车辆目前是已经还清贷款了吗？",
        task_cls= VehicleNotUnderRepayment
        ),
    
    "vehicle_liscence_under_control": AgentNode(
        name="vehicle_liscence_under_control",
        model=model,
        instruction=instruction,
        example="请问您的车辆行驶证现在是在您本人手上吗？",
        task_cls= VehicleLiscenceUnderControl
        ),
}

    
runner = AgentRunner(
    agents=agents, 
    entry_agent_name="customer_name_inquiry"
)
    
            
if __name__ == "__main__":
    customer_utterances = [
        "喂，你好",
        "呃",
        "再说一次，你说啥？",
        "哦哦，我有需求",
        "还清了",
        "在手上"  
    ]
    
    for utterance in customer_utterances:
        print("Customer:", utterance)
        response = runner.run(utterance)
        print("Agent:", response)
        
    print(runner.show_information())