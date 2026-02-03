from task_cls import (
    CustomerName,
    FinancialSupportStatus,
    VehicleNotUnderRepayment, 
    VehicleLiscenceUnderControl
)

from node import AgentNode
from agentic_data.llms import pydantic_openai_like_async
from pydantic import BaseModel

model = pydantic_openai_like_async(
    model_name = "Qwen3-32B-AWQ"
)

instruction = """
You are a customer service representative in a auto finance company. You task is to talk with customer via telephone to collection information.
You will be given one task defined by a schema, you should talk with the user until you can fill the schema completely.
You shoule speak in Chinese.
On each text reply, you should output as less text as possible to collect information.

Overall Conversation Policy: For yes or not question, if the customer's response is not explicitly reject or refuse, you should assume the customer agree or accept it.
Then create the schema with the corresponding fields filled.

Note you are part of a multi-agent system, do not add additional explaination, greeting or closing statement, just focus on collecting information to fill the schema.

Example inquiry/response: {example}
"""

agents = {
    "customer_name_inquiry": AgentNode(
        name="customer_name_inquiry",
        model=model,
        instruction=instruction,
        example="您好，请问是李明吗？",
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



class AgentRunner:
    def __init__(self, agents, entry_agent_name: str):
        
        self.agents = agents  
        self.entry_agent = self.get_agent(entry_agent_name) 
        self.current_agent = self.entry_agent
        self.all_messages = []
        
        self.collected_information = []
        
    def get_agent(self, name: str):
        agent_node = self.agents[name]
        return agent_node.create()
    
    def run(self, input_text:str):
        # first run the current agent
        res = self.current_agent.run_sync(input_text, message_history = self.all_messages)
        
        # get the output
        output = res.output
        
        # update message history
        self.all_messages = res.all_messages()
        
        # if string, return directly
        if isinstance(output, str):
            return output
        
        # if base model, run transfer to get next agent
        elif isinstance(output, BaseModel):
            self.collected_information.append(output)
            
            target_agent = output.transfer()
            if target_agent == 'end':
                return "任务完成，谢谢您的配合！"
            
            self.current_agent = self.get_agent(target_agent)
            
            # rerun next agent
            res = self.current_agent.run_sync(input_text, message_history = self.all_messages)
            
            # update message history
            self.all_messages = res.all_messages()
            
            # get output
            output = res.output
            return output
        
        return "Error"
    
    def show_information(self):
        objs = self.collected_information
        return [obj.model_dump() for obj in objs]
        
    
    
    
    
    
            