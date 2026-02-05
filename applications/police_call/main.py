from task_cls import (
    PoliceCallBasicInfo,
    SafetySuggestionProvided
    
)

from voice_agent_flow.node import AgentNode
from voice_agent_flow.llms import create_pydantic_azure_openai
from voice_agent_flow.llms import create_ollama_model

from voice_agent_flow.runner import AgentRunner

model = create_pydantic_azure_openai('gpt-4o-mini')

instruction = """
You are a police call center agent(working at 110). You task is to talk with caller via telephone to collection information.
You resopnse should be berief and direct to the point.
"""

agents = {
    "police_call_basic_info": AgentNode(
        name="police_call_basic_info",
        model=model,
        instruction=instruction,
        task_cls= PoliceCallBasicInfo,
        step_instruction="Collect basic information about the police call including case location, case type, description and caller name. ask the question one at a time, do not ask multiple questions in one message.",
        examples=["请问您遇到什么紧急情况？ / 发生在哪里？/ 能简单描述一下吗？/ 您的姓名是？"],
        ),
    "safety_suggestion": AgentNode(
        name="safety_suggestion",
        model=model,
        instruction=instruction,
        task_cls= SafetySuggestionProvided,
        step_instruction="Based on the collected information, provide safety suggestion(A clear command to keep safe) to the caller. After the caller responded to your suggestion, create the schema.",
        examples=["请您保持冷静，.......(provide clear safety suggestion)", "请您打开窗户，保持空气流通"],
        ),
    }

runner = AgentRunner(
    agents=agents, 
    entry_agent_name="police_call_basic_info",
    ending_message="好的，我们这就派人过去处理您的情况，请您保持电话畅通！"
)  
       
     