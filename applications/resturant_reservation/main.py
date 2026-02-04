from task_cls import (
    PartySizeResult,
    TimeResult
)

from voice_agent_flow.node import AgentNode
from voice_agent_flow.llms import create_pydantic_azure_openai

from voice_agent_flow.runner import AgentRunner


model = create_pydantic_azure_openai()

instruction = """
You are a restaurant reservation agent, you help customer to make a reservation at a restaurant.
You will be given one task defined by a schema, you should talk with the user until you can fill the schema completely.
You are friendly and polite, you answer in a concise way: Reply to the user first, then back to your task.
You shoule speak in the same language as the customer.

Note you are part of a multi-agent system, do not add additional explaination, greeting or closing statement, just continue talking.
BE SURE THAT YOU DON"T IGNORE THE CURRENT QUESTION CUSTOMER ASKS.

Handle Deviation first, then back to the main task.

Restaurant Location: Beijing, China, Wangfujing Street, NewWorld Department Store, 3rd Floor 2203.
"""

def check_availability(
        party_size: int, 
        reservation_time:str
    ) -> bool:
    """Check the availability of the restaurant for the given party size and reservation time."""
    print("Checking availability for party size", party_size, "at time", reservation_time)
    return True

agents = {
    "party_size_collector": AgentNode(
        name="party_size_collector",
        model=model,
        instruction=instruction,
        step_instruction="Ask the customer about the party size for the reservation.",
        examples=[
            "Hi there, how can I help you today? (First turn greeting example)",
            "How many people will be joining you for the reservation? (Second turn party size inquiry example)"],
        task_cls= PartySizeResult
        ),
    
    "time_collector": AgentNode(
        name="time_collector",
        model=model,
        instruction=instruction,
        step_instruction="Ask the customer about the reservation time. You should check the availability of the restaurant before confirming the reservation.",
        examples=["What time would you like to make the reservation for?"],
        task_cls= TimeResult,
        tools = [check_availability]
        ),
}

runner = AgentRunner(
    agents=agents, 
    entry_agent_name="party_size_collector",
    ending_message='感谢您的接听，已经帮您预定好了，再见！'
)
    
    
if __name__ == "__main__":
    customer_utterances = [
        " hi there",
        "I want to know the location of the restaurant.",
        "Okay, There will be totally 4 people.",
        "About 2026-3-3 20:00" 
    ]
    
    for utterance in customer_utterances:
        print("Customer:", utterance)
        response = runner.run(utterance)
        print("Agent:", response)
        
    print(runner.show_information())