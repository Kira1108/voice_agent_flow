from voice_agent_flow.llms import create_pydantic_azure_openai
from pydantic_ai import( 
    Agent, 
    ModelRequest, 
    UserPromptPart,
    SystemPromptPart,
    TextPart,
    ModelResponse
)

import nest_asyncio
nest_asyncio.apply()

model = create_pydantic_azure_openai()

agent = Agent(
    model=model,
    instructions="You are a helpful assistant."
)

messages = [
    
    ModelRequest(
        parts = [
        UserPromptPart(content = "Hello, what is your model name?"),
        SystemPromptPart(content = "Please answer in a single word."), 
    ]),
    
    ModelResponse(
        parts = [TextPart(content = "ChatGPT")]
    ),
    
    ModelRequest(
        parts = [
            UserPromptPart(content = "Thanks! Can you tell me a joke?"),
            SystemPromptPart(content = "Mix a lot of emojis in your joke.")
        ]
    )
]

result = agent.run_sync(
    message_history = messages
)

print("Response: ",result.output)

messages = result.all_messages()
print("All messages: ")
print(messages)