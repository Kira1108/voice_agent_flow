from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from voice_agent_flow.llms.openai_provider import create_async_client

def create_pydantic_azure_openai(
    model_name:str  = "gpt-5.2-chat"
) -> OpenAIChatModel:
    
    client = create_async_client()

    model = OpenAIChatModel(
        model_name=model_name,
        provider=OpenAIProvider(openai_client=client),
    )
    
    
    return model