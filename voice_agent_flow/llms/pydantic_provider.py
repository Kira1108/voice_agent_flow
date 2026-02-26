from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.settings import ModelSettings

from voice_agent_flow.llms.openai_provider import create_async_client
from openai import AsyncOpenAI

def create_pydantic_azure_openai(
    model_name:str  = "gpt-5.2-chat"
) -> OpenAIChatModel:
    
    client = create_async_client()

    model = OpenAIChatModel(
        model_name=model_name,
        provider=OpenAIProvider(openai_client=client),
    )
    
    
    return model


def create_ollama_model(model_name: str = 'gpt-oss:20b'):
    """Create and return an OpenAIChatModel configured to use the Ollama provider."""
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.ollama import OllamaProvider

    return OpenAIChatModel(
        model_name=model_name,
        provider=OllamaProvider(base_url='http://localhost:11434/v1'),
    )
    
