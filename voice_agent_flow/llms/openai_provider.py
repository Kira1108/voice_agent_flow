from openai import AzureOpenAI, AsyncAzureOpenAI
from voice_agent_flow.load_env import load_environment

def create_sync_client(*args, **kwargs):
    load_environment()  
    return AzureOpenAI(*args, **kwargs)

def create_async_client(*args, **kwargs):
    load_environment()  
    return AsyncAzureOpenAI(*args, **kwargs)