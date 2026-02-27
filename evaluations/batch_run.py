from voice_agent_flow.apps.car_loan import create_agent_session
from voice_agent_flow.memory import Memory
from agentic_data.testset import load_dataset
import json
import asyncio
from datetime import datetime
from pathlib import Path
from uuid import uuid4

eval_folder = Path(__file__).parent / 'runs'

if not eval_folder.exists():
    eval_folder.mkdir(parents=True, exist_ok=True)

async def run_single(messages:list[dict]):
    memory = Memory.from_dict(messages)
    chat = create_agent_session()
    chat.set_agent("wechat_account_confirm")
    chat.set_memory(memory)
    _ = await chat._chat()
    events = chat.new_events
    output = events.get("output", "None")
    del events["output"]
    events = str(events)
    return output, events


async def run_sample(sample):
    try:
        step2agent = {
            "greeting": "customer_name_inquiry",
            "financial_support": "financial_support_inquiry",
            "car_ownership":"vehicle_payment_status",
            "vehicle_payment_type":"vehicle_payment_status",
            "green_book_avaliable":"vehicle_liscence_under_control",
            "city":"vehicle_liscence_under_control",
            "wechat_account_confirm":"wechat_account_confirm",
            "sending_wechat_request":"wechat_add_request",
            "wechat_guidance":"wechat_guide"
        }

        agent_name = step2agent[sample['step_tag']]
        memory = Memory.from_dict(sample['messages'])
        chat = create_agent_session()
        chat.set_agent(agent_name)
        chat.set_memory(memory)
        _ = await chat._chat()
        events = chat.new_events
        output = events.get("output", "None")
        del events["output"]
        events = str(events)
        sample['agent_output'] = output
        sample['agent_run_items'] = events
        return sample
    
    except Exception as e:
        print(f"Error processing sample: {e}")
        return None


async def run_batch(samples, concurrency:int = 5):
    # use semaphore to limit concurrency
    semaphore = asyncio.Semaphore(concurrency)  # limit to concurrency concurrent tasks
    async def sem_run(sample):
        async with semaphore:
            revised_sample = await run_sample(sample)
            return revised_sample
    tasks = [sem_run(sample) for sample in samples]
    return await asyncio.gather(*tasks)


async def eval_dataset(dataset_name:str = 'common', limit:int = None):
    dt_string = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = eval_folder / f"{dataset_name}_{dt_string}.json"
    dataset = load_dataset(dataset_name)
    if limit is not None:
        dataset = dataset[:limit]
    results = await run_batch(dataset, concurrency=5)
    with open(filename, 'w') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":  
    asyncio.run(eval_dataset('common'))
    asyncio.run(eval_dataset('handoff'))