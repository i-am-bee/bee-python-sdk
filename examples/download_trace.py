"""
Download run trace from the observe API

Note: observe api requires a different base_url `/observe` vs `/v1`
"""

import json
import os
import time
from contextlib import suppress
from pprint import pprint

from openai import BaseModel, OpenAI, NotFoundError


def heading(text: str) -> str:
    """Helper function for centering text."""
    return "\n" + f" {text} ".center(80, "=") + "\n"


# Instantiate Bee client with Bee credentials from env
bee_client = OpenAI(base_url=f'{os.getenv("BEE_API")}/v1', api_key=os.getenv("BEE_API_KEY"))

# Instantiate Observe client with Bee credentials from env, but DIFFERENT base_url (!)
observe_client = OpenAI(base_url=f'{os.getenv("BEE_API")}/observe/v1', api_key=os.getenv("BEE_API_KEY"))

print(heading("Create run"))
assistant = bee_client.beta.assistants.create(model="meta-llama/llama-3-1-70b-instruct")
question = "What is the opposite color of blue"
thread = bee_client.beta.threads.create(messages=[{"role": "user", "content": question}])
run = bee_client.beta.threads.runs.create_and_poll(thread_id=thread.id, assistant_id=assistant.id)

if run.status != "completed":
    raise RuntimeError(f"Run is in an unexpected state: {run.status}\nError: {run.last_error}")

print("Run:")
pprint(run.model_dump())

assert run.status == "completed"

print(heading("Download trace"))
# Get trace_id
trace_info = bee_client.get(f"/threads/{thread.id}/runs/{run.id}/trace", cast_to=BaseModel)


def get_trace(trace_id: str, params: dict):
    # Uploading trace is an asynchronous process that takes 40-60s, hence we need to retry a few times.
    for attempt in range(1, 10):
        with suppress(NotFoundError):
            return observe_client.get(f"/traces/{trace_info.id}", options={"params": params}, cast_to=BaseModel)
        time.sleep(attempt * 0.5)
    raise RuntimeError("Unable to download trace")


trace = get_trace(trace_info.id, {"include_tree": True})
print("Trace:")
print(json.dumps(trace.model_dump(mode="json"), indent=2))

# Cleanup
bee_client.beta.assistants.delete(assistant_id=assistant.id)
