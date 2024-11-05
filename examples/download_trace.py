"""
Download run trace from the observe API

Note: observe api requires a different base_url `/observe` vs `/v1`
"""

import json
import os
from pprint import pprint

from openai import BaseModel, OpenAI


def heading(text: str) -> str:
    """Helper function for centering text."""
    return "\n" + f" {text} ".center(80, "=") + "\n"


# Instantiate Bee client with Bee credentials from env
bee_client = OpenAI(base_url=f'{os.getenv("BEE_API")}/v1', api_key=os.getenv("BEE_API_KEY"))

# Instantiate Observe client with Bee credentials from env, but DIFFERENT base_url (!)
observe_client = OpenAI(base_url=f'{os.getenv("BEE_API")}/observe', api_key=os.getenv("BEE_API_KEY"))

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

# Get trace
params = {"include_tree": True, "include_mlflow": True}
trace = observe_client.get(f"/trace/{trace_info.id}", options={"params": params}, cast_to=BaseModel)
print("Trace:")
print(json.dumps(trace.model_dump(mode="json"), indent=2))

# Cleanup
bee_client.beta.assistants.delete(assistant_id=assistant.id)
