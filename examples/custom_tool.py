"""
Define a custom python tool

This is a hosted tool, the code defined locally is uploaded to a server and executed automatically.

See `requirements.txt` in bee-code-interpreter executor for the list of available packages:
https://github.com/i-am-bee/bee-code-interpreter/blob/main/executor/requirements.txt
"""

import os
import warnings
from datetime import datetime
from pprint import pprint
from typing import Literal

import openai
from openai import NotFoundError, OpenAI


def heading(text: str) -> str:
    """Helper function for centering text."""
    return "\n" + f" {text} ".center(80, "=") + "\n"


client = OpenAI(api_key=os.getenv("BEE_API_KEY"), base_url=f'{os.getenv("BEE_API")}/v1')


# Define hosted function
# - imports that are not present in the executor environment are installed automatically (slows down execution)
# - list of installed packages: https://github.com/i-am-bee/bee-code-interpreter/blob/main/executor/requirements-skip.txt
# - docstring must be defined and satisfy the following format:
ip_info_code = '''
import requests

def ip_info(ip: str) -> dict:
    """
    Get information about an IP address, such as location, company, and carrier name.

    :param ip: IP address in the 255.255.255.255 format
    :return: Information about the IP address
    """
    response = requests.get(f"https://ipinfo.io/{ip}/geo")
    response.raise_for_status()
    return response.json()
'''


# Get existing tools using Bee API extension
# You can use cast_to `openai.BaseModel` for basic usage
tools = client.get("/tools", cast_to=openai.BaseModel)

# Delete existing tools with name ip_info to avoid conflicts
for tool in (tool for tool in tools.data if tool["name"] == "ip_info"):
    client.delete(f"/tools/{tool['id']}", cast_to=openai.BaseModel)


class SourceCodeTool(openai.BaseModel):
    id: str
    name: str
    description: str | None = None
    source_code: str
    type: Literal["source_code"]
    created_at: datetime
    json_schema: str | None = None


# You can also cast_to a custom model that inherits from `openai.BaseModel`
print(heading("Create custom source code tool"))
custom_tool = client.post(
    "/tools",
    cast_to=SourceCodeTool,
    # You can also pass the source code directly as a string without python definition
    body={"source_code": ip_info_code},
)
print("Tool:")
pprint(custom_tool.model_dump())

print(heading("Create assistant with the custom tool."))
assistant = client.beta.assistants.create(
    instructions="You are IP address analytic. Use the provided tools to get info about IP address.",
    model="meta-llama/llama-3-1-70b-instruct",
    tools=[{"type": "user", "user": {"tool": {"id": custom_tool.id}}}],
)
print("Assistant:")
# User tools are unexpected in the original type, but we can suppress the warnings
pprint(assistant.model_dump(warnings="none"))

print(heading("Run model and get answer"))
thread = client.beta.threads.create(messages=[{"role": "user", "content": "Who owns the IP 8.8.8.8?"}])
run = client.beta.threads.runs.create_and_poll(thread_id=thread.id, assistant_id=assistant.id)

if run.status != "completed":
    raise RuntimeError(f"Run is in an unexpected state: {run.status}\nError: {run.last_error}")

answer = client.beta.threads.messages.list(thread_id=thread.id).data[0].content[0].text.value
print("Answer:", answer)

# Cleanup
client.beta.threads.delete(thread.id)
client.beta.assistants.delete(assistant.id)
try:
    client.delete(f"/tools/{custom_tool.id}", cast_to=openai.BaseModel)
except NotFoundError:
    warnings.warn("Tool was already deleted, there is probably another test running.", stacklevel=0)
