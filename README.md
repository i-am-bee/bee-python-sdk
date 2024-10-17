# Bee API Python experience

Interact with bee-api through **OpenAI Python SDK**

> **⚠️ COMPATIBILITY DISCLAIMER ⚠️**
> The bee-api is designed to overlap with OpenAI API
> to a degree that is sufficitent for most use-cases, however some types and operations are not 100%
> compatible, see the section on [OpenAI Compatibility](#openai-compatibility-) below. The examples provided in this
> repository are regularly tested to be working, however we will never guarantee full compatibility with OpenAI.
> Please create an issue if you stumbled upon non-compatible behavior that is blocking you.

## Quick Start

### Installation 🔧

We are using purely OpenAI SDK, so the only required package is `openai`.

```shell
pip install openai
```

### Environment variables ♻️

Make sure to have the following environment variables configured,
see [example.env](example.env):

```
BEE_API=http://localhost:4000
BEE_API_KEY=sk-testkey

# This is required to prevent some pydantic serialization errors
DEFER_PYDANTIC_BUILD=false
```

### Basic usage 🧑‍💻

```python
import os
from openai import OpenAI

# Instantiate OpenAI client with Bee credentials from env
client = OpenAI(base_url=f'{os.getenv("BEE_API")}/v1', api_key=os.getenv("BEE_API_KEY"))

# Create assistant
assistant = client.beta.assistants.create(
    model="meta-llama/llama-3-1-70b-instruct", tools=[{"type": "code_interpreter"}]
)

# Create a thread with user message
question = "What is the unicode character U+1F41D? Create a haiku about it."
thread = client.beta.threads.create(messages=[{"role": "user", "content": question}])

# Create a run and wait for completion
run = client.beta.threads.runs.create_and_poll(thread_id=thread.id, assistant_id=assistant.id)
assert run.status == "completed"

# List messages and get an answer
messages = client.beta.threads.messages.list(thread_id=thread.id)
print("Answer:", messages.data[0].content[0].text.value)
```

## Run examples 🏃‍♀️

If you want to run the examples in this repository, install all dependencies
using [poetry](https://python-poetry.org/).

```shell
# Install dependencies
poetry install
```

Create a correct `.env` file.

```shell
# Create .env file
cp example.env .env

# Insert your API key
open .env
```

Run examples through poetry

```shell
poetry run python -m examples.basic_usage
```

## OpenAI Compatibility 🧭
Here are the important differences from the official [OpenAI SDK](https://github.com/openai/openai-python).

### Supported OpenAI APIs ✅

```python
from openai import OpenAI

client: OpenAI = ...

# OpenAI
# Assistants
client.beta.assistants
client.beta.threads
client.beta.threads.messages
client.beta.threads.runs
client.beta.threads.runs.steps
client.beta.threads.runs.stream  # early stage event compatibility
client.beta.vector_stores
client.beta.vector_stores.files

## Not supported:
# client.beta.vector_stores.file_batches

# Files
client.files
```

### Bee API extensions 🐝

Extensions can be called using the low-level OpenAI client methods (get, post, put, delete). See the
definition of a custom tool in [examples/custom_tool.py](examples/custom_tool.py) as an example.

```python
from openai import OpenAI, BaseModel

client: OpenAI = ...

# Tools
client.get('/tools', cast_to=BaseModel)  # list tools
client.post('/tools', cast_to=BaseModel)  # create tool
client.post('/tools/:tool_id', cast_to=BaseModel)  # update tool
client.delete('/tools/:tool_id', cast_to=BaseModel)  # delete tool

# Observe
client.get('/threads/:thread_id/runs/:run_id/trace')  # Get trace ID for a run
```

### Bee observe 🎥

Observe API module is designed to provide full trace of everything that happened during a run.
You can obtain the trace using a special `/observe` endpoint, here is a brief example, for full code
see [examples/download_trace.py](examples/download_trace.py):

```python
import os
from openai import OpenAI, BaseModel

# Normal bee client
bee_client = OpenAI(base_url=f'{os.getenv("BEE_API")}/v1', api_key=os.getenv("BEE_API_KEY"))
thread = ...
run = ...
trace_info = bee_client.get(f"/threads/{thread.id}/runs/{run.id}/trace", cast_to=BaseModel)

# (!) Note different base_url
observe_client = OpenAI(base_url=f'{os.getenv("BEE_API")}/observe', api_key=os.getenv("BEE_API_KEY"))

# Get trace
params = {"include_tree": True, "include_mlflow": True}
trace = observe_client.get(f"/trace/{trace_info.id}", options={"params": params}, cast_to=BaseModel)
```

### OpenAI documentation 📄

Use the official OpenAI documentation with caution (see **caveats** below), here are links to the relevant topics:

- [Assistants](https://platform.openai.com/docs/assistants/overview)
- [API reference](https://platform.openai.com/docs/api-reference/assistants)

### ⚠️ Caveats ⚠️

- *streaming* events are not fully compatible yet, the "With streaming" portions of OpenAI documentation will not work
  as expected (for example in
  [file search](https://platform.openai.com/docs/assistants/tools/file-search/step-5-create-a-run-and-check-the-output))
- *some features are not implemented*:
    - vector store file batches - `client.beta.vector_stores.file_batches`
    - adding message attachment to `file_search` without previously embedding the file in a thread vector store,
      see [examples/vector_store.py](examples/vector_store.py)
    - ... and more
- *some type unions are extended* so the data returned does not match the original openai models:
    - you may see pydantic warningns during serialization, you can avoid this by using `warnings="none"` when dumping a
      model, for example `assistant.model_dump(warnings="none")`
    - you must set the `DEFER_PYDANTIC_BUILD=false` environment variable before all imports, if you see an error similar
      to ['MockValSer' object cannot be converted to 'SchemaSerializer'](https://github.com/pydantic/pydantic/discussions/7710),
      you are probably missing this configuration

# Contributing
This is an open-source project and we ❤️ contributions.

If you'd like to contribute to Bee, please take a look at our [contribution guidelines](./CONTRIBUTING.md).
