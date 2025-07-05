# SummaryAgent: Tutorial & Reference

## Overview

`SummaryAgent` is a conversational agent designed to summarize and interact with users while maintaining a condensed memory of the conversation. It leverages `CondensedMemoryBlock` to efficiently store and retrieve key context, enabling coherent multi-turn interactions.

## Usage with CLI

At command line, after entering the virtual env (`memagents`), type the following to invoke the `SummaryAgent` at the project toplevel directory:

For Mac or Linux:
```bash
export PYTHONPATH="`pwd`"

python -m asdrp.agent.summary_agent
```

For Windows or Powershell:
```powershell
# using Get-Location (alias: pwd) and grabbing its Path as a string
$Env:PYTHONPATH = (Get-Location).Path

python -m asdrp.agent.summary_agent  
```

## Key Features
- **Condensed memory:** Uses an LLM-powered memory block to keep conversation context within a token limit.
- **Flexible tools:** Can be extended with custom tools for additional capabilities.
- **Verbose mode:** Optionally prints memory summaries for debugging or transparency.

## Memory Setup
- By default, `SummaryAgent` creates a `CondensedMemoryBlock` as its memory block.
- You can also inject a custom memory instance for testing or advanced use cases.

```python
from asdrp.agent.summary_agent import SummaryAgent
from llama_index.llms.openai import OpenAI

agent = SummaryAgent(llm=OpenAI(model="gpt-4.1-mini"))
```

## Generating and Displaying Summaries
- The agent summarizes each message using the LLM and stores the result in memory.
- When `verbose=True`, the agent prints a formatted summary of all memory fragments after each turn.

## Example Usage
```python
reply = await agent.achat("What is the weather today?")
print(reply.response_str)
```

## Integration with CondensedMemoryBlock
- The agent's memory is a `CondensedMemoryBlock`, which summarizes and stores each message fragment.
- You can access or print the memory summary using the agent's internal methods.

## See Also
- [condensed_memory.md](./condensed_memory.md) 