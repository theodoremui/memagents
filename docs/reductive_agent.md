# ReductiveAgent: Tutorial & Reference

## Overview

`ReductiveAgent` is a conversational agent that performs reductive reasoning by extracting, storing, and referencing key propositions from conversation history. It uses `PropositionExtractionMemoryBlock` to build a knowledge base of facts, beliefs, and goals, enabling more informed and context-aware responses.

## Key Features
- **Proposition extraction:** Automatically identifies and stores key propositions from user input.
- **Knowledge referencing:** Prepends known propositions to user queries for more informed answers.
- **Flexible memory:** Can be combined with other memory blocks for advanced use cases.

## Memory Setup
- By default, `ReductiveAgent` creates a `PropositionExtractionMemoryBlock` as its memory block.
- You can also inject a custom memory instance for testing or advanced use cases.

```python
from asdrp.agent.reductive_agent import ReductiveAgent
from llama_index.llms.openai import OpenAI

agent = ReductiveAgent(llm=OpenAI(model="gpt-4.1-mini"))
```

## Proposition Extraction and Referencing
- The agent extracts propositions from each message and stores them in memory.
- When responding, it prepends known propositions to the user message, instructing the LLM to reference them if relevant.

## Example Usage
```python
reply = await agent.achat("What do you know about the sky?")
print(reply.response_str)
```

## Integration with PropositionExtractionMemoryBlock
- The agent's memory is a `PropositionExtractionMemoryBlock`, which extracts and stores propositions in XML format.
- You can access or print the extracted propositions using the memory block's methods.

## See Also
- [proposition_extraction_memory.md](./proposition_extraction_memory.md) 