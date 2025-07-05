# CondensedMemoryBlock: Tutorial & Reference

## Overview

`CondensedMemoryBlock` is a smart memory buffer for conversational agents. It maintains a condensed, token-limited summary of conversation history, using an LLM to semantically summarize each message fragment. This enables agents to retain context efficiently, even with long or multi-turn conversations.

## Key Features
- **Token-limited memory:** Keeps only as much context as fits within a specified token limit.
- **Semantic summarization:** Uses an LLM to summarize each message to a configurable character limit.
- **Flexible integration:** Can be used as a memory block in any agent using the `llama_index` memory system.

## Class Reference

### `CondensedMemoryBlock`
```python
class CondensedMemoryBlock(BaseMemoryBlock[str]):
    name: str
    token_limit: int = DEFAULT_TOKEN_LIMIT
    current_memory: List[str] = Field(default_factory=list)
    _tokenizer: tiktoken.Encoding = PrivateAttr(...)
    _llm: OpenAI = PrivateAttr(...)
```

#### Constructor
```python
CondensedMemoryBlock(name: str, token_limit: int = DEFAULT_TOKEN_LIMIT)
```
- `name`: Name of the memory block.
- `token_limit`: Maximum number of tokens to store (default: 50,000).

#### Important Methods
- `async def _aput(self, messages: List[ChatMessage]) -> None`  
  Adds new messages to memory, summarizing each with the LLM.
- `async def summarize_text_with_llm(self, text: str, max_chars: int) -> str`  
  Summarizes a string using the LLM, respecting a character limit.
- `async def _aget(self, ...) -> str`  
  Returns the current memory as a single string.

## Summarization Logic
- Each message is summarized to `MAX_SUMMARY_CHARS` (default: 128 chars) using an LLM (e.g., OpenAI GPT-4.1-mini).
- The summarization prompt is customizable via `SUMMARIZE_TEXT_PROMPT`.
- Summaries are stored in `current_memory` and trimmed if the token limit is exceeded.

## Example Usage
```python
from asdrp.memory.condensed_memory import CondensedMemoryBlock
from llama_index.llms.openai import OpenAI

memory_block = CondensedMemoryBlock(name="my_memory", token_limit=100)
memory_block._llm = OpenAI(model="gpt-4.1-mini")  # Set LLM if needed

# Add messages (async context)
await memory_block._aput([ChatMessage(blocks=[TextBlock(text="Hello, world!")], additional_kwargs={})])

# Retrieve memory
summary = await memory_block._aget()
print(summary)
```

## Integration with Agents
- Used in `SummaryAgent` as a memory block to provide condensed conversation context.
- Can be combined with other memory blocks for more complex agent memory architectures.

## See Also
- [summary_agent.py documentation](./summary_agent.md)
- [llama_index memory documentation](https://docs.llamaindex.ai/en/stable/module_guides/memory/) 