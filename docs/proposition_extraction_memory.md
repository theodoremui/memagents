# PropositionExtractionMemoryBlock: Tutorial & Reference

## Overview

`PropositionExtractionMemoryBlock` is a memory block for extracting, storing, and condensing key propositions (facts, opinions, beliefs, etc.) from conversations. It uses an LLM to parse and condense information into a structured XML format, enabling agents to reason over user-disclosed knowledge.

## Key Features
- **Proposition extraction:** Identifies and stores key facts, beliefs, and goals from conversation turns.
- **Condensing:** Limits the number of stored propositions, merging or dropping as needed.
- **XML structure:** Propositions are stored in a machine-readable XML format for easy parsing.

## Class Reference

### `PropositionExtractionMemoryBlock`
```python
class PropositionExtractionMemoryBlock(BaseMemoryBlock[str]):
    name: str
    llm: LLM
    propositions: List[str]
    max_propositions: int
    ...
```

#### Constructor
```python
PropositionExtractionMemoryBlock(name: str, llm: LLM, max_propositions: int = 50)
```
- `name`: Name of the memory block.
- `llm`: The LLM instance to use for extraction.
- `max_propositions`: Maximum number of propositions to store.

#### Important Methods
- `async def _aput(self, messages: List[ChatMessage]) -> None`  
  Extracts and adds new propositions from messages.
- `async def _aget(self, ...) -> str`  
  Returns the current propositions as XML-formatted string.

## Extraction & Condensing Logic
- Uses LLM prompts to extract new propositions from each message, avoiding duplicates.
- If the number of propositions exceeds `max_propositions`, the LLM is prompted to condense them.
- Propositions are always returned in XML format for downstream parsing.

## Example Usage
```python
from asdrp.memory.proposition_extraction_memory import PropositionExtractionMemoryBlock
from llama_index.llms.openai import OpenAI

block = PropositionExtractionMemoryBlock(name="propositions", llm=OpenAI(model="gpt-4.1-mini"), max_propositions=10)

# Add messages (async context)
await block._aput([ChatMessage(blocks=[TextBlock(text="The sky is blue.")], additional_kwargs={})])

# Retrieve propositions
xml = await block._aget()
print(xml)
```

## Integration with Agents
- Used in `ReductiveAgent` to provide a knowledge base of extracted propositions for reasoning and reference.
- Can be combined with other memory blocks for richer agent memory.

## See Also
- [reductive_agent.py documentation](./reductive_agent.md)
- [llama_index memory documentation](https://docs.llamaindex.ai/en/stable/module_guides/memory/) 