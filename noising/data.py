from __future__ import annotations

from typing import Any, Dict, List, Tuple

from datasets import load_dataset


SYSTEM_PROMPT = {"role": "system", "content": "You are a helpful assistant."}


def _first_user_message(seq: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    for m in seq:
        if isinstance(m, dict) and m.get("role") == "user" and isinstance(m.get("content"), str):
            return [{"role": "user", "content": m["content"]}]
    # If no explicit user role, try first item with content
    for m in seq:
        if isinstance(m, dict) and isinstance(m.get("content"), str):
            return [{"role": "user", "content": m["content"]}]
    return [{"role": "user", "content": ""}]


def _extract_messages(example: Dict[str, Any]) -> List[Dict[str, str]]:
    # Prefer the 'conversation' field used in lmsys/lmsys-chat-1m
    if "conversation" in example and isinstance(example["conversation"], list):
        return _first_user_message(example["conversation"])  # only first user message
    # Support alternate naming
    if "conversations" in example and isinstance(example["conversations"], list):
        return _first_user_message(example["conversations"])  # only first user message
    if "messages" in example and isinstance(example["messages"], list):
        return _first_user_message(example["messages"])  # only first user message
    if "prompt" in example and isinstance(example["prompt"], str):
        return [{"role": "user", "content": example["prompt"]}]
    # Fallback: pick the first string field as a user prompt
    for _, v in example.items():
        if isinstance(v, str) and v.strip():
            return [{"role": "user", "content": v}]
    return [{"role": "user", "content": ""}]


def load_prompt_messages(dataset_name: str, num_prompts: int) -> List[List[Dict[str, str]]]:
    split = "train"
    ds = load_dataset(dataset_name, split=split)
    ds = ds.shuffle(seed=42)
    if num_prompts is not None:
        ds = ds.select(range(num_prompts))
    messages_list: List[List[Dict[str, str]]] = []
    for ex in ds:
        msgs = _extract_messages(ex)
        msgs = [SYSTEM_PROMPT] + msgs
        messages_list.append(msgs)
    return messages_list


def apply_chat_templates(tokenizer, messages_list: List[List[Dict[str, str]]], add_generation_prompt: bool = True) -> List[str]:
    prompts: List[str] = []
    for msgs in messages_list:
        text = tokenizer.apply_chat_template(
            msgs,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
        prompts.append(text)
    return prompts
