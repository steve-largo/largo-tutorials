"""Lightweight OpenAI-compatible vLLM adapter (`BaseLlm`) for Google ADK 2.14+."""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator
from typing import Any

import httpx
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types
from pydantic import ConfigDict, Field, PrivateAttr
from typing_extensions import override

from gemma4_serving.config import ServingRuntimePolicy, ServingSettings


def _schema_to_json_dict(schema: types.Schema | dict[str, Any] | None) -> dict[str, Any]:
    """Convert a `google.genai.types.Schema` into a lowercase OpenAI JSON Schema dict."""
    if schema is None:
        return {"type": "object", "properties": {}}
    raw = schema.model_dump(exclude_none=True) if isinstance(schema, types.Schema) else dict(schema)

    def _normalize(node: Any) -> Any:
        if isinstance(node, dict):
            out: dict[str, Any] = {}
            for k, v in node.items():
                if k == "type" and isinstance(v, str):
                    out[k] = v.lower()
                else:
                    out[k] = _normalize(v)
            return out
        if isinstance(node, list):
            return [_normalize(item) for item in node]
        return node

    return _normalize(raw)


def translate_adk_request_to_openai_payload(
    llm_request: LlmRequest,
    served_model_name: str,
    policy: ServingRuntimePolicy,
) -> dict[str, Any]:
    """Translate ADK `LlmRequest` into vLLM's `/v1/chat/completions` request JSON.

    Unlike `google.adk.models.gemma_llm.Gemma` (which is restricted to Gemma 3 on Google AI
    Studio and strips `config.tools` into user-role prompt text), Gemma 4 natively processes
    `role='system'` messages and structured OpenAI `tools` declarations in vLLM.
    """
    messages: list[dict[str, Any]] = []

    if llm_request.config and llm_request.config.system_instruction:
        sys_inst = llm_request.config.system_instruction
        sys_text = sys_inst if isinstance(sys_inst, str) else str(sys_inst)
        messages.append({"role": "system", "content": sys_text})

    for content in llm_request.contents or []:
        role = content.role or "user"
        parts = content.parts or []

        fn_calls = [p.function_call for p in parts if p.function_call is not None]
        fn_resps = [p.function_response for p in parts if p.function_response is not None]
        text_chunks = [
            p.text
            for p in parts
            if p.text and not (policy.strip_thought_blocks and getattr(p, "thought", False))
        ]

        if fn_calls:
            oai_calls: list[dict[str, Any]] = []
            for idx, fc in enumerate(fn_calls):
                call_id = fc.id or f"call_{fc.name}_{idx}"
                oai_calls.append(
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": fc.name,
                            "arguments": json.dumps(fc.args or {}),
                        },
                    }
                )
            messages.append(
                {
                    "role": "assistant",
                    "content": "\n".join(text_chunks) if text_chunks else "",
                    "tool_calls": oai_calls,
                }
            )
        elif fn_resps:
            for idx, fr in enumerate(fn_resps):
                call_id = fr.id or f"call_{fr.name}_{idx}"
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": json.dumps(fr.response or {}),
                    }
                )
        else:
            mapped_role = "assistant" if role == "model" else "user"
            messages.append({"role": mapped_role, "content": "\n".join(text_chunks)})

    tools_payload: list[dict[str, Any]] = []
    if llm_request.config and llm_request.config.tools:
        for tool_item in llm_request.config.tools:
            if isinstance(tool_item, types.Tool) and tool_item.function_declarations:
                for decl in tool_item.function_declarations:
                    tools_payload.append(
                        {
                            "type": "function",
                            "function": {
                                "name": decl.name,
                                "description": decl.description or "",
                                "parameters": _schema_to_json_dict(
                                    decl.parameters or decl.parameters_json_schema
                                ),
                            },
                        }
                    )

    payload: dict[str, Any] = {
        "model": served_model_name,
        "messages": messages,
        "temperature": (
            llm_request.config.temperature
            if llm_request.config and llm_request.config.temperature is not None
            else policy.temperature
        ),
        "max_tokens": (
            llm_request.config.max_output_tokens
            if llm_request.config and llm_request.config.max_output_tokens is not None
            else policy.max_output_tokens
        ),
    }
    if tools_payload:
        payload["tools"] = tools_payload
        payload["tool_choice"] = "auto"

    if llm_request.config and llm_request.config.response_schema is not None:
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "structured_output",
                "strict": True,
                "schema": _schema_to_json_dict(llm_request.config.response_schema),
            },
        }

    return payload


class VllmOpenAiLlm(BaseLlm):
    """Direct Google ADK `BaseLlm` client for vLLM OpenAI-compatible serving endpoints."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    settings: ServingSettings
    policy: ServingRuntimePolicy = Field(default_factory=ServingRuntimePolicy)
    _http_client: httpx.AsyncClient | None = PrivateAttr(default=None)

    def __init__(
        self,
        settings: ServingSettings,
        policy: ServingRuntimePolicy | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        resolved_policy = policy or ServingRuntimePolicy()
        super().__init__(
            model=settings.served_model_name,
            settings=settings,
            policy=resolved_policy,
        )
        self._http_client = http_client

    @override
    async def generate_content_async(
        self,
        llm_request: LlmRequest,
        stream: bool = False,
    ) -> AsyncGenerator[LlmResponse, None]:
        payload = translate_adk_request_to_openai_payload(
            llm_request=llm_request,
            served_model_name=self.settings.served_model_name,
            policy=self.policy,
        )
        headers = {"Content-Type": "application/json"}
        if self.settings.vllm_api_key:
            headers["Authorization"] = f"Bearer {self.settings.vllm_api_key}"

        url = f"{self.settings.vllm_api_base}/chat/completions"
        owns_client = self._http_client is None
        client = self._http_client or httpx.AsyncClient(
            timeout=self.settings.request_timeout_seconds
        )
        try:
            resp = await client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            body = resp.json()
        finally:
            if owns_client:
                await client.aclose()

        choice = body["choices"][0]
        msg = choice["message"]
        parts: list[types.Part] = []

        if msg.get("content"):
            parts.append(types.Part.from_text(text=str(msg["content"])))

        for tc in msg.get("tool_calls") or []:
            fn = tc["function"]
            raw_args = fn.get("arguments", "{}")
            parsed_args = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args)
            fc = types.FunctionCall(
                id=tc.get("id"),
                name=fn["name"],
                args=parsed_args,
            )
            parts.append(types.Part(function_call=fc))

        usage = body.get("usage") or {}
        usage_meta = types.GenerateContentResponseUsageMetadata(
            prompt_token_count=int(usage.get("prompt_tokens", 0)),
            candidates_token_count=int(usage.get("completion_tokens", 0)),
            total_token_count=int(usage.get("total_tokens", 0)),
        )

        yield LlmResponse(
            model_version=body.get("model", self.settings.served_model_name),
            content=types.Content(role="model", parts=parts),
            turn_complete=True,
            finish_reason=types.FinishReason.STOP,
            usage_metadata=usage_meta,
        )
