"""
Azure API clients for Afri-Kenya-MT-Bench.

Supported models:
- GPT-5.6-sol
- Claude Sonnet 5
"""

import os
import random
import time
from dataclasses import dataclass
from typing import Optional

from openai import AzureOpenAI
from anthropic import AnthropicFoundry


GPT_ENDPOINT = (
    "https://adelani-multilingual-lab.cognitiveservices.azure.com/"
)

CLAUDE_ENDPOINT = (
    "https://adelani-multilingual-lab.services.ai.azure.com/anthropic"
)

MODEL_NAMES = {
    "gpt": "gpt-5.6-sol",
    "claude": "claude-sonnet-5",
}


@dataclass
class TranslationResponse:
    text: str
    model: str
    latency_seconds: float
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    finish_reason: Optional[str] = None


class AzureMTClient:
    """Unified interface for GPT and Claude translation."""

    def __init__(
        self,
        model: str,
        max_output_tokens: int = 4096,
        max_retries: int = 3,
        timeout: float = 120.0,
    ):
        if model not in MODEL_NAMES:
            raise ValueError(
                f"Unknown model: {model}. "
                f"Choose from {list(MODEL_NAMES)}"
            )

        api_key = os.getenv("AZURE_API_KEY")

        if not api_key:
            raise RuntimeError(
                "AZURE_API_KEY is not configured."
            )

        self.model = model
        self.deployment = MODEL_NAMES[model]
        self.max_output_tokens = max_output_tokens
        self.max_retries = max_retries

        if model == "gpt":
            self.client = AzureOpenAI(
                api_version="2024-12-01-preview",
                azure_endpoint=GPT_ENDPOINT,
                api_key=api_key,
                timeout=timeout,
                max_retries=0,
            )
        else:
            self.client = AnthropicFoundry(
                api_key=api_key,
                base_url=CLAUDE_ENDPOINT,
                timeout=timeout,
                max_retries=0,
            )

    def _request(self, prompt: str) -> TranslationResponse:
        start = time.perf_counter()

        if self.model == "gpt":
            response = self.client.chat.completions.create(
                model=self.deployment,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_completion_tokens=self.max_output_tokens,
            )

            text = response.choices[0].message.content or ""
            usage = response.usage

            return TranslationResponse(
                text=text.strip(),
                model=self.deployment,
                latency_seconds=time.perf_counter() - start,
                input_tokens=(
                    usage.prompt_tokens if usage else None
                ),
                output_tokens=(
                    usage.completion_tokens if usage else None
                ),
                finish_reason=response.choices[0].finish_reason,
            )

        response = self.client.messages.create(
            model=self.deployment,
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=self.max_output_tokens,
        )

        text = "".join(
            block.text
            for block in response.content
            if block.type == "text"
        )

        usage = response.usage

        return TranslationResponse(
            text=text.strip(),
            model=self.deployment,
            latency_seconds=time.perf_counter() - start,
            input_tokens=(
                usage.input_tokens if usage else None
            ),
            output_tokens=(
                usage.output_tokens if usage else None
            ),
            finish_reason=response.stop_reason,
        )

    def translate(self, prompt: str) -> TranslationResponse:
        """Translate with retries for transient API failures."""

        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty.")

        for attempt in range(self.max_retries + 1):
            try:
                result = self._request(prompt)

                if not result.text:
                    raise ValueError(
                        f"Empty translation from {self.deployment}; "
                        f"finish_reason={result.finish_reason}"
                    )

                return result

            except Exception as exc:
                status = getattr(exc, "status_code", None)

                retryable = (
                    status in (408, 409, 429, 500, 502, 503, 504)
                    or type(exc).__name__ in (
                        "APIConnectionError",
                        "APITimeoutError",
                    )
                )

                if not retryable or attempt >= self.max_retries:
                    raise

                delay = min(2 ** attempt, 30)
                delay += random.uniform(0, 1)
                time.sleep(delay)

        raise RuntimeError("Maximum retries exceeded.")

    def close(self):
        self.client.close()
