"""Extract structured preferences from conversation using LLM."""

from __future__ import annotations

import json
from typing import Any

EXTRACTION_PROMPT = """\
아래 대화에서 사용자의 선호도, 취향, 습관, 관심사를 추출하세요.

대화:
{conversation}

다음 JSON 배열 형식으로만 응답하세요. 추출할 것이 없으면 빈 배열 []을 반환하세요.
각 항목:
- category: 카테고리 (food, shopping, entertainment, lifestyle, work, content, schedule, general 등)
- key: 구체적 항목 (예: "커피", "매운_음식", "나이키", "SF_영화")
- value: 선호도 설명 (예: "좋아함", "싫어함", "매주 수요일 저녁")
- confidence: 확신도 0.0~1.0 (직접 언급 = 0.9~1.0, 맥락상 추론 = 0.5~0.8)

중요:
- 일반적인 대화 내용이 아닌, 사용자 개인의 선호도/취향/습관만 추출
- "좋아해", "싫어해", "자주 ~해", "항상 ~해" 같은 표현에 주목
- 이미 알려진 보편적 사실은 제외 (예: "비가 오면 우산이 필요해" ← 이런 건 제외)
"""


def extract_preferences(
    client: Any,
    messages: list[dict[str, str]],
) -> list[dict]:
    """Extract user preferences from a conversation.

    Args:
        client: Either an LLMClient instance or an anthropic.Anthropic client (legacy).
        messages: Conversation messages.
    """
    conversation_text = "\n".join(
        f"{'사용자' if m['role'] == 'user' else 'AI'}: {m['content']}"
        for m in messages
    )

    prompt = EXTRACTION_PROMPT.format(conversation=conversation_text)

    # Support both LLMClient and raw Anthropic client
    from tamagotchi.llm import LLMClient

    if isinstance(client, LLMClient):
        text = client.extract(prompt)
    else:
        # Legacy: raw anthropic.Anthropic client
        from tamagotchi.config import EXTRACTION_MODEL

        response = client.messages.create(
            model=EXTRACTION_MODEL,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.content[0].text

    text = text.strip()

    # Parse JSON from response (handle markdown code blocks)
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1])

    try:
        result = json.loads(text)
        if isinstance(result, list):
            return [
                p
                for p in result
                if all(k in p for k in ("category", "key", "value"))
            ]
    except json.JSONDecodeError:
        pass

    return []
