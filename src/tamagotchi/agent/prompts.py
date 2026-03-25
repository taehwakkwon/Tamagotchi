"""Dynamic system prompt generation based on user profile and growth state."""

from __future__ import annotations

from tamagotchi.growth.personality import PersonalityTraits
from tamagotchi.growth.state import LEVELS, GrowthManager
from tamagotchi.memory.profile import UserProfile

SYSTEM_PROMPT_TEMPLATE = """\
당신은 사용자의 개인 AI 비서 "다마고치"입니다.
사용자와의 대화를 통해 취향, 습관, 관심사를 학습하고, 이를 기반으로 개인화된 도움을 제공합니다.

## 성장 상태
레벨 {level}: {level_name}
성격: {personality_summary}
총 대화 횟수: {total_conversations}회

## 사용자 프로필
{profile_text}
{episodic_context}
{patterns_context}

## 성격 지침
{personality_instructions}

## 행동 지침
- 사용자의 프로필 정보를 자연스럽게 활용하세요. "프로필에 따르면..." 같은 표현은 피하세요.
- 사용자가 새로운 취향이나 정보를 언급하면 자연스럽게 대화에 반영하세요.
- {behavior_instruction}
- 한국어로 대화하되, 사용자가 다른 언어를 쓰면 맞춰주세요.
- 다마고치답게 사용자와의 유대감을 보여주세요.
"""


def build_system_prompt(
    profile: UserProfile,
    growth: GrowthManager,
    episodic_context: str = "",
    patterns_context: str = "",
    personality: PersonalityTraits | None = None,
) -> str:
    level_info = LEVELS.get(profile.growth_level, LEVELS[1])
    state = growth.get_state()

    if profile.growth_level < 3:
        behavior = "아직 사용자에 대해 배우는 중입니다. 적극적으로 질문하고 취향을 파악하세요."
    elif profile.growth_level < 4:
        behavior = "학습한 선호도를 바탕으로 자연스럽게 추천을 시작하세요."
    else:
        behavior = "사용자가 요청하지 않아도 관심사에 맞는 컨텐츠나 정보를 선제적으로 추천하세요."

    if personality:
        personality_summary = personality.summary()
        personality_instructions = personality.to_prompt()
    else:
        personality_summary = "아직 발달 중"
        personality_instructions = "균형 잡힌 중립적 톤으로 대화하세요."

    return SYSTEM_PROMPT_TEMPLATE.format(
        level=profile.growth_level,
        level_name=level_info["name"],
        total_conversations=state["total_conversations"],
        profile_text=profile.to_prompt(),
        episodic_context=episodic_context,
        patterns_context=patterns_context,
        behavior_instruction=behavior,
        personality_summary=personality_summary,
        personality_instructions=personality_instructions,
    )
