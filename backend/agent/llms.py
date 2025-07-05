import os
from dotenv import load_dotenv
from pydantic import SecretStr
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

load_dotenv()

openai_api_key = SecretStr(os.getenv("OPENAI_API_KEY") or "")
anthropic_api_key = SecretStr(os.getenv("ANTHROPIC_API_KEY") or "")

if not openai_api_key:
    raise ValueError("OPENAI_API_KEY is not set")
if not anthropic_api_key:
    raise ValueError("ANTHROPIC_API_KEY is not set")


# ===========================================
#              Model selection
# ===========================================
openai_chat_model_default = "gpt-4o"
# openai_chat_model_default = "gpt-4.1"

anthropic_chat_model_default = "claude-3-5-sonnet-latest"
# anthropic_chat_model_default = "claude-3-7-sonnet-latest"
# anthropic_chat_model_default = "claude-sonnet-4-20250514"

openai_chat_model_small = "gpt-4o-mini"

anthropic_chat_model_small = "claude-3-5-haiku-latest"

openai_reasoning_model_default = "o4-mini"

# openai_reasoning_model_large = "o3"
openai_reasoning_model_large = "o3-mini"


# ===========================================
#             Non-Reasoning Models
# ===========================================
chat_model_anthropic_first = ChatAnthropic(
    model_name=anthropic_chat_model_default,
    api_key=anthropic_api_key,
    temperature=0.5,
    timeout=120,
    stop=None,
).with_fallbacks(
    [
        ChatOpenAI(
            model=openai_chat_model_default,
            temperature=0.1,
            api_key=openai_api_key,
        ),  # switch provider
        ChatOpenAI(
            model=openai_reasoning_model_default,
            temperature=None,
            api_key=openai_api_key,
        ),  # try with reasoning model
    ]
)

chat_model_openai_first = ChatOpenAI(
    model=openai_chat_model_default,
    api_key=openai_api_key,
    temperature=0.5,
).with_fallbacks(
    [
        ChatAnthropic(
            model_name=anthropic_chat_model_default,
            api_key=anthropic_api_key,
            temperature=0.1,
            timeout=120,
            stop=None,
        ),  # switch provider
        ChatOpenAI(
            model=openai_reasoning_model_default,
            temperature=None,
            api_key=openai_api_key,
        ),  # try with reasoning model
    ]
)


# ===========================================
#                Small Models
# ===========================================

chat_model_anthropic_first_small = ChatAnthropic(
    model_name=anthropic_chat_model_small,
    temperature=0.7,
    api_key=anthropic_api_key,
    timeout=120,
    stop=None,
).with_fallbacks(
    [
        ChatOpenAI(
            model=openai_chat_model_small,
            temperature=0.1,
            api_key=openai_api_key,
        ),  # switch provider
    ]
)

chat_model_openai_first_small = ChatOpenAI(
    model=openai_chat_model_small,
    temperature=None,
    api_key=openai_api_key,
).with_fallbacks(
    [
        ChatAnthropic(
            model_name=anthropic_chat_model_small,
            api_key=anthropic_api_key,
            temperature=0.1,
            timeout=120,
            stop=None,
        ),  # switch provider
    ]
)

# ===========================================
#              Reasoning Models
# ===========================================


reasoning_model = ChatOpenAI(
    model=openai_reasoning_model_default,
    temperature=None,
    # reasoning_effort="high",
    api_key=openai_api_key,
).with_fallbacks(
    [
        ChatOpenAI(
            model=openai_reasoning_model_default,
            temperature=None,
            # reasoning_effort="high",
            api_key=openai_api_key,
        ),  # try one more time
    ]
)

reasoning_model_large = ChatOpenAI(
    model=openai_reasoning_model_large,
    temperature=None,
    # reasoning_effort="high",
    api_key=openai_api_key,
).with_fallbacks(
    [
        ChatOpenAI(
            model=openai_reasoning_model_large,
            temperature=None,
            # reasoning_effort="high",
            api_key=openai_api_key,
        ),  # try one more time
    ]
)


# ===========================================
#              Fixed Models
# ===========================================
o3 = ChatOpenAI(
    model="o3",
    temperature=None,
    api_key=openai_api_key,
)

claude_4_opus = ChatAnthropic(
    model_name="claude-opus-4-20250514",
    api_key=anthropic_api_key,
    timeout=120,
    stop=None,
    # temperature=1,  # When thikning mode is on, temp should be 1
    # max_tokens_to_sample=8000,  # This should be larger then the budget tokens
    # thinking={"type": "enabled", "budget_tokens": 5000},
)
