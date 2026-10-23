"""统一文本 LLM 调度：按平台路由到 OpenAI 兼容 Chat Completions。

市面主流云厂商（百炼、OpenAI、DeepSeek、Kimi、智谱、方舟、OpenRouter 等）
以及本地 Ollama 都提供该协议。未内置的平台走 ``custom`` + base_url 即可接入。
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass
from typing import Any, Iterator, Optional

from openai import OpenAI

from onemix.services import prompt_templates as prompts

logger = logging.getLogger(__name__)

DEFAULT_PROVIDER = "dashscope"
DEFAULT_MODEL = "qwen-max"


@dataclass(frozen=True)
class LLMProviderSpec:
    id: str
    name: str
    base_url: str
    default_model: str
    models: tuple[str, ...]
    env_key: str
    setting_key: str
    requires_base_url: bool = False
    allow_empty_key: bool = False
    dummy_key: str = "sk-placeholder"


@dataclass(frozen=True)
class LLMRuntime:
    provider: str
    model: str
    api_key: str
    base_url: str


PROVIDERS: dict[str, LLMProviderSpec] = {
    "dashscope": LLMProviderSpec(
        id="dashscope",
        name="阿里百炼",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        default_model=DEFAULT_MODEL,
        models=("qwen-max", "qwen-plus", "qwen-turbo", "qwen-long", "qwen-max-latest"),
        env_key="DASHSCOPE_API_KEY",
        setting_key="dashscope_api_key",
    ),
    "openai": LLMProviderSpec(
        id="openai",
        name="OpenAI",
        base_url="https://api.openai.com/v1",
        default_model="gpt-4o-mini",
        models=("gpt-4.1", "gpt-4o", "gpt-4o-mini", "o4-mini"),
        env_key="OPENAI_API_KEY",
        setting_key="llm_openai_api_key",
    ),
    "azure": LLMProviderSpec(
        id="azure",
        name="Azure OpenAI",
        base_url="",
        default_model="gpt-4o-mini",
        models=("gpt-4o", "gpt-4o-mini", "gpt-4.1"),
        env_key="AZURE_OPENAI_API_KEY",
        setting_key="llm_azure_api_key",
        requires_base_url=True,
    ),
    "anthropic": LLMProviderSpec(
        id="anthropic",
        name="Anthropic",
        base_url="https://api.anthropic.com/v1/",
        default_model="claude-sonnet-4-5",
        models=("claude-sonnet-4-5", "claude-opus-4-1", "claude-haiku-4-5"),
        env_key="ANTHROPIC_API_KEY",
        setting_key="llm_anthropic_api_key",
    ),
    "gemini": LLMProviderSpec(
        id="gemini",
        name="Google Gemini",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        default_model="gemini-2.5-flash",
        models=("gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.0-flash"),
        env_key="GEMINI_API_KEY",
        setting_key="llm_gemini_api_key",
    ),
    "deepseek": LLMProviderSpec(
        id="deepseek",
        name="DeepSeek",
        base_url="https://api.deepseek.com",
        default_model="deepseek-chat",
        models=("deepseek-chat", "deepseek-reasoner"),
        env_key="DEEPSEEK_API_KEY",
        setting_key="llm_deepseek_api_key",
    ),
    "moonshot": LLMProviderSpec(
        id="moonshot",
        name="月之暗面 Kimi",
        base_url="https://api.moonshot.cn/v1",
        default_model="kimi-k2-turbo-preview",
        models=("kimi-k2-turbo-preview", "kimi-k2-0711-preview", "moonshot-v1-auto"),
        env_key="MOONSHOT_API_KEY",
        setting_key="llm_moonshot_api_key",
    ),
    "zhipu": LLMProviderSpec(
        id="zhipu",
        name="智谱 GLM",
        base_url="https://open.bigmodel.cn/api/paas/v4",
        default_model="glm-4-flash",
        models=("glm-4-plus", "glm-4-air", "glm-4-flash"),
        env_key="ZHIPU_API_KEY",
        setting_key="llm_zhipu_api_key",
    ),
    "volcengine": LLMProviderSpec(
        id="volcengine",
        name="火山方舟",
        base_url="https://ark.cn-beijing.volces.com/api/v3",
        default_model="",
        models=(),
        env_key="ARK_API_KEY",
        setting_key="ark_api_key",
        requires_base_url=False,
    ),
    "siliconflow": LLMProviderSpec(
        id="siliconflow",
        name="硅基流动",
        base_url="https://api.siliconflow.cn/v1",
        default_model="deepseek-ai/DeepSeek-V3",
        models=("deepseek-ai/DeepSeek-V3", "Qwen/Qwen2.5-72B-Instruct", "Pro/THUDM/glm-4-9b-chat"),
        env_key="SILICONFLOW_API_KEY",
        setting_key="llm_siliconflow_api_key",
    ),
    "openrouter": LLMProviderSpec(
        id="openrouter",
        name="OpenRouter",
        base_url="https://openrouter.ai/api/v1",
        default_model="openai/gpt-4o-mini",
        models=("openai/gpt-4o-mini", "anthropic/claude-sonnet-4.5", "google/gemini-2.5-flash"),
        env_key="OPENROUTER_API_KEY",
        setting_key="llm_openrouter_api_key",
    ),
    "minimax": LLMProviderSpec(
        id="minimax",
        name="MiniMax",
        base_url="https://api.minimax.chat/v1",
        default_model="MiniMax-Text-01",
        models=("MiniMax-Text-01", "abab6.5s-chat"),
        env_key="MINIMAX_API_KEY",
        setting_key="llm_minimax_api_key",
    ),
    "groq": LLMProviderSpec(
        id="groq",
        name="Groq",
        base_url="https://api.groq.com/openai/v1",
        default_model="llama-3.3-70b-versatile",
        models=("llama-3.3-70b-versatile", "openai/gpt-oss-120b"),
        env_key="GROQ_API_KEY",
        setting_key="llm_groq_api_key",
    ),
    "mistral": LLMProviderSpec(
        id="mistral",
        name="Mistral",
        base_url="https://api.mistral.ai/v1",
        default_model="mistral-small-latest",
        models=("mistral-large-latest", "mistral-small-latest", "ministral-8b-latest"),
        env_key="MISTRAL_API_KEY",
        setting_key="llm_mistral_api_key",
    ),
    "ollama": LLMProviderSpec(
        id="ollama",
        name="Ollama（本地）",
        base_url="http://127.0.0.1:11434/v1",
        default_model="qwen2.5",
        models=("qwen2.5", "llama3.1", "deepseek-r1"),
        env_key="OLLAMA_API_KEY",
        setting_key="llm_ollama_api_key",
        allow_empty_key=True,
        dummy_key="ollama",
    ),
    "custom": LLMProviderSpec(
        id="custom",
        name="自定义 OpenAI 兼容",
        base_url="",
        default_model="",
        models=(),
        env_key="ONEMIX_LLM_API_KEY",
        setting_key="llm_custom_api_key",
        requires_base_url=True,
    ),
}

_runtime: ContextVar[Optional[LLMRuntime]] = ContextVar("onemix_llm_runtime", default=None)


def list_providers() -> list[LLMProviderSpec]:
    return list(PROVIDERS.values())


def get_spec(provider_id: str) -> LLMProviderSpec:
    pid = (provider_id or DEFAULT_PROVIDER).strip().lower()
    if pid == "ark":
        pid = "volcengine"
    spec = PROVIDERS.get(pid)
    if spec is None:
        raise RuntimeError(f"未知 LLM 平台: {provider_id}。可用：{', '.join(PROVIDERS)}")
    return spec


def current_runtime() -> Optional[LLMRuntime]:
    return _runtime.get()


def set_runtime(runtime: LLMRuntime) -> Token:
    return _runtime.set(runtime)


def reset_runtime(token: Token) -> None:
    _runtime.reset(token)


@contextmanager
def use_runtime(runtime: LLMRuntime) -> Iterator[LLMRuntime]:
    token = set_runtime(runtime)
    try:
        yield runtime
    finally:
        reset_runtime(token)


def _strip(value: Optional[str]) -> str:
    return (value or "").strip()


def load_runtime_from_settings(
    *,
    fallback_api_key: str = "",
    fallback_model: str = "",
) -> LLMRuntime:
    """从环境变量 / SQLite 读取当前规划模型；无库时回退百炼。"""
    provider_id = _strip(os.environ.get("ONEMIX_LLM_PROVIDER"))
    model = _strip(os.environ.get("ONEMIX_LLM_MODEL"))
    base_url = _strip(os.environ.get("ONEMIX_LLM_BASE_URL"))
    stored_key = ""

    db = None
    try:
        from app.database import SessionLocal
        from app import crud_settings

        db = SessionLocal()
        provider_id = provider_id or _strip(crud_settings.get_value(db, "llm_provider"))
        spec = get_spec(provider_id or DEFAULT_PROVIDER)
        model = model or _strip(crud_settings.get_value(db, "llm_model")) or spec.default_model
        base_url = base_url or _strip(crud_settings.get_value(db, "llm_base_url")) or spec.base_url
        stored_key = _strip(os.environ.get(spec.env_key)) or _strip(
            crud_settings.get_value(db, spec.setting_key)
        )
    except Exception:  # noqa: BLE001
        spec = get_spec(provider_id or DEFAULT_PROVIDER)
        model = model or spec.default_model
        base_url = base_url or spec.base_url
        stored_key = _strip(os.environ.get(spec.env_key))
    finally:
        if db is not None:
            db.close()

    spec = get_spec(provider_id or DEFAULT_PROVIDER)
    model = model or fallback_model or spec.default_model
    base_url = base_url or spec.base_url
    api_key = stored_key
    if not api_key and spec.id == "dashscope":
        api_key = _strip(fallback_api_key)
    if not api_key and spec.allow_empty_key:
        api_key = spec.dummy_key
    if not api_key:
        raise RuntimeError(
            f"缺少 {spec.name} API Key：请在设置中保存，或配置环境变量 {spec.env_key}。"
        )
    if spec.requires_base_url and not base_url:
        raise RuntimeError(f"{spec.name} 需要填写 OpenAI 兼容 Base URL。")
    if not model:
        raise RuntimeError(f"{spec.name} 需要填写模型名（方舟请填推理接入点 ID）。")
    return LLMRuntime(provider=spec.id, model=model, api_key=api_key, base_url=base_url)


def _build_messages(*, prompt: str, model: str) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    if "qwen-max" in (model or "").lower():
        messages.append(
            {"role": "system", "content": prompts.QWEN_MAX_PROMPT_SYSTEM_CONSTRAINT}
        )
    messages.append({"role": "user", "content": prompt})
    return messages


def _extract_text(resp: Any) -> str:
    choices = getattr(resp, "choices", None) or []
    if not choices:
        raise RuntimeError("规划模型无输出")
    msg = getattr(choices[0], "message", None)
    content = getattr(msg, "content", None) if msg else None
    if isinstance(content, str) and content.strip():
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text") or ""))
            elif isinstance(block, str):
                parts.append(block)
        text = "\n".join(x for x in parts if x).strip()
        if text:
            return text
    raise RuntimeError("无法解析规划模型输出")


def chat_text(
    *,
    prompt: str,
    fallback_api_key: str = "",
    fallback_model: str = "",
) -> str:
    runtime = current_runtime() or load_runtime_from_settings(
        fallback_api_key=fallback_api_key,
        fallback_model=fallback_model,
    )
    client = OpenAI(
        api_key=runtime.api_key,
        base_url=runtime.base_url,
        timeout=120.0,
    )
    logger.info("LLM chat provider=%s model=%s", runtime.provider, runtime.model)
    try:
        resp = client.chat.completions.create(
            model=runtime.model,
            messages=_build_messages(prompt=prompt, model=runtime.model),
        )
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"LLM 调用失败[{runtime.provider}/{runtime.model}]: {exc}"
        ) from exc
    return _extract_text(resp)
