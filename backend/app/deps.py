from __future__ import annotations

import os
from collections.abc import Generator
from typing import Annotated, Optional

from fastapi import Depends, Header, HTTPException
from sqlalchemy.orm import Session

from app import crud_settings
from app.database import get_db
from onemix.services.llm_router import (
    DEFAULT_PROVIDER,
    LLMRuntime,
    get_spec,
    reset_runtime,
    set_runtime,
)


def get_dashscope_api_key_optional(
    *,
    db: Session,
    x_dashscope_key: Optional[str],
    authorization: Optional[str],
) -> str | None:
    if x_dashscope_key and x_dashscope_key.strip():
        return x_dashscope_key.strip()
    if authorization and authorization.lower().startswith("bearer "):
        tok = authorization[7:].strip()
        if tok:
            return tok
    env = (os.environ.get("DASHSCOPE_API_KEY") or "").strip()
    if env:
        return env
    return crud_settings.get_dashscope_api_key(db)


def resolve_api_key(
    db: Annotated[Session, Depends(get_db)],
    x_dashscope_key: Annotated[Optional[str], Header(alias="X-DashScope-Key")] = None,
    authorization: Annotated[Optional[str], Header()] = None,
) -> str:
    stored = get_dashscope_api_key_optional(
        db=db, x_dashscope_key=x_dashscope_key, authorization=authorization
    )
    if stored:
        return stored
    raise HTTPException(
        status_code=401,
        detail=(
            "缺少 API Key：请使用请求头 X-DashScope-Key 或 Authorization: Bearer <key>，"
            "或设置环境变量 DASHSCOPE_API_KEY，或通过 PUT /api/settings/dashscope-key 写入服务端 SQLite。"
        ),
    )


def get_ark_api_key_optional(
    *,
    db: Session,
    x_ark_key: Optional[str],
) -> str | None:
    if x_ark_key and x_ark_key.strip():
        return x_ark_key.strip()
    env = (os.environ.get("ARK_API_KEY") or "").strip()
    if env:
        return env
    return crud_settings.get_ark_api_key(db)


def resolve_ark_api_key(
    db: Annotated[Session, Depends(get_db)],
    x_ark_key: Annotated[Optional[str], Header(alias="X-Ark-Key")] = None,
) -> str:
    key = get_ark_api_key_optional(db=db, x_ark_key=x_ark_key)
    if key:
        return key
    raise HTTPException(
        status_code=401,
        detail=(
            "缺少 ARK API Key：请使用请求头 X-Ark-Key，"
            "或设置环境变量 ARK_API_KEY，或通过 PUT /api/settings/ark-key 写入服务端 SQLite。"
        ),
    )


def resolve_provider_key(
    db: Session,
    provider_id: str,
    *,
    x_llm_key: Optional[str] = None,
    x_dashscope_key: Optional[str] = None,
    x_ark_key: Optional[str] = None,
    authorization: Optional[str] = None,
) -> str | None:
    if x_llm_key and x_llm_key.strip():
        return x_llm_key.strip()
    spec = get_spec(provider_id)
    if spec.id == "dashscope":
        return get_dashscope_api_key_optional(
            db=db, x_dashscope_key=x_dashscope_key, authorization=authorization
        )
    if spec.id == "volcengine":
        return get_ark_api_key_optional(db=db, x_ark_key=x_ark_key)
    env = (os.environ.get(spec.env_key) or "").strip()
    if env:
        return env
    return crud_settings.get_value(db, spec.setting_key)


def build_llm_runtime(
    db: Session,
    *,
    x_llm_provider: Optional[str] = None,
    x_llm_model: Optional[str] = None,
    x_llm_base_url: Optional[str] = None,
    x_llm_key: Optional[str] = None,
    x_dashscope_key: Optional[str] = None,
    x_ark_key: Optional[str] = None,
    authorization: Optional[str] = None,
) -> LLMRuntime:
    provider_id = (
        (x_llm_provider or "").strip()
        or (os.environ.get("ONEMIX_LLM_PROVIDER") or "").strip()
        or crud_settings.get_value(db, "llm_provider")
        or DEFAULT_PROVIDER
    )
    spec = get_spec(provider_id)
    model = (
        (x_llm_model or "").strip()
        or (os.environ.get("ONEMIX_LLM_MODEL") or "").strip()
        or crud_settings.get_value(db, "llm_model")
        or spec.default_model
    )
    base_url = (
        (x_llm_base_url or "").strip()
        or (os.environ.get("ONEMIX_LLM_BASE_URL") or "").strip()
        or crud_settings.get_value(db, "llm_base_url")
        or spec.base_url
    )
    api_key = resolve_provider_key(
        db,
        spec.id,
        x_llm_key=x_llm_key,
        x_dashscope_key=x_dashscope_key,
        x_ark_key=x_ark_key,
        authorization=authorization,
    )
    if not api_key and spec.allow_empty_key:
        api_key = spec.dummy_key
    if not api_key:
        raise HTTPException(
            status_code=401,
            detail=(
                f"缺少 {spec.name} API Key：请在设置中保存该平台 Key，"
                f"或使用请求头 X-LLM-Key，或配置环境变量 {spec.env_key}。"
            ),
        )
    if spec.requires_base_url and not (base_url or "").strip():
        raise HTTPException(400, f"{spec.name} 需要填写 OpenAI 兼容 Base URL。")
    if not (model or "").strip():
        raise HTTPException(400, f"{spec.name} 需要填写模型名。")
    return LLMRuntime(
        provider=spec.id,
        model=model,
        api_key=api_key,
        base_url=base_url,
    )


def bind_llm_runtime(
    db: Annotated[Session, Depends(get_db)],
    x_llm_provider: Annotated[Optional[str], Header(alias="X-LLM-Provider")] = None,
    x_llm_model: Annotated[Optional[str], Header(alias="X-LLM-Model")] = None,
    x_llm_base_url: Annotated[Optional[str], Header(alias="X-LLM-Base-Url")] = None,
    x_llm_key: Annotated[Optional[str], Header(alias="X-LLM-Key")] = None,
    x_dashscope_key: Annotated[Optional[str], Header(alias="X-DashScope-Key")] = None,
    x_ark_key: Annotated[Optional[str], Header(alias="X-Ark-Key")] = None,
    authorization: Annotated[Optional[str], Header()] = None,
) -> Generator[LLMRuntime, None, None]:
    runtime = build_llm_runtime(
        db,
        x_llm_provider=x_llm_provider,
        x_llm_model=x_llm_model,
        x_llm_base_url=x_llm_base_url,
        x_llm_key=x_llm_key,
        x_dashscope_key=x_dashscope_key,
        x_ark_key=x_ark_key,
        authorization=authorization,
    )
    token = set_runtime(runtime)
    try:
        yield runtime
    finally:
        reset_runtime(token)
