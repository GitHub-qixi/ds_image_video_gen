"""服务端设置（SQLite）：LLM 平台、DashScope / ARK API Key 等。"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app import crud_settings
from app.database import get_db
from app.schemas import (
    ArkKeyIn,
    DashScopeKeyIn,
    LLMConfigIn,
    LLMKeyIn,
    LLMProviderOut,
    SettingsOut,
)
from onemix.services.llm_router import DEFAULT_MODEL, DEFAULT_PROVIDER, get_spec, list_providers

router = APIRouter(prefix="/settings", tags=["settings"])


def _preview_key(raw: str | None) -> str | None:
    if not raw:
        return None
    s = raw.strip()
    if not s:
        return None
    return ("****" + s[-4:]) if len(s) >= 4 else "****"


def _provider_key(db: Session, setting_key: str) -> str | None:
    return crud_settings.get_value(db, setting_key)


def _build_provider_outs(db: Session) -> list[LLMProviderOut]:
    out: list[LLMProviderOut] = []
    for spec in list_providers():
        raw = _provider_key(db, spec.setting_key)
        out.append(
            LLMProviderOut(
                id=spec.id,
                name=spec.name,
                default_model=spec.default_model,
                models=list(spec.models),
                base_url=spec.base_url,
                requires_base_url=spec.requires_base_url,
                has_key=bool(raw) or spec.allow_empty_key,
                key_preview=_preview_key(raw),
                env_key=spec.env_key,
            )
        )
    return out


@router.get("", response_model=SettingsOut)
def read_settings(db: Session = Depends(get_db)) -> SettingsOut:
    ds_row = crud_settings.get_dashscope_record(db)
    ark_row = crud_settings.get_ark_record(db)

    ds_preview = None
    ds_updated = None
    has_ds = bool(ds_row and (ds_row.value or "").strip())
    if has_ds and ds_row:
        raw = ds_row.value.strip()
        ds_preview = _preview_key(raw)
        if ds_row.updated_at:
            ds_updated = ds_row.updated_at.isoformat()

    ark_preview = None
    ark_updated = None
    has_ark = bool(ark_row and (ark_row.value or "").strip())
    if has_ark and ark_row:
        raw = ark_row.value.strip()
        ark_preview = _preview_key(raw)
        if ark_row.updated_at:
            ark_updated = ark_row.updated_at.isoformat()

    llm_provider = crud_settings.get_value(db, "llm_provider") or DEFAULT_PROVIDER
    try:
        spec = get_spec(llm_provider)
    except RuntimeError:
        spec = get_spec(DEFAULT_PROVIDER)
        llm_provider = spec.id
    llm_model = crud_settings.get_value(db, "llm_model") or spec.default_model or DEFAULT_MODEL
    llm_base_url = crud_settings.get_value(db, "llm_base_url") or None

    return SettingsOut(
        has_dashscope_key=has_ds,
        dashscope_key_preview=ds_preview,
        dashscope_key_updated_at=ds_updated,
        has_ark_key=has_ark,
        ark_key_preview=ark_preview,
        ark_key_updated_at=ark_updated,
        llm_provider=llm_provider,
        llm_model=llm_model,
        llm_base_url=llm_base_url,
        llm_providers=_build_provider_outs(db),
    )


@router.put("/llm")
def save_llm_config(body: LLMConfigIn, db: Session = Depends(get_db)) -> dict[str, bool]:
    try:
        spec = get_spec(body.provider)
    except RuntimeError as exc:
        raise HTTPException(400, str(exc)) from exc
    model = (body.model or "").strip() or spec.default_model
    base_url = (body.base_url or "").strip()
    if spec.requires_base_url and not base_url:
        raise HTTPException(400, f"{spec.name} 需要填写 OpenAI 兼容 Base URL。")
    if not model:
        raise HTTPException(400, f"{spec.name} 需要填写模型名。")
    crud_settings.set_value(db, "llm_provider", spec.id)
    crud_settings.set_value(db, "llm_model", model)
    if base_url:
        crud_settings.set_value(db, "llm_base_url", base_url)
    else:
        crud_settings.delete_value(db, "llm_base_url")
    return {"ok": True}


@router.put("/llm-key")
def save_llm_key(body: LLMKeyIn, db: Session = Depends(get_db)) -> dict[str, bool]:
    try:
        spec = get_spec(body.provider)
    except RuntimeError as exc:
        raise HTTPException(400, str(exc)) from exc
    if spec.id == "dashscope":
        crud_settings.set_dashscope_api_key(db, body.api_key)
    elif spec.id == "volcengine":
        crud_settings.set_ark_api_key(db, body.api_key)
    else:
        crud_settings.set_value(db, spec.setting_key, body.api_key.strip())
    return {"ok": True}


@router.delete("/llm-key/{provider}")
def clear_llm_key(provider: str, db: Session = Depends(get_db)) -> dict[str, bool]:
    try:
        spec = get_spec(provider)
    except RuntimeError as exc:
        raise HTTPException(400, str(exc)) from exc
    if spec.id == "dashscope":
        crud_settings.delete_dashscope_api_key(db)
    elif spec.id == "volcengine":
        crud_settings.delete_ark_api_key(db)
    else:
        crud_settings.delete_value(db, spec.setting_key)
    return {"ok": True}


@router.put("/dashscope-key")
def save_dashscope_key(body: DashScopeKeyIn, db: Session = Depends(get_db)) -> dict[str, bool]:
    crud_settings.set_dashscope_api_key(db, body.api_key)
    return {"ok": True}


@router.delete("/dashscope-key")
def clear_dashscope_key(db: Session = Depends(get_db)) -> dict[str, bool]:
    crud_settings.delete_dashscope_api_key(db)
    return {"ok": True}


@router.put("/ark-key")
def save_ark_key(body: ArkKeyIn, db: Session = Depends(get_db)) -> dict[str, bool]:
    crud_settings.set_ark_api_key(db, body.api_key)
    return {"ok": True}


@router.delete("/ark-key")
def clear_ark_key(db: Session = Depends(get_db)) -> dict[str, bool]:
    crud_settings.delete_ark_api_key(db)
    return {"ok": True}
