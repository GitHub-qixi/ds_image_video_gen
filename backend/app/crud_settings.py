"""settings 表读写。"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy.orm import Session

from app.models import Setting

KEY_DASHSCOPE_API = "dashscope_api_key"
KEY_ARK_API = "ark_api_key"


def get_dashscope_record(db: Session) -> Setting | None:
    return db.get(Setting, KEY_DASHSCOPE_API)


def get_dashscope_api_key(db: Session) -> str | None:
    row = get_dashscope_record(db)
    if not row or not row.value:
        return None
    s = row.value.strip()
    return s or None


def set_dashscope_api_key(db: Session, api_key: str) -> None:
    key = api_key.strip()
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    row = db.get(Setting, KEY_DASHSCOPE_API)
    if row:
        row.value = key
        row.updated_at = now
    else:
        db.add(Setting(key=KEY_DASHSCOPE_API, value=key, updated_at=now))
    db.commit()


def delete_dashscope_api_key(db: Session) -> None:
    row = db.get(Setting, KEY_DASHSCOPE_API)
    if row:
        db.delete(row)
        db.commit()


def get_ark_record(db: Session) -> Setting | None:
    return db.get(Setting, KEY_ARK_API)


def get_ark_api_key(db: Session) -> str | None:
    row = get_ark_record(db)
    if not row or not row.value:
        return None
    s = row.value.strip()
    return s or None


def set_ark_api_key(db: Session, api_key: str) -> None:
    key = api_key.strip()
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    row = db.get(Setting, KEY_ARK_API)
    if row:
        row.value = key
        row.updated_at = now
    else:
        db.add(Setting(key=KEY_ARK_API, value=key, updated_at=now))
    db.commit()


def delete_ark_api_key(db: Session) -> None:
    row = db.get(Setting, KEY_ARK_API)
    if row:
        db.delete(row)
        db.commit()


def get_value(db: Session, key: str) -> str | None:
    row = db.get(Setting, key)
    if not row or not row.value:
        return None
    s = row.value.strip()
    return s or None


def set_value(db: Session, key: str, value: str) -> None:
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    row = db.get(Setting, key)
    if row:
        row.value = value
        row.updated_at = now
    else:
        db.add(Setting(key=key, value=value, updated_at=now))
    db.commit()


def delete_value(db: Session, key: str) -> None:
    row = db.get(Setting, key)
    if row:
        db.delete(row)
        db.commit()
