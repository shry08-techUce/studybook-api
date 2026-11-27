from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from supabase import create_client

_PROFILE_COLLECTION = "profile"
_COLLECTIONS = {
    "subjects",
    "assignments",
    "notebooks",
    "documents",
    "chats",
    "notifications",
    "reports",
}


class ProfileRequest(BaseModel):
    profile: dict[str, Any]


class RecordRequest(BaseModel):
    data: dict[str, Any]


def _records_table():
    supabase_url = os.getenv("SUPABASE_URL", "").strip()
    service_role_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip()
    if not supabase_url or not service_role_key:
        raise HTTPException(
            status_code=503,
            detail="Supabase sync is not configured.",
        )
    return create_client(supabase_url, service_role_key).table("user_sync_records")


def _response_rows(response: object) -> list[dict[str, Any]]:
    data = response.get("data") if isinstance(response, Mapping) else getattr(
        response, "data", None
    )
    if not isinstance(data, list):
        raise HTTPException(
            status_code=502,
            detail="Supabase returned an invalid sync response.",
        )
    return [dict(row) for row in data if isinstance(row, Mapping)]


def _require_collection(collection: str) -> str:
    if collection not in _COLLECTIONS:
        raise HTTPException(status_code=404, detail="Unknown sync collection.")
    return collection


def create_sync_router(authenticated_user: Callable[..., str]) -> APIRouter:
    router = APIRouter(prefix="/sync")

    @router.get("/profile")
    def fetch_profile(user_id: str = Depends(authenticated_user)) -> dict[str, Any]:
        try:
            rows = _response_rows(
                _records_table()
                .select("data")
                .eq("user_id", user_id)
                .eq("collection", _PROFILE_COLLECTION)
                .eq("record_id", "profile")
                .limit(1)
                .execute()
            )
        except HTTPException:
            raise
        except Exception as error:
            raise HTTPException(
                status_code=502,
                detail="Could not fetch the Supabase profile.",
            ) from error
        return {"profile": rows[0]["data"] if rows else None}

    @router.put("/profile")
    def save_profile(
        body: ProfileRequest,
        user_id: str = Depends(authenticated_user),
    ) -> dict[str, bool]:
        try:
            _records_table().upsert(
                {
                    "user_id": user_id,
                    "collection": _PROFILE_COLLECTION,
                    "record_id": "profile",
                    "data": body.profile,
                },
                on_conflict="user_id,collection,record_id",
            ).execute()
        except HTTPException:
            raise
        except Exception as error:
            raise HTTPException(
                status_code=502,
                detail="Could not save the Supabase profile.",
            ) from error
        return {"saved": True}

    @router.get("/collections/{collection}")
    def fetch_collection(
        collection: str,
        user_id: str = Depends(authenticated_user),
    ) -> dict[str, list[dict[str, Any]]]:
        collection = _require_collection(collection)
        try:
            rows = _response_rows(
                _records_table()
                .select("record_id,data")
                .eq("user_id", user_id)
                .eq("collection", collection)
                .execute()
            )
        except HTTPException:
            raise
        except Exception as error:
            raise HTTPException(
                status_code=502,
                detail="Could not fetch the Supabase collection.",
            ) from error

        records = []
        for row in rows:
            data = row.get("data")
            if not isinstance(data, Mapping):
                raise HTTPException(
                    status_code=502,
                    detail="Supabase returned an invalid sync record.",
                )
            records.append({**dict(data), "id": row["record_id"]})
        return {"records": records}

    @router.put("/collections/{collection}/{record_id}")
    def save_record(
        collection: str,
        record_id: str,
        body: RecordRequest,
        user_id: str = Depends(authenticated_user),
    ) -> dict[str, bool]:
        collection = _require_collection(collection)
        if not record_id.strip():
            raise HTTPException(status_code=400, detail="Record ID is required.")
        data = {**body.data, "id": record_id}
        try:
            _records_table().upsert(
                {
                    "user_id": user_id,
                    "collection": collection,
                    "record_id": record_id,
                    "data": data,
                },
                on_conflict="user_id,collection,record_id",
            ).execute()
        except HTTPException:
            raise
        except Exception as error:
            raise HTTPException(
                status_code=502,
                detail="Could not save the Supabase sync record.",
            ) from error
        return {"saved": True}

    @router.delete("/collections/{collection}/{record_id}")
    def delete_record(
        collection: str,
        record_id: str,
        user_id: str = Depends(authenticated_user),
    ) -> dict[str, bool]:
        collection = _require_collection(collection)
        if not record_id.strip():
            raise HTTPException(status_code=400, detail="Record ID is required.")
        try:
            _records_table().delete().eq("user_id", user_id).eq(
                "collection", collection
            ).eq("record_id", record_id).execute()
        except HTTPException:
            raise
        except Exception as error:
            raise HTTPException(
                status_code=502,
                detail="Could not delete the Supabase sync record.",
            ) from error
        return {"deleted": True}

    return router
