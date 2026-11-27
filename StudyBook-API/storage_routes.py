from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from supabase import create_client
from storage3.types import CreateSignedUploadUrlOptions
import logging
_BUCKET_NAME = "documents"
_DOWNLOAD_URL_TTL_SECONDS = 60


class UploadUrlRequest(BaseModel):
    subject_id: str = Field(alias="subjectId")
    notebook_id: str = Field(alias="notebookId")
    document_id: str = Field(alias="documentId")
    version: int = Field(ge=1)


class ObjectPathRequest(BaseModel):
    path: str


def _validate_segment(value: str) -> str:
    if (
        not value.strip()
        or value in {".", ".."}
        or ".." in value
        or "/" in value
        or "\\" in value
    ):
        raise HTTPException(status_code=400, detail="Invalid storage path segment.")
    return value


def build_object_path(
    user_id: str,
    subject_id: str,
    notebook_id: str,
    document_id: str,
    version: int,
) -> str:
    parts = (
        _validate_segment(user_id),
        _validate_segment(subject_id),
        _validate_segment(notebook_id),
        _validate_segment(f"{_validate_segment(document_id)}-v{version}"),
    )
    return "/".join(parts)


def validate_owned_path(user_id: str, object_path: str) -> str:
    if not object_path or "\\" in object_path or ".." in object_path:
        raise HTTPException(status_code=400, detail="Invalid storage object path.")
    parts = object_path.split("/")
    if (
        len(parts) != 4
        or any(not part or part in {".", ".."} for part in parts)
    ):
        raise HTTPException(status_code=400, detail="Invalid storage object path.")
    if parts[0] != user_id:
        raise HTTPException(status_code=403, detail="Storage object access denied.")
    return object_path


def _storage_bucket():
    supabase_url = os.getenv("SUPABASE_URL", "").strip()
    service_role_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip()
    if not supabase_url or not service_role_key:
        raise HTTPException(
            status_code=503,
            detail="File storage is not configured.",
        )
    return create_client(supabase_url, service_role_key).storage.from_(_BUCKET_NAME)


def _response_value(response: object, *keys: str) -> str | None:
    if isinstance(response, Mapping):
        for key in keys:
            value = response.get(key)
            if isinstance(value, str) and value:
                return value
    for key in keys:
        value = getattr(response, key, None)
        if isinstance(value, str) and value:
            return value
    return None


def create_storage_router(
    authenticated_user: Callable[..., str],
) -> APIRouter:
    router = APIRouter()

    @router.post("/files/upload-url")
    def create_upload_url(
        body: UploadUrlRequest,
        user_id: str = Depends(authenticated_user),
    ) -> dict[str, str]:
        object_path = build_object_path(
            user_id,
            body.subject_id,
            body.notebook_id,
            body.document_id,
            body.version,
        )
        try:
            response = _storage_bucket().create_signed_upload_url(
                object_path,
                options= CreateSignedUploadUrlOptions(upsert=True),
            )
        except Exception as exc:
            logging.exception(
            "Supabase signed URL creation failed. Bucket=%s Path=%s",
            _BUCKET_NAME,
            object_path,
            )
            raise HTTPException(
            status_code=502,
            detail=f"Signed URL error: {type(exc).__name__}: {exc}",
        ) from exc

        token = _response_value(response, "token")
        if token is None:
            raise HTTPException(
                status_code=502,
                detail="Storage did not return an upload token.",
            )
        return {"path": object_path, "token": token}

    @router.post("/files/download-url")
    def create_download_url(
        body: ObjectPathRequest,
        user_id: str = Depends(authenticated_user),
    ) -> dict[str, str]:
        object_path = validate_owned_path(user_id, body.path)
        try:
            response = _storage_bucket().create_signed_url(
                object_path,
                _DOWNLOAD_URL_TTL_SECONDS,
            )
        except HTTPException:
            raise
        except Exception as error:
            raise HTTPException(
                status_code=502,
                detail="Could not create a signed download URL.",
            ) from error

        signed_url = _response_value(response, "signedURL", "signedUrl", "url")
        if signed_url is None:
            raise HTTPException(
                status_code=502,
                detail="Storage did not return a signed download URL.",
            )
        return {"url": signed_url}

    @router.post("/files/delete")
    def delete_object(
        body: ObjectPathRequest,
        user_id: str = Depends(authenticated_user),
    ) -> dict[str, bool]:
        object_path = validate_owned_path(user_id, body.path)
        try:
            _storage_bucket().remove([object_path])
        except HTTPException:
            raise
        except Exception as error:
            raise HTTPException(
                status_code=502,
                detail="Could not delete the storage object.",
            ) from error
        return {"deleted": True}

    return router
