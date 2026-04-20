import hashlib
import os
from pathlib import Path

from agents import apply_diff
from agents.editor import ApplyPatchOperation
from agents.editor import ApplyPatchResult

from glyph.approvals import request_tool_approval
from glyph.options import ApprovalHandler


class ApprovalTracker:
    def __init__(self) -> None:
        self._approved: set[str] = set()

    def fingerprint(self, operation: ApplyPatchOperation, relative_path: str) -> str:
        hasher = hashlib.sha256()
        hasher.update(operation.type.encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(relative_path.encode("utf-8"))
        hasher.update(b"\0")
        hasher.update((operation.diff or "").encode("utf-8"))
        return hasher.hexdigest()

    def remember(self, fingerprint: str) -> None:
        self._approved.add(fingerprint)

    def is_approved(self, fingerprint: str) -> bool:
        return fingerprint in self._approved


class WorkspaceEditor:
    """ApplyPatchEditor implementation scoped to a single root (see vendor apply_patch example)."""

    def __init__(
        self,
        root: Path,
        confirm_patches: bool,
        approval_handler: ApprovalHandler | None = None,
    ) -> None:
        self._root = root.resolve()
        self._confirm_patches = confirm_patches
        self._approval_handler = approval_handler
        self._approvals = ApprovalTracker()
        self._env_auto = os.environ.get("APPLY_PATCH_AUTO_APPROVE") == "1"

    def create_file(self, operation: ApplyPatchOperation) -> ApplyPatchResult:
        relative = self._relative_path(operation.path)
        denied_reason = self._require_approval(operation, relative)
        if denied_reason is not None:
            return ApplyPatchResult(output=f"Declined: {denied_reason}")
        target = self._resolve(operation.path, ensure_parent=True)
        diff = operation.diff or ""
        content = apply_diff("", diff, mode="create")
        target.write_text(content, encoding="utf-8")
        return ApplyPatchResult(output=f"Created {relative}")

    def update_file(self, operation: ApplyPatchOperation) -> ApplyPatchResult:
        relative = self._relative_path(operation.path)
        denied_reason = self._require_approval(operation, relative)
        if denied_reason is not None:
            return ApplyPatchResult(output=f"Declined: {denied_reason}")
        target = self._resolve(operation.path)
        original = target.read_text(encoding="utf-8")
        diff = operation.diff or ""
        patched = apply_diff(original, diff)
        target.write_text(patched, encoding="utf-8")
        return ApplyPatchResult(output=f"Updated {relative}")

    def delete_file(self, operation: ApplyPatchOperation) -> ApplyPatchResult:
        relative = self._relative_path(operation.path)
        denied_reason = self._require_approval(operation, relative)
        if denied_reason is not None:
            return ApplyPatchResult(output=f"Declined: {denied_reason}")
        target = self._resolve(operation.path)
        target.unlink(missing_ok=True)
        return ApplyPatchResult(output=f"Deleted {relative}")

    def _relative_path(self, value: str) -> str:
        resolved = self._resolve(value)
        return resolved.relative_to(self._root).as_posix()

    def _resolve(self, relative: str, ensure_parent: bool = False) -> Path:
        candidate = Path(relative)
        target = candidate if candidate.is_absolute() else (self._root / candidate)
        target = target.resolve()
        try:
            target.relative_to(self._root)
        except ValueError:
            raise RuntimeError(f"Operation outside workspace: {relative}") from None
        if ensure_parent:
            target.parent.mkdir(parents=True, exist_ok=True)
        return target

    def _require_approval(self, operation: ApplyPatchOperation, display_path: str) -> str | None:
        if self._env_auto or not self._confirm_patches:
            return None

        fingerprint = self._approvals.fingerprint(operation, display_path)
        if self._approvals.is_approved(fingerprint):
            return None

        approved, denied_reason = request_tool_approval(
            handler=self._approval_handler,
            capability="edit",
            tool_name="apply_patch",
            payload={
                "type": operation.type,
                "path": display_path,
                "diff": operation.diff,
            },
        )

        if approved:
            self._approvals.remember(fingerprint)
            return None
        return denied_reason

