"""Refreshable provenance manifests for saved MO-OBAM artifacts."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Iterator


def atomic_write_text(path: str | Path, content: str) -> Path:
    """Durably replace one text file without exposing a partial write."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        directory_descriptor = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


@contextmanager
def exclusive_directory_lock(
    directory: str | Path,
    lock_name: str = ".prti-study.lock",
) -> Iterator[Path]:
    """Prevent two local processes from mutating one study directory."""

    try:
        import fcntl
    except ImportError as error:  # pragma: no cover - the supported server is Linux.
        raise RuntimeError("PRTI study locking requires a POSIX/Linux server.") from error

    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / lock_name
    with lock_path.open("a+", encoding="utf-8") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                f"Another PRTI process is already using {root.resolve()}. "
                "Do not run two processes against the same output directory."
            ) from error
        stream.seek(0)
        stream.truncate()
        stream.write(
            json.dumps(
                {
                    "pid": os.getpid(),
                    "acquired_at_utc": datetime.now(timezone.utc).isoformat(
                        timespec="seconds"
                    ),
                }
            )
        )
        stream.flush()
        try:
            yield lock_path
        finally:
            stream.seek(0)
            stream.truncate()
            stream.flush()
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


class ArtifactManifestWriter:
    """Hash a saved run and record the code state needed to identify it."""

    MANIFEST_VERSION = 2

    def __init__(self, project_root: str | Path) -> None:
        self.project_root = Path(project_root).resolve()

    def write(
        self,
        output_directory: str | Path,
        source_path: str | Path,
        summary: dict[str, object],
        manifest_name: str = "prti_artifact_manifest.json",
    ) -> Path:
        """Create or refresh a manifest without changing any research data."""

        output = Path(output_directory)
        artifacts = self._artifact_records(output, summary)
        source = Path(source_path)
        generation_snapshot = summary.get("generation_snapshot")
        if not isinstance(generation_snapshot, dict):
            generation_snapshot = self.capture_snapshot(
                "captured_at_manifest_refresh_not_run_start"
            )

        manifest = {
            "manifest_version": self.MANIFEST_VERSION,
            "generated_at_utc": self._utc_now(),
            "update_policy": (
                "Artifact hashes may be refreshed, but a generation snapshot "
                "captured at run start is preserved. The manifest never modifies "
                "research data."
            ),
            "run": {
                "run_id": summary.get("run_id", self._legacy_run_id(summary)),
                "run_label": summary.get("run_label"),
                "started_at_utc": summary.get("run_started_at_utc"),
                "completed_at_utc": summary.get("run_completed_at_utc"),
                "timestamp_status": (
                    "recorded during execution"
                    if summary.get("run_started_at_utc")
                    else "not recorded by the legacy run; not reconstructed"
                ),
            },
            "source": {
                "filename": source.name,
                "sha256": self.file_sha256(source),
            },
            "configuration": {
                "model": summary.get("model_configuration"),
                "pso": summary.get("pso_configuration"),
            },
            "execution": summary.get("execution"),
            "code": generation_snapshot,
            "artifacts": artifacts,
        }
        path = output / manifest_name
        atomic_write_text(path, json.dumps(manifest, indent=2))
        return path

    @staticmethod
    def file_sha256(path: str | Path) -> str:
        """Return the SHA-256 digest of one file."""

        digest = sha256()
        with Path(path).open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def _artifact_records(
        self,
        output: Path,
        summary: dict[str, object],
    ) -> dict[str, dict[str, object]]:
        names = summary.get("outputs")
        if not isinstance(names, dict):
            raise ValueError("Saved run summary does not contain an outputs dictionary.")

        records: dict[str, dict[str, object]] = {}
        for role, filename in sorted(names.items()):
            if role == "artifact_manifest":
                continue
            path = output / str(filename)
            if not path.is_file():
                raise FileNotFoundError(f"Saved run artifact not found: {path}")
            records[str(role)] = {
                "filename": path.name,
                "bytes": path.stat().st_size,
                "sha256": self.file_sha256(path),
            }
        return records

    def capture_snapshot(self, status: str = "captured_at_run_start") -> dict[str, object]:
        """Capture Git state and separate generation, test, and document hashes."""

        generation_files = [
            *sorted((self.project_root / "mo_obam").glob("*.py")),
            self.project_root / "generate_prti_results.py",
            self.project_root / "generate_prti_results_package.py",
            self.project_root / "run_prti.py",
            self.project_root / "run_prti_experiments.py",
            self.project_root / "run_prti_release.py",
            self.project_root / "run_prti_selected_candidate.py",
            self.project_root / "run_prti_cluster_study.sh",
            self.project_root / "run_prti_cluster_range_study.sh",
            self.project_root / "requirements.txt",
        ]
        validation_files = sorted((self.project_root / "tests").glob("test_*.py"))
        documentation_files = [
            self.project_root / "README.md",
            self.project_root / "PAPER_REVIEW_V2.md",
            self.project_root / "PRTI_MO_OBAM_DOCUMENTATION.md",
            self.project_root / "PRTI_MO_OBAM_RESULTS.md",
            self.project_root / "PRTI_CLUSTER_RANGE_STUDY.md",
            self.project_root / "PRTI_GH_RELEASE_READINESS.md",
        ]
        return {
            "snapshot_status": status,
            "git": self._git_state(),
            "generation_files": self._file_records(generation_files),
            "validation_files": self._file_records(validation_files),
            "documentation_files": self._file_records(documentation_files),
        }

    def _file_records(self, candidates: list[Path]) -> dict[str, str]:
        return {
            str(path.relative_to(self.project_root)): self.file_sha256(path)
            for path in candidates
            if path.is_file()
        }

    def _git_state(self) -> dict[str, object]:
        def run(*arguments: str) -> str | None:
            result = subprocess.run(
                ["git", *arguments],
                cwd=self.project_root,
                capture_output=True,
                check=False,
                text=True,
                timeout=5,
            )
            return result.stdout.strip() if result.returncode == 0 else None

        commit = run("rev-parse", "HEAD")
        status = run("status", "--porcelain")
        return {
            "commit": commit,
            "working_tree_clean": status == "" if status is not None else None,
            "changed_path_count": len(status.splitlines()) if status else 0,
        }

    @staticmethod
    def _legacy_run_id(summary: dict[str, object]) -> str:
        model = summary.get("model_configuration") or {}
        pso = summary.get("pso_configuration") or {}
        if not isinstance(model, dict) or not isinstance(pso, dict):
            return "legacy-prti-run"
        weight = str(model.get("lambda_weight", "unknown")).replace(".", "p")
        return (
            f"legacy-prti-{summary.get('run_label', 'run')}"
            f"-k{model.get('k', 'unknown')}"
            f"-c{model.get('n_clusters', 'unknown')}"
            f"-lambda{weight}"
            f"-seed{pso.get('seed', 'unknown')}"
        )

    @staticmethod
    def _utc_now() -> str:
        return datetime.now(timezone.utc).isoformat(timespec="seconds")
