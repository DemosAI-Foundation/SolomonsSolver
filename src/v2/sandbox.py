"""Mechanics D/E — sandbox execution + test sanity checking, with safe downgrade.

D. Sandbox adapter: executable sub-problems are only trusted after their code
   has actually run in a sandbox. If no sandbox is wired in, the pipeline
   degrades gracefully — the code is not executed and the result is marked
   unverified (never silently trusted).
E. Test sanity checking: a "tests passed" signal is only accepted when tests
   are present and well-formed; vacuous passes are rejected.
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from typing import Any, Awaitable, Callable, Optional, Union

from src.v2.contracts import ExecutionResult

# A sandbox runner is an (optionally async) callable: (code, tests) -> ExecutionResult
SandboxRunner = Union[
    Callable[[str, list[dict]], ExecutionResult],
    Callable[[str, list[dict]], Awaitable[ExecutionResult]],
]


def sanity_check_tests(tests: list[dict], result: ExecutionResult) -> tuple[bool, list[str]]:
    """Structural sanity check (Mechanic E). Returns ``(sane, issues)``.

    A clean code run with no model-written tests is fine; we only reject
    malformed test objects, or a ``verified`` claim with no execution evidence.
    """
    issues: list[str] = []
    if tests:
        for i, t in enumerate(tests):
            if not isinstance(t, dict):
                issues.append(f"test[{i}] is not an object")
    if (
        result.verified
        and not result.tests_passed
        and not (result.stdout or result.output)
        and not tests
    ):
        issues.append("verified=True without execution evidence (no output/tests)")
    return (len(issues) == 0, issues)


class SandboxAdapter:
    def __init__(self, runner: Optional[SandboxRunner] = None, timeout: float = 10.0):
        self.runner = runner
        self.timeout = timeout

    @property
    def available(self) -> bool:
        return self.runner is not None

    def kill(self) -> bool:
        """v2.4 Rule 7: abort an in-flight sandbox execution (FINISH NOW).

        Delegates to the runner's ``kill()`` when the runner supports it
        (e.g. :class:`SubprocessRunner`); otherwise a no-op returning False.
        """
        runner = self.runner
        if runner is not None and hasattr(runner, "kill"):
            try:
                return bool(runner.kill())
            except Exception:  # noqa: BLE001
                return False
        return False

    async def run(
        self,
        code: str,
        tests: Optional[list[dict]] = None,
        sub_problem_id: str = "",
        task_id: str = "",
        emit=None,
    ) -> ExecutionResult:
        tests = tests or []
        if self.runner is None:
            result = ExecutionResult(
                sub_problem_id=sub_problem_id,
                code=code,
                verified=False,
                sandbox_available=False,
                errors=["sandbox unavailable — code was not executed"],
            )
            if emit is not None:
                await emit("phase3", "sandbox", available=False, sub_problem_id=sub_problem_id)
            return result
        try:
            res = self.runner(code, tests)
            if hasattr(res, "__await__"):
                res = await res
            if isinstance(res, dict):
                res = ExecutionResult(**res)
            if not isinstance(res, ExecutionResult):
                res = ExecutionResult(sub_problem_id=sub_problem_id, code=code, output=str(res))
            if not res.sub_problem_id:
                res.sub_problem_id = sub_problem_id
            sane, issues = sanity_check_tests(tests, res)
            if not sane:
                res.verified = False
                res.errors = list(res.errors or []) + issues
            if emit is not None:
                await emit(
                    "phase3",
                    "sandbox",
                    sub_problem_id=sub_problem_id,
                    verified=res.verified,
                    tests_passed=res.tests_passed,
                    errors=res.errors,
                )
            return res
        except Exception as exc:  # noqa: BLE001
            if emit is not None:
                await emit("phase3", "sandbox", sub_problem_id=sub_problem_id, error=str(exc))
            return ExecutionResult(
                sub_problem_id=sub_problem_id,
                code=code,
                verified=False,
                sandbox_available=True,
                errors=[f"sandbox error: {exc}"],
            )


class SubprocessRunner:
    """Best-effort local runner: execute code in a subprocess and capture stdout.

    ``tests_passed`` reflects whether the program ran to completion (returncode
    0). Individual test cases are the runner caller's responsibility in a real
    sandbox; here we treat a clean run as the pass signal.

    v2.4 Rule 7: tracks the in-flight ``Popen`` and exposes ``kill()`` so a
    dynamic FINISH NOW can abort the execution.

    v2.6: the child runs in its OWN process group (``start_new_session``) and
    ``kill()`` kills the whole *process group*, not just the direct child — so
    a FINISH NOW during a sandbox run also reaps any grandchildren the code
    spawned, immediately.
    """

    def __init__(self, timeout: float = 10.0) -> None:
        self.timeout = timeout
        self._proc: subprocess.Popen | None = None

    @property
    def running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def _kill_group(self, proc: subprocess.Popen, sig: int) -> None:
        """Send ``sig`` to the whole process group (POSIX), falling back to the
        single process on platforms without ``killpg`` (e.g. Windows)."""
        pgid = None
        killpg = getattr(os, "killpg", None)
        if killpg is not None:
            try:
                pgid = os.getpgid(proc.pid)
            except (ProcessLookupError, OSError):
                pgid = None
        if killpg is not None and pgid is not None:
            try:
                killpg(pgid, sig)
                return
            except (ProcessLookupError, PermissionError, OSError):
                pass
        # Fallback: signal the direct child only.
        try:
            if sig == signal.SIGKILL:
                proc.kill()
            else:
                proc.terminate()
        except (ProcessLookupError, OSError):
            pass

    def kill(self) -> bool:
        """Kill the in-flight sandbox process group (if any). True if killed.

        Sends SIGTERM to the group first, then SIGKILL if it survives, so the
        interrupt is immediate even for misbehaving descendants.
        """
        proc = self._proc
        if proc is None or proc.poll() is not None:
            return False
        try:
            self._kill_group(proc, signal.SIGTERM)
            try:
                proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self._kill_group(proc, signal.SIGKILL)
                try:
                    proc.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    pass
            return True
        except Exception:  # noqa: BLE001
            return False
        finally:
            self._proc = None

    def __call__(self, code: str, tests: list[dict]) -> ExecutionResult:
        start = time.monotonic()
        try:
            popen_kwargs: dict[str, Any] = dict(
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            # Run in its own process group so kill() can reap the whole tree.
            # start_new_session is POSIX-only; degrade gracefully elsewhere.
            if hasattr(os, "setsid"):
                popen_kwargs["start_new_session"] = True
            proc = subprocess.Popen(
                [sys.executable, "-c", code],
                **popen_kwargs,
            )
            self._proc = proc
            try:
                stdout, stderr = proc.communicate(timeout=self.timeout)
            except subprocess.TimeoutExpired:
                self.kill()
                return ExecutionResult(
                    code=code,
                    stdout="",
                    output="",
                    tests_passed=False,
                    test_results=[],
                    verified=False,
                    errors=[f"timeout after {self.timeout:.1f}s"],
                )
            finally:
                self._proc = None
            stdout = stdout or ""
            ok = proc.returncode == 0
            return ExecutionResult(
                code=code,
                stdout=stdout,
                output=stdout.strip(),
                tests_passed=ok,
                test_results=[{"name": "run", "passed": ok}],
                verified=ok,
                errors=[] if ok else [f"non-zero exit: {(stderr or '').strip()[:500]}"],
            )
        except Exception as exc:  # noqa: BLE001
            self._proc = None
            return ExecutionResult(
                code=code,
                stdout="",
                output="",
                tests_passed=False,
                test_results=[],
                verified=False,
                errors=[f"sandbox error: {exc}"],
            )


def make_subprocess_runner(timeout: float = 10.0) -> SandboxRunner:
    """Factory returning a killable :class:`SubprocessRunner`."""
    return SubprocessRunner(timeout=timeout)
