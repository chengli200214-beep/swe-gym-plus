"""Lease-based worker; every model run and evaluator use separate processes."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from codeagentbench.service.repository import JobRepository
from codeagentbench.tasks.manifest import load_manifest


class Worker:
    def __init__(self, repository, artifact_root, manifests, *, backend="local", model_path=None, script=None, executor="bwrap"):
        self.repository = repository
        self.root = Path(artifact_root).resolve()
        self.manifests = {t.instance_id: str(Path(p).resolve()) for p in manifests for t in load_manifest(p).tasks}
        self.backend, self.model_path, self.script, self.executor = backend, model_path, script, executor

    def _env(self, *, model=False):
        env = {k: v for k, v in os.environ.items() if not any(x in k.lower() for x in ("key", "token", "password", "secret", "credential"))}
        env.update(PYTHONPATH=str(Path(__file__).resolve().parents[2]), CODEAGENTBENCH_EXECUTOR=self.executor)
        if model and self.backend == "deepseek":
            for name in ("DEEPSEEK_API_KEY", "DEEPSEEK_MIN_BALANCE_CNY", "DEEPSEEK_MAX_OUTPUT_TOKENS", "DEEPSEEK_MODEL"):
                if name in os.environ:
                    env[name] = os.environ[name]
        return env

    @staticmethod
    def _stop(process, *, force=False):
        if process.poll() is not None:
            return
        if os.name == "nt":
            process.kill() if force else process.terminate()
        else:
            os.killpg(process.pid, signal.SIGKILL if force else signal.SIGTERM)

    def _execute(self, job, command, log, *, model=False):
        cancel_at = None
        started = time.monotonic()
        with log.open("x", encoding="utf-8") as stream:
            process = subprocess.Popen([sys.executable, "-m", "codeagentbench", *command], stdout=stream, stderr=subprocess.STDOUT, env=self._env(model=model), start_new_session=os.name != "nt")
            try:
                while process.poll() is None:
                    row = self.repository.get(job["run_id"])
                    owned = self.repository.heartbeat(job["run_id"], job["lease"])
                    cancelled = not owned or row["status"] == "cancelling"
                    expired = time.monotonic() - started > job["payload"]["max_seconds"] + 150
                    if cancelled or expired:
                        if cancel_at is None:
                            self._stop(process)
                            cancel_at = time.monotonic()
                        elif time.monotonic() - cancel_at > 10:
                            self._stop(process, force=True)
                    time.sleep(0.25)
                return process.returncode, cancel_at is not None
            finally:
                if process.poll() is None:
                    self._stop(process, force=True)
                process.wait()

    def run_once(self):
        self.repository.reap_stale()
        job = self.repository.claim()
        if job is None:
            return False
        run_id, lease = job["run_id"], job["lease"]
        run_dir = self.root / "runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        suffix = str(job["resume_count"])
        try:
            manifest = self.manifests[job["task_id"]]
            command = ["run", manifest, job["task_id"], "--repo-root", str(self.root), "--run-id", run_id, "--skip-evaluation"]
            for name in ("max_steps", "max_tool_calls", "max_tokens", "max_seconds"):
                command.extend(["--" + name.replace("_", "-"), str(job["payload"][name])])
            if self.script:
                command.extend(["--script", str(Path(self.script).resolve())])
            else:
                command.extend(["--model-backend", self.backend])
                if self.model_path:
                    command.extend(["--model-path", str(Path(self.model_path).resolve())])
            if job["resume_count"] and (run_dir / "run.json").exists():
                command.append("--resume")
            code, cancelled = self._execute(job, command, run_dir / ("worker-" + suffix + ".log"), model=True)
            summary_path = run_dir / "summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {"reason": "runner exited without summary", "exit_code": code}
            if cancelled:
                self.repository.finish(run_id, lease, "cancelled", summary)
            elif code:
                self.repository.finish(run_id, lease, "interrupted", summary)
            else:
                # Demo uses the credential-free local executor; real tasks use
                # the evaluator CLI which enforces bubblewrap.
                if self.executor == "local" and self.script:
                    from codeagentbench.models import Candidate
                    from codeagentbench.verification.evaluator import Evaluator
                    task = next(t for t in load_manifest(manifest).tasks if t.instance_id == job["task_id"])
                    evaluation = Evaluator(self.root / "evaluations").evaluate(task, Candidate("0", run_id, summary.get("diff", ""), summary.get("status", "unknown"))).to_dict()
                else:
                    eval_log = run_dir / ("evaluation-" + suffix + ".log")
                    _, eval_cancelled = self._execute(job, ["evaluate-run", manifest, run_id, "--repo-root", str(self.root)], eval_log)
                    if eval_cancelled:
                        self.repository.finish(run_id, lease, "cancelled", summary)
                        return True
                    events = [json.loads(s) for s in (run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines() if s.strip()]
                    evaluation = next((e for e in reversed(events) if e.get("type") == "evaluation"), {"verdict": "blocked", "reason": "evaluation did not produce a result"})
                summary["evaluation"] = evaluation
                self.repository.finish(run_id, lease, "completed" if evaluation["verdict"] == "passed" else evaluation["verdict"], summary)
        except Exception as exc:
            self.repository.finish(run_id, lease, "interrupted", {"reason": str(exc)})
        return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--manifests", type=Path, nargs="+", required=True)
    parser.add_argument("--backend", choices=["local", "deepseek"], default="local")
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--script", type=Path)
    parser.add_argument("--executor", choices=["local", "bwrap"], default="bwrap")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.executor == "local" and not args.script:
        parser.error("local executor is only allowed for trusted scripted demos")
    worker = Worker(JobRepository(args.database), args.artifact_root, args.manifests, backend=args.backend, model_path=args.model_path, script=args.script, executor=args.executor)
    while True:
        worked = worker.run_once()
        if args.once:
            break
        if not worked:
            time.sleep(1)


if __name__ == "__main__":
    main()
