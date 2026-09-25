"""Transactional job queue shared by SQLite development and PostgreSQL workers."""
from __future__ import annotations

import json
from pathlib import Path
import time
import uuid

from sqlalchemy import Column, Float, Integer, MetaData, String, Table, Text, create_engine, select, update


class JobRepository:
    def __init__(self, database: str | Path):
        value = str(database)
        if "://" not in value:
            path = Path(value).resolve()
            path.parent.mkdir(parents=True, exist_ok=True)
            value = "sqlite:///" + path.as_posix()
        self.engine = create_engine(value, connect_args={"check_same_thread": False, "timeout": 30} if value.startswith("sqlite:") else {}, pool_pre_ping=True)
        metadata = MetaData()
        self.jobs = Table("jobs", metadata,
            Column("run_id", String(160), primary_key=True), Column("task_id", String(200), nullable=False),
            Column("status", String(32), nullable=False), Column("payload", Text, nullable=False),
            Column("summary", Text, nullable=False, default="{}"), Column("lease", String(40)),
            Column("heartbeat", Float), Column("created", Float, nullable=False), Column("updated", Float, nullable=False),
            Column("resume_count", Integer, nullable=False, default=0))
        metadata.create_all(self.engine)

    @staticmethod
    def decode(row):
        if row is None:
            return None
        value = dict(row)
        for field in ("payload", "summary"):
            value[field] = json.loads(value[field])
        return value

    def submit(self, task_id, payload):
        run_id, now = uuid.uuid4().hex, time.time()
        with self.engine.begin() as connection:
            connection.execute(self.jobs.insert().values(run_id=run_id, task_id=task_id, status="queued", payload=json.dumps(payload), summary="{}", created=now, updated=now, resume_count=0))
        return self.get(run_id)

    def get(self, run_id):
        with self.engine.connect() as connection:
            return self.decode(connection.execute(select(self.jobs).where(self.jobs.c.run_id == run_id)).mappings().first())

    def list(self, limit=100):
        with self.engine.connect() as connection:
            return [self.decode(row) for row in connection.execute(select(self.jobs).order_by(self.jobs.c.created.desc()).limit(limit)).mappings()]

    def claim(self):
        with self.engine.connect() as connection:
            if self.engine.dialect.name == "sqlite":
                connection.exec_driver_sql("BEGIN IMMEDIATE")
            else:
                connection.begin()
            try:
                query = select(self.jobs).where(self.jobs.c.status == "queued").order_by(self.jobs.c.created).limit(1)
                if self.engine.dialect.name == "postgresql":
                    query = query.with_for_update(skip_locked=True)
                row = connection.execute(query).mappings().first()
                if row is None:
                    connection.commit()
                    return None
                lease, now = uuid.uuid4().hex, time.time()
                connection.execute(update(self.jobs).where(self.jobs.c.run_id == row["run_id"], self.jobs.c.status == "queued").values(status="running", lease=lease, heartbeat=now, updated=now))
                connection.commit()
                return {**self.decode(row), "status": "running", "lease": lease}
            except BaseException:
                connection.rollback()
                raise

    def heartbeat(self, run_id, lease):
        with self.engine.begin() as connection:
            result = connection.execute(update(self.jobs).where(self.jobs.c.run_id == run_id, self.jobs.c.lease == lease, self.jobs.c.status.in_(["running", "cancelling"])).values(heartbeat=time.time()))
            return result.rowcount == 1

    def finish(self, run_id, lease, status, summary):
        with self.engine.begin() as connection:
            result = connection.execute(update(self.jobs).where(self.jobs.c.run_id == run_id, self.jobs.c.lease == lease, self.jobs.c.status.in_(["running", "cancelling"])).values(status=status, summary=json.dumps(summary), lease=None, updated=time.time()))
            return result.rowcount == 1

    def cancel(self, run_id):
        with self.engine.begin() as connection:
            connection.execute(update(self.jobs).where(self.jobs.c.run_id == run_id, self.jobs.c.status == "queued").values(status="cancelled", updated=time.time()))
            connection.execute(update(self.jobs).where(self.jobs.c.run_id == run_id, self.jobs.c.status == "running").values(status="cancelling", updated=time.time()))
        return self.get(run_id)

    def resume(self, run_id):
        with self.engine.begin() as connection:
            result = connection.execute(update(self.jobs).where(self.jobs.c.run_id == run_id, self.jobs.c.status.in_(["cancelled", "interrupted"])).values(status="queued", lease=None, updated=time.time(), resume_count=self.jobs.c.resume_count + 1))
            if result.rowcount != 1:
                raise ValueError("only cancelled/interrupted jobs can resume")
        return self.get(run_id)

    def reap_stale(self, *, seconds=60):
        with self.engine.begin() as connection:
            result = connection.execute(update(self.jobs).where(self.jobs.c.status.in_(["running", "cancelling"]), self.jobs.c.heartbeat < time.time() - seconds).values(status="interrupted", lease=None, updated=time.time(), summary=json.dumps({"reason": "worker heartbeat expired; explicit recovery required"})))
            return result.rowcount
