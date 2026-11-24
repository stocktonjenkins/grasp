"""Run Neo4j as a plain child process from an unpacked distribution (no Docker).

Needs `NEO4J_HOME` pointing at an unpacked neo4j-community-5.26.x tarball and a
JDK 17 on PATH (or `JAVA_HOME`). Each graph lives in its own `data_dir`, laid out
as `data_dir/data`, so stores built on one machine can be copied to another.
"""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from neo4j import Driver, GraphDatabase
from neo4j.exceptions import ServiceUnavailable

READY_TIMEOUT_SECONDS = 180


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def neo4j_home() -> Path:
    home = os.getenv("NEO4J_HOME")
    if not home or not (Path(home) / "bin" / "neo4j").exists():
        raise RuntimeError(
            "Set NEO4J_HOME to an unpacked neo4j-community-5.26.x distribution "
            "(https://neo4j.com/deployment-center/), or set NEO4J_URI to use a running server."
        )
    return Path(home)


def install_apoc(home: Path) -> None:
    if any((home / "plugins").glob("apoc-*-core.jar")):
        return
    bundled = sorted((home / "labs").glob("apoc-*-core.jar"))
    if not bundled:
        raise RuntimeError(f"No APOC core jar found in {home / 'labs'}.")
    shutil.copy(bundled[-1], home / "plugins")


class Neo4jServer:
    def __init__(self, data_dir: Path) -> None:
        self.data_dir = Path(data_dir).resolve()
        self.bolt_port = free_port()
        self.http_port = free_port()
        self._process: subprocess.Popen | None = None

    @property
    def uri(self) -> str:
        return f"bolt://127.0.0.1:{self.bolt_port}"

    def start(self) -> None:
        home = neo4j_home()
        install_apoc(home)
        conf_dir = self.write_conf(home)
        log = (self.data_dir / "logs" / "console.log").open("ab")
        self._process = subprocess.Popen(
            [str(home / "bin" / "neo4j"), "console"],
            env={**os.environ, "NEO4J_CONF": str(conf_dir)},
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        self.wait_until_ready()

    def stop(self) -> None:
        if self._process is None or self._process.poll() is not None:
            return
        self._process.terminate()
        try:
            self._process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait(timeout=10)

    def write_conf(self, home: Path) -> Path:
        conf_dir = self.data_dir / "conf"
        for sub in (conf_dir, self.data_dir / "data", self.data_dir / "logs"):
            sub.mkdir(parents=True, exist_ok=True)
        overrides = f"""
server.default_listen_address=127.0.0.1
server.bolt.listen_address=127.0.0.1:{self.bolt_port}
server.http.listen_address=127.0.0.1:{self.http_port}
server.directories.data={self.data_dir / "data"}
server.directories.logs={self.data_dir / "logs"}
server.directories.run={self.data_dir / "run"}
dbms.security.auth_enabled=false
dbms.security.procedures.unrestricted=apoc.*
server.memory.heap.initial_size={os.getenv("NEO4J_HEAP_INITIAL", "1g")}
server.memory.heap.max_size={os.getenv("NEO4J_HEAP_MAX", "4g")}
server.memory.pagecache.size={os.getenv("NEO4J_PAGECACHE", "2g")}
dbms.memory.transaction.total.max={os.getenv("NEO4J_TX_MAX", "4g")}
dbms.usage_report.enabled=false
"""
        (conf_dir / "neo4j.conf").write_text((home / "conf" / "neo4j.conf").read_text() + overrides)
        return conf_dir

    def wait_until_ready(self) -> None:
        deadline = time.monotonic() + READY_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            if self._process is not None and self._process.poll() is not None:
                break
            try:
                with GraphDatabase.driver(self.uri) as driver:
                    driver.verify_connectivity()
                return
            except (ServiceUnavailable, OSError):
                time.sleep(2)
        tail = (self.data_dir / "logs" / "console.log").read_text(errors="replace").splitlines()[-40:]
        self.stop()
        raise RuntimeError("Neo4j did not become ready:\n" + "\n".join(tail))


@contextmanager
def open_graph(data_dir: Path | None) -> Iterator[Driver]:
    """Connect to `NEO4J_URI` if set; otherwise run a local server on `data_dir` for the duration."""
    from grasp.graph.store import connect

    if os.getenv("NEO4J_URI"):
        driver = connect()
        try:
            yield driver
        finally:
            driver.close()
        return
    if data_dir is None:
        raise ValueError("Set neo4j_dir in the config (or NEO4J_URI) to choose where the graph lives.")
    server = Neo4jServer(data_dir)
    server.start()
    driver = connect(server.uri)
    try:
        yield driver
    finally:
        driver.close()
        server.stop()
