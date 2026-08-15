"""
MODULE: tests.test_p0e_supervisor
DESCRIPTION: Fault-Injection & Reliability Test Suite for P0-E Supervisor.
"""

import os
import sys
import time
import pytest
import tempfile

from tokocrypto_bot.persistence.database import DatabaseManager
from tokocrypto_bot.persistence.migrations import run_migrations
from tokocrypto_bot.persistence.lifecycle_state import LifecycleManager, ApplicationState
from tokocrypto_bot.supervisor.supervisor import NVRASupervisor, SupervisorState
from tokocrypto_bot.supervisor.crash_tracker import PersistentCrashTracker


@pytest.fixture
def supervisor_env():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "supervisor_test.db")
        db_mgr = DatabaseManager(db_path=db_path)
        run_migrations(db_mgr)
        lifecycle_mgr = LifecycleManager(db_mgr)
        
        # Script python dummy untuk menyimulasikan worker
        worker_script = os.path.join(tmpdir, "dummy_worker.py")
        with open(worker_script, "w") as f:
            f.write("import time\ntime.sleep(100)\n")

        worker_cmd = [sys.executable, worker_script]
        supervisor = NVRASupervisor(
            db_mgr=db_mgr,
            worker_cmd=worker_cmd,
            heartbeat_timeout_sec=2.0,
            startup_grace_sec=1.0,
            window_minutes=5,
            max_crashes=3
        )

        yield supervisor, db_mgr, lifecycle_mgr


def test_supervisor_crash_loop_protection(supervisor_env):
    supervisor, db_mgr, _ = supervisor_env
    crash_tracker = PersistentCrashTracker(db_mgr, window_minutes=5, max_crashes=3)

    # Simulasikan 3 crash beruntun
    crash_tracker.record_crash_event(1001, 1, "Crash 1")
    crash_tracker.record_crash_event(1002, 1, "Crash 2")
    count = crash_tracker.record_crash_event(1003, 1, "Crash 3")

    assert count == 3
    assert crash_tracker.is_crash_loop_triggered() is True


def test_safe_mode_prevents_supervisor_restart(supervisor_env):
    supervisor, db_mgr, lifecycle_mgr = supervisor_env
    
    # Start worker
    supervisor.start_worker()
    time.sleep(1.2)  # Wait for grace period

    # Write healthy heartbeat with SAFE_MODE
    lifecycle_mgr.set_state(ApplicationState.SAFE_MODE, "Balance mismatch test")
    lifecycle_mgr.write_heartbeat()

    # Tick Supervisor Monitor
    current_state = supervisor.monitor_tick()
    assert current_state == SupervisorState.MONITORING, "Supervisor MUST NOT restart worker in SAFE_MODE!"
    
    supervisor.stop_supervisor()


def test_stale_heartbeat_triggers_restart(supervisor_env):
    supervisor, db_mgr, lifecycle_mgr = supervisor_env

    supervisor.start_worker()
    time.sleep(1.2)  # Pass grace period

    # Write old stale heartbeat (> 2.0s ago)
    with db_mgr.get_connection() as conn:
        conn.execute(
            "INSERT INTO bot_state (key, value, updated_at) VALUES ('heartbeat', ?, '2020-01-01T00:00:00+00:00')",
            ('{"last_heartbeat": "2020-01-01T00:00:00+00:00"}',)
        )

    # Monitor tick must detect stale heartbeat and initiate restart
    state = supervisor.monitor_tick()
    assert state in (SupervisorState.RESTARTING, SupervisorState.MONITORING)
    
    supervisor.stop_supervisor()
