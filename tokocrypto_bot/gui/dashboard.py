"""
MODULE: tokocrypto_bot.gui.dashboard
DESCRIPTION: Decoupled Control & Observation Plane (PyQt6 Dashboard) for NVRA Engine.
"""

import sys
import json
from datetime import datetime
from typing import Optional

try:
    from PyQt6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QLabel, QPushButton, QTableWidget, QTableWidgetItem, QGroupBox,
        QGridLayout, QMessageBox, QFrame
    )
    from PyQt6.QtCore import QTimer, Qt, pyqtSignal
    from PyQt6.QtGui import QColor, QFont
except ImportError:
    # Safe Fallback jika run tanpa GUI dependency di headless server
    pass

from tokocrypto_bot.persistence.database import DatabaseManager
from tokocrypto_bot.persistence.lifecycle_state import ApplicationState


class NVRADashboardWindow(QMainWindow):
    def __init__(self, db_mgr: DatabaseManager):
        super().__init__()
        self.db = db_mgr
        self.setWindowTitle("NVRA Tokocrypto Quantitative Trading Engine v2026.5.9")
        self.resize(1100, 700)

        self._init_ui()
        self._start_state_polling()

    def _init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)

        # 1. Header & Application Status Bar
        header_group = QGroupBox("System Health & Status")
        header_layout = QGridLayout(header_group)

        self.lbl_app_state = QLabel("OFFLINE")
        self.lbl_app_state.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        self.lbl_app_state.setStyleSheet("color: gray;")

        self.lbl_db_health = QLabel("Database: OK [WAL]")
        self.lbl_heartbeat = QLabel("Heartbeat: -")
        self.lbl_unresolved = QLabel("Unresolved Orders: 0")

        header_layout.addWidget(QLabel("Application State:"), 0, 0)
        header_layout.addWidget(self.lbl_app_state, 0, 1)
        header_layout.addWidget(self.lbl_db_health, 0, 2)
        header_layout.addWidget(self.lbl_heartbeat, 1, 0)
        header_layout.addWidget(self.lbl_unresolved, 1, 1)

        main_layout.addWidget(header_group)

        # 2. Control Panel Buttons (Only triggers state intents, NO DIRECT EXCHANGE POST!)
        controls_group = QGroupBox("Engine Controls")
        controls_layout = QHBoxLayout(controls_group)

        self.btn_pause = QPushButton("PAUSE TRADING")
        self.btn_pause.setStyleSheet("background-color: #FFA500; color: white; font-weight: bold;")
        self.btn_pause.clicked.connect(self._on_pause_clicked)

        self.btn_safemode = QPushButton("ENTER SAFE MODE")
        self.btn_safemode.setStyleSheet("background-color: #DC143C; color: white; font-weight: bold;")
        self.btn_safemode.clicked.connect(self._on_safemode_clicked)

        self.btn_reconcile = QPushButton("TRIGGER RECONCILIATION")
        self.btn_reconcile.setStyleSheet("background-color: #4682B4; color: white; font-weight: bold;")
        self.btn_reconcile.clicked.connect(self._on_reconcile_clicked)

        controls_layout.addWidget(self.btn_pause)
        controls_layout.addWidget(self.btn_safemode)
        controls_layout.addWidget(self.btn_reconcile)

        main_layout.addWidget(controls_group)

        # 3. Live Balances & Position Table
        tables_layout = QHBoxLayout()

        pos_group = QGroupBox("Active Positions & Balances")
        pos_vbox = QVBoxLayout(pos_group)
        self.tbl_positions = QTableWidget(0, 4)
        self.tbl_positions.setHorizontalHeaderLabels(["Asset/Symbol", "Total Qty", "Locked Qty", "Avg Price"])
        pos_vbox.addWidget(self.tbl_positions)
        tables_layout.addWidget(pos_group)

        main_layout.addLayout(tables_layout)

    def _start_state_polling(self):
        """Poll database state every 1.5s (Decoupled Read-Only)."""
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._refresh_dashboard)
        self.timer.start(1500)

    def _refresh_dashboard(self):
        conn = self.db.get_connection()
        try:
            cursor = conn.cursor()
            
            # Read App State
            cursor.execute("SELECT value FROM bot_state WHERE key = 'application_state'")
            row_state = cursor.fetchone()
            state_val = row_state[0] if row_state else "OFFLINE"
            self.lbl_app_state.setText(state_val)

            # Style App State
            if state_val in ("TRADING", "READY"):
                self.lbl_app_state.setStyleSheet("color: green;")
            elif state_val in ("SAFE_MODE", "ERROR"):
                self.lbl_app_state.setStyleSheet("color: red;")
            elif state_val == "PAUSED":
                self.lbl_app_state.setStyleSheet("color: orange;")

            # Read Unresolved Orders Count
            cursor.execute("SELECT COUNT(*) FROM orders WHERE status IN ('CREATED', 'SUBMITTING', 'UNKNOWN', 'RECONCILING')")
            unresolved_cnt = cursor.fetchone()[0]
            self.lbl_unresolved.setText(f"Unresolved Orders: {unresolved_cnt}")

        except Exception as e:
            self.lbl_app_state.setText("DB ERROR")
        finally:
            conn.close()

    def _on_pause_clicked(self):
        self._write_gui_command("PAUSE_TRADING")
        QMessageBox.information(self, "Intent Sent", "Sent PAUSE intent to Engine Controller.")

    def _on_safemode_clicked(self):
        reply = QMessageBox.warning(
            self, "Confirm Safe Mode",
            "Are you sure you want to force SAFE_MODE? New orders will be blocked.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.Yes:
            self._write_gui_command("FORCE_SAFE_MODE")

    def _on_reconcile_clicked(self):
        self._write_gui_command("TRIGGER_RECONCILIATION")

    def _write_gui_command(self, cmd_type: str):
        now_str = datetime.utcnow().isoformat()
        with self.db.transaction() as conn:
            conn.execute(
                "INSERT INTO system_events (level, component, message, payload_json, created_at) VALUES (?, ?, ?, ?, ?)",
                ("WARNING", "GUI_CONTROL", f"User triggered action: {cmd_type}", json.dumps({"cmd": cmd_type}), now_str)
            )
