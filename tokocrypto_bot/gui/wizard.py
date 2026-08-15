"""
MODULE: tokocrypto_bot.gui.wizard
DESCRIPTION: Graphical Setup Wizard for Encrypted Credentials and System Configuration.
"""

import sys
import logging
from typing import Optional

try:
    from PyQt6.QtWidgets import (
        QApplication, QDialog, QVBoxLayout, QFormLayout, QLineEdit,
        QCheckBox, QComboBox, QPushButton, QMessageBox, QGroupBox, QLabel
    )
    from PyQt6.QtCore import Qt
except ImportError:
    pass

from tokocrypto_bot.security.credential_manager import SecureCredentialStore
from tokocrypto_bot.exchange.tokocrypto_client import TokocryptoDirectClient

logger = logging.getLogger("NVRA.SetupWizard")


class NVRASetupWizardDialog(QDialog):
    def __init__(self, cred_store: Optional[SecureCredentialStore] = None):
        super().__init__()
        self.cred_store = cred_store or SecureCredentialStore()
        self.setWindowTitle("NVRA Trading Engine - Credential & Setup Wizard")
        self.resize(500, 450)
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)

        # 1. Tokocrypto Credentials Group
        ex_group = QGroupBox("Tokocrypto Exchange Credentials")
        ex_form = QFormLayout(ex_group)

        self.txt_t_key = QLineEdit()
        self.txt_t_key.setEchoMode(QLineEdit.EchoMode.Password)
        self.txt_t_secret = QLineEdit()
        self.txt_t_secret.setEchoMode(QLineEdit.EchoMode.Password)

        ex_form.addRow("API Key:", self.txt_t_key)
        ex_form.addRow("API Secret:", self.txt_t_secret)
        layout.addWidget(ex_group)

        # 2. Gemini AI Administrator Group
        ai_group = QGroupBox("Gemini AI God Administrator (Optional Evaluator)")
        ai_form = QFormLayout(ai_group)

        self.chk_gemini_enabled = QCheckBox("Enable Gemini Periodic Evaluation")
        self.chk_gemini_enabled.setChecked(True)
        self.txt_g_key = QLineEdit()
        self.txt_g_key.setEchoMode(QLineEdit.EchoMode.Password)

        ai_form.addRow(self.chk_gemini_enabled)
        ai_form.addRow("Gemini API Key:", self.txt_g_key)
        layout.addWidget(ai_group)

        # 3. Action Buttons
        btn_box = QVBoxLayout()
        self.btn_validate = QPushButton("Test Connections (No Log Leak)")
        self.btn_validate.setStyleSheet("background-color: #4682B4; color: white; font-weight: bold;")
        self.btn_validate.clicked.connect(self._on_validate_clicked)

        self.btn_save = QPushButton("Save Encrypted & Continue")
        self.btn_save.setStyleSheet("background-color: #2E8B57; color: white; font-weight: bold;")
        self.btn_save.clicked.connect(self._on_save_clicked)

        btn_box.addWidget(self.btn_validate)
        btn_box.addWidget(self.btn_save)
        layout.addLayout(btn_box)

    def _on_validate_clicked(self):
        key = self.txt_t_key.text().strip()
        secret = self.txt_t_secret.text().strip()

        if not key or not secret:
            QMessageBox.warning(self, "Validation Error", "Please provide both Tokocrypto API Key and Secret.")
            return

        try:
            client = TokocryptoDirectClient(key, secret)
            balances = client.fetch_account_balances()
            QMessageBox.information(
                self, "Success",
                f"Tokocrypto Connection SUCCESSFUL!\nVerified USDT Balance: ${balances.get('USDT', {}).get('free', 0.0):.2f}"
            )
        except Exception as e:
            QMessageBox.critical(self, "Connection Failed", f"Tokocrypto Authentication Failed: {str(e)}")

    def _on_save_clicked(self):
        t_key = self.txt_t_key.text().strip()
        t_secret = self.txt_t_secret.text().strip()
        g_key = self.txt_g_key.text().strip() if self.chk_gemini_enabled.isChecked() else None

        if not t_key or not t_secret:
            QMessageBox.warning(self, "Input Required", "Tokocrypto Key & Secret are required.")
            return

        success = self.cred_store.save_credentials(t_key, t_secret, g_key)
        if success:
            QMessageBox.information(self, "Saved", "Credentials encrypted via Windows DPAPI and stored safely!")
            self.accept()
        else:
            QMessageBox.critical(self, "Error", "Failed to encrypt credentials.")
