"""
MODULE: tokocrypto_bot.security.credential_manager
DESCRIPTION: Windows DPAPI Secure Credential Storage Manager (Zero Plaintext Secrets).
"""

import os
import sys
import base64
import logging
from typing import Tuple, Optional

logger = logging.getLogger("NVRA.Security")

class SecureCredentialStore:
    def __init__(self, service_name: str = "NVRA_Tokocrypto"):
        self.service_name = service_name

    def save_api_credentials(self, api_key: str, api_secret: str) -> bool:
        """Enskripsi dan simpan API key & secret ke Windows Credential / DPAPI Encrypted File."""
        if sys.platform != "win32":
            logger.warning("Non-Windows OS detected. Falling back to environment variables.")
            return False

        try:
            import win32crypt # pywin32 DPAPI binding

            encrypted_key = win32crypt.CryptProtectData(api_key.encode('utf-8'), f"{self.service_name}_KEY", None, None, None, 0)
            encrypted_secret = win32crypt.CryptProtectData(api_secret.encode('utf-8'), f"{self.service_name}_SECRET", None, None, None, 0)

            cred_dir = os.path.expandvars(r"%LOCALAPPDATA%\NVRA\Trading\credentials")
            os.makedirs(cred_dir, exist_ok=True)

            with open(os.path.join(cred_dir, "key.dat"), "wb") as f:
                f.write(encrypted_key)
            with open(os.path.join(cred_dir, "secret.dat"), "wb") as f:
                f.write(encrypted_secret)

            logger.info("API Credentials successfully encrypted and stored via Windows DPAPI.")
            return True
        except Exception as e:
            logger.error(f"Failed to encrypt credentials via DPAPI: {e}")
            return False

    def load_api_credentials(self) -> Tuple[Optional[str], Optional[str]]:
        """Membaca dan men-dekripsi API Credentials di memori saat runtime."""
        cred_dir = os.path.expandvars(r"%LOCALAPPDATA%\NVRA\Trading\credentials")
        key_file = os.path.join(cred_dir, "key.dat")
        secret_file = os.path.join(cred_dir, "secret.dat")

        if not (os.path.exists(key_file) and os.path.exists(secret_file)):
            # Fallback to ENV if DPAPI file not found
            return os.environ.get("TOKOCRYPTO_API_KEY"), os.environ.get("TOKOCRYPTO_API_SECRET")

        try:
            import win32crypt

            with open(key_file, "rb") as f:
                enc_key = f.read()
            with open(secret_file, "rb") as f:
                enc_secret = f.read()

            _, dec_key = win32crypt.CryptUnprotectData(enc_key, None, None, None, 0)
            _, dec_secret = win32crypt.CryptUnprotectData(enc_secret, None, None, None, 0)

            return dec_key.decode('utf-8'), dec_secret.decode('utf-8')
        except Exception as e:
            logger.error(f"Failed to decrypt credentials via DPAPI: {e}")
            return None, None
