"""
MODULE: tokocrypto_bot.recovery.single_instance
DESCRIPTION: Cross-process Single Instance Lock using Named Mutex (Windows) and Flock (POSIX).
"""

import os
import sys
import logging
from typing import Optional

logger = logging.getLogger("NVRA.SingleInstance")

class InstanceAlreadyRunningException(Exception):
    """Exception jika instance bot lain sudah berjalan."""
    pass

class SingleInstanceLock:
    def __init__(self, lock_name: str = "NVRA_TOKOCRYPTO_TRADING_INSTANCE"):
        self.lock_name = lock_name
        self._mutex = None
        self._file_handle = None
        self.is_acquired = False

    def acquire(self) -> bool:
        """Mengunci instance. Melempar Exception jika instance lain sudah aktif."""
        if self.is_acquired:
            return True

        if sys.platform == "win32":
            import ctypes
            from ctypes import wintypes

            kernel32 = ctypes.windll.kernel32
            
            # CreateMutexW(lpMutexAttributes, bInitialOwner, lpName)
            CreateMutexW = kernel32.CreateMutexW
            CreateMutexW.argtypes = [wintypes.LPVOID, wintypes.BOOL, wintypes.LPCWSTR]
            CreateMutexW.restype = wintypes.HANDLE

            GetLastError = kernel32.GetLastError
            GetLastError.restype = wintypes.DWORD

            ERROR_ALREADY_EXISTS = 183

            mutex = CreateMutexW(None, False, self.lock_name)
            if not mutex:
                raise RuntimeError("Failed to create system mutex.")

            if GetLastError() == ERROR_ALREADY_EXISTS:
                kernel32.CloseHandle(mutex)
                err_msg = f"SINGLE INSTANCE LOCK FAILED: Another instance with mutex '{self.lock_name}' is already running!"
                logger.critical(err_msg)
                raise InstanceAlreadyRunningException(err_msg)

            self._mutex = mutex
            self.is_acquired = True
            logger.info(f"Acquired Windows Named Mutex Lock: [{self.lock_name}] (PID: {os.getpid()})")
            return True
        else:
            # POSIX Fallback using fcntl
            import fcntl
            lock_dir = os.path.expanduser("~/.nvra")
            os.makedirs(lock_dir, exist_ok=True)
            lock_file = os.path.join(lock_dir, f"{self.lock_name}.lock")

            try:
                self._file_handle = open(lock_file, "w")
                fcntl.flock(self._file_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                self._file_handle.write(str(os.getpid()))
                self._file_handle.flush()
                self.is_acquired = True
                logger.info(f"Acquired POSIX File Lock: [{lock_file}] (PID: {os.getpid()})")
                return True
            except IOError:
                err_msg = f"SINGLE INSTANCE LOCK FAILED: Lock file '{lock_file}' is locked by another process."
                logger.critical(err_msg)
                raise InstanceAlreadyRunningException(err_msg)

    def release(self) -> None:
        """Melepas kunci instance secara bersih saat shutdown."""
        if not self.is_acquired:
            return

        if sys.platform == "win32" and self._mutex:
            import ctypes
            ctypes.windll.kernel32.CloseHandle(self._mutex)
            self._mutex = None
            logger.info("Released Windows Named Mutex Lock.")
        elif self._file_handle:
            import fcntl
            fcntl.flock(self._file_handle, fcntl.LOCK_UN)
            self._file_handle.close()
            self._file_handle = None
            logger.info("Released POSIX File Lock.")

        self.is_acquired = False
