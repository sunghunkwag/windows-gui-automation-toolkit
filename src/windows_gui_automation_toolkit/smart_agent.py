"""Windows GUI Automation Toolkit.

Core utilities for:
- focusing windows by title (Windows / Win32 only)
- clipboard-based text input
- common hotkey and copy workflows

Security/Safety note:
This library controls your GUI. Prefer a sandbox/test machine.
PyAutoGUI failsafe: move mouse to top-left corner to abort.
"""

from __future__ import annotations

import logging
import sys
import time
from dataclasses import dataclass
from typing import Optional

import ctypes
from ctypes import wintypes

import pyautogui
import pyperclip

logger = logging.getLogger(__name__)


class ClipboardGuard:
    """Context manager that restores clipboard content after use."""

    def __init__(self) -> None:
        self._before: Optional[str] = None

    def __enter__(self) -> "ClipboardGuard":
        try:
            self._before = pyperclip.paste()
        except Exception:
            self._before = None
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._before is not None:
            try:
                pyperclip.copy(self._before)
            except Exception:
                pass


@dataclass(frozen=True)
class FocusResult:
    success: bool
    hwnd: Optional[int] = None


class WindowController:
    """Window focusing utilities using Win32 user32 APIs (Windows only)."""

    SW_RESTORE = 9

    @staticmethod
    def _require_windows() -> None:
        if sys.platform != "win32":
            raise NotImplementedError("Window focusing is only implemented for Windows (win32).")

    @staticmethod
    def bring_window_to_front(window_title_keyword: str) -> FocusResult:
        """Bring the first visible window whose title contains the keyword to the foreground."""

        WindowController._require_windows()

        keyword = (window_title_keyword or "").strip()
        if not keyword:
            raise ValueError("window_title_keyword must be a non-empty string.")

        user32 = ctypes.windll.user32
        found_hwnd: Optional[int] = None

        EnumWindowsProc = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

        def enum_proc(hwnd: wintypes.HWND, lparam: wintypes.LPARAM) -> wintypes.BOOL:
            nonlocal found_hwnd

            if not user32.IsWindowVisible(hwnd):
                return True

            length = user32.GetWindowTextLengthW(hwnd)
            if length <= 0:
                return True

            buff = ctypes.create_unicode_buffer(length + 1)
            user32.GetWindowTextW(hwnd, buff, length + 1)
            title = buff.value or ""

            if keyword.lower() in title.lower():
                found_hwnd = int(hwnd)
                return False
            return True

        user32.EnumWindows(EnumWindowsProc(enum_proc), 0)

        if not found_hwnd:
            logger.warning("Window not found (keyword=%r)", keyword)
            return FocusResult(False, None)

        user32.ShowWindow(found_hwnd, WindowController.SW_RESTORE)
        user32.SetForegroundWindow(found_hwnd)
        time.sleep(0.2)

        logger.info("Window focused (keyword=%r, hwnd=%s)", keyword, found_hwnd)
        return FocusResult(True, found_hwnd)


class SmartAgent:
    """A practical GUI automation agent."""

    def __init__(self, *, pause: float = 0.05, failsafe: bool = True) -> None:
        pyautogui.PAUSE = float(pause)
        pyautogui.FAILSAFE = bool(failsafe)
        self.window_ctrl = WindowController()

        logger.info("SmartAgent initialized (pause=%s, failsafe=%s)", pause, failsafe)

    def hotkey(self, *keys: str, sleep: float = 0.1) -> None:
        """Safe hotkey wrapper."""
        try:
            pyautogui.hotkey(*keys)
        except pyautogui.FailSafeException as e:
            logger.error("PyAutoGUI failsafe triggered during hotkey %s: %s", keys, e)
            raise
        time.sleep(float(sleep))

    def type_text(self, text: str, *, interval: float = 0.0, restore_clipboard: bool = True) -> None:
        """Type text by pasting from clipboard."""
        if restore_clipboard:
            with ClipboardGuard():
                pyperclip.copy(text)
                time.sleep(0.05)
                self.hotkey("ctrl", "v", sleep=0.05)
                time.sleep(float(interval))
        else:
            pyperclip.copy(text)
            time.sleep(0.05)
            self.hotkey("ctrl", "v", sleep=0.05)
            time.sleep(float(interval))

        logger.debug("Typed text via clipboard (len=%d)", len(text))

    def focus_window(self, title_keyword: str) -> bool:
        """Focus a window by title keyword (Windows only)."""
        return self.window_ctrl.bring_window_to_front(title_keyword).success

    def wait_for_window(self, title_keyword: str, *, timeout: float = 10.0, poll: float = 0.5) -> bool:
        """Wait until a window becomes focusable."""
        deadline = time.time() + float(timeout)
        while time.time() < deadline:
            try:
                if self.focus_window(title_keyword):
                    return True
            except Exception as e:
                logger.debug("wait_for_window retry due to error: %s", e)
            time.sleep(float(poll))

        logger.error("Timeout waiting for window: %r", title_keyword)
        return False

    def select_all_and_copy(self) -> str:
        """Ctrl+A then Ctrl+C, returning clipboard contents."""
        self.hotkey("ctrl", "a")
        self.hotkey("ctrl", "c", sleep=0.2)
        time.sleep(0.1)
        return pyperclip.paste()


def _configure_default_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


if __name__ == "__main__":
    _configure_default_logging()
    agent = SmartAgent()
    print("Windows GUI Automation Toolkit loaded. (SmartAgent ready)")
