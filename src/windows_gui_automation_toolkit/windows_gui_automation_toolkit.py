"""
Windows GUI Automation Toolkit (LLM-free, deterministic)

Repository title target: "windows-gui-automation-toolkit"

What this provides (single-file toolkit):
- Window management (Windows/Win32): list + focus by title substring
- Reliable clipboard workflows (copy/select-all/copy with verification)
- Keyboard + mouse actions via PyAutoGUI
- Screen utilities: screenshot, locate image, wait for image, click image
- Deterministic script runner: execute JSON "steps" with vars, retries,
  if/else, repeat, wait_until, assert

This toolkit does NOT call any LLM or external web/API services.
It is designed to be used as:
1) a Python library (SmartAgent + ScriptRunner), or
2) a CLI runner for JSON scripts.

Safety:
- This library controls your GUI. Prefer a sandbox/test machine.
- PyAutoGUI failsafe is enabled by default: move mouse to top-left to abort.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
import platform
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# External deps (expected):
# - pyautogui
# - pyperclip
#
# Optional (Windows window focus/list):
# - pywin32 (win32gui, win32con)
#
# Optional (better image confidence matching):
# - opencv-python (needed by pyautogui.locateOnScreen with confidence=...)


log = logging.getLogger("windows_gui_automation_toolkit")


def _is_windows() -> bool:
    return platform.system().lower().startswith("win")


def _sleep(seconds: float) -> None:
    if seconds <= 0:
        return
    time.sleep(seconds)


def _now() -> float:
    return time.time()


def _read_json(path: Union[str, Path]) -> Dict[str, Any]:
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8"))


def _safe_str(x: Any) -> str:
    return "" if x is None else str(x)


def _substitute_vars(obj: Any, vars_: Dict[str, Any]) -> Any:
    """
    Recursively substitute ${VAR} in strings using vars_ dict.
    Non-strings are returned as-is (but traversed for list/dict).
    """
    if isinstance(obj, str):
        def repl(m: re.Match) -> str:
            key = m.group(1)
            return _safe_str(vars_.get(key, ""))
        return re.sub(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}", repl, obj)
    if isinstance(obj, list):
        return [_substitute_vars(x, vars_) for x in obj]
    if isinstance(obj, dict):
        return {k: _substitute_vars(v, vars_) for k, v in obj.items()}
    return obj


def _configure_default_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


@dataclass
class AgentConfig:
    pause: float = 0.05              # global pyautogui pause
    failsafe: bool = True            # pyautogui failsafe
    default_timeout: float = 10.0
    default_poll: float = 0.25
    clipboard_verify_timeout: float = 3.0
    clipboard_verify_poll: float = 0.05


class SmartAgent:
    """
    GUI automation primitives. LLM-free.

    Window focus/list is only supported on Windows AND if pywin32 is installed.
    """

    def __init__(self, config: Optional[AgentConfig] = None, dry_run: bool = False):
        self.config = config or AgentConfig()
        self.dry_run = dry_run

        try:
            import pyautogui  # type: ignore
            pyautogui.PAUSE = self.config.pause
            pyautogui.FAILSAFE = self.config.failsafe
            self._pyautogui = pyautogui
        except Exception as e:
            raise RuntimeError("pyautogui is required but failed to import") from e

        try:
            import pyperclip  # type: ignore
            self._pyperclip = pyperclip
        except Exception as e:
            raise RuntimeError("pyperclip is required but failed to import") from e

        self._win32_available = False
        self._win32gui = None
        self._win32con = None
        if _is_windows():
            try:
                import win32gui  # type: ignore
                import win32con  # type: ignore
                self._win32_available = True
                self._win32gui = win32gui
                self._win32con = win32con
            except Exception:
                # Not fatal; focus/list will be unavailable.
                self._win32_available = False

    # ---------------------------
    # Window utilities (Windows)
    # ---------------------------

    def list_windows(self) -> List[Tuple[int, str]]:
        """
        Returns list of (hwnd, title). Windows only + pywin32 required.
        """
        if not (_is_windows() and self._win32_available):
            return []
        win32gui = self._win32gui
        assert win32gui is not None

        results: List[Tuple[int, str]] = []

        def enum_handler(hwnd: int, _: Any) -> None:
            if win32gui.IsWindowVisible(hwnd):
                title = win32gui.GetWindowText(hwnd) or ""
                if title.strip():
                    results.append((hwnd, title))

        win32gui.EnumWindows(enum_handler, None)
        return results

    def focus_window(self, title_substring: str, timeout: Optional[float] = None, raise_on_fail: bool = False) -> bool:
        """
        Focus the first visible window whose title contains title_substring.
        Windows only + pywin32 required.
        """
        if not (_is_windows() and self._win32_available):
            msg = "focus_window requires Windows + pywin32 (win32gui/win32con)."
            if raise_on_fail:
                raise RuntimeError(msg)
            log.warning(msg)
            return False

        win32gui = self._win32gui
        win32con = self._win32con
        assert win32gui is not None and win32con is not None

        t0 = _now()
        to = timeout if timeout is not None else self.config.default_timeout
        needle = (title_substring or "").lower()

        while _now() - t0 <= to:
            for hwnd, title in self.list_windows():
                if needle in title.lower():
                    if self.dry_run:
                        log.info("[dry-run] focus_window: %s (hwnd=%s)", title, hwnd)
                        return True
                    try:
                        # Restore if minimized, then bring to foreground
                        win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
                        win32gui.SetForegroundWindow(hwnd)
                        return True
                    except Exception as e:
                        log.warning("Failed to focus window '%s' (hwnd=%s): %s", title, hwnd, e)
            _sleep(0.2)

        if raise_on_fail:
            raise RuntimeError(f"Window not found/focus failed: {title_substring}")
        return False

    # ---------------------------
    # Clipboard utilities
    # ---------------------------

    def get_clipboard_text(self) -> str:
        return _safe_str(self._pyperclip.paste())

    def set_clipboard_text(self, text: str) -> None:
        if self.dry_run:
            log.info("[dry-run] set_clipboard_text: %r", text[:80])
            return
        self._pyperclip.copy(text)

    def wait_for_clipboard_change(self, previous: str, timeout: Optional[float] = None, poll: Optional[float] = None) -> str:
        """
        Wait until clipboard differs from 'previous'. Returns new clipboard text.
        """
        to = timeout if timeout is not None else self.config.clipboard_verify_timeout
        pl = poll if poll is not None else self.config.clipboard_verify_poll
        t0 = _now()
        while _now() - t0 <= to:
            cur = self.get_clipboard_text()
            if cur != previous:
                return cur
            _sleep(pl)
        return self.get_clipboard_text()

    # ---------------------------
    # Keyboard / mouse actions
    # ---------------------------

    def hotkey(self, *keys: str) -> None:
        if self.dry_run:
            log.info("[dry-run] hotkey: %s", keys)
            return
        self._pyautogui.hotkey(*keys)

    def press(self, key: str) -> None:
        if self.dry_run:
            log.info("[dry-run] press: %s", key)
            return
        self._pyautogui.press(key)

    def type_text(self, text: str, interval: float = 0.0) -> None:
        if self.dry_run:
            log.info("[dry-run] type_text: %r", text[:120])
            return
        self._pyautogui.write(text, interval=interval)

    def paste_text(self, text: str) -> None:
        """
        Paste via clipboard + Ctrl+V. More reliable for non-ASCII.
        """
        prev = self.get_clipboard_text()
        self.set_clipboard_text(text)
        if self.dry_run:
            log.info("[dry-run] paste_text: %r", text[:120])
            return
        self.hotkey("ctrl", "v")
        # Best-effort restore
        _sleep(0.05)
        self.set_clipboard_text(prev)

    def move_to(self, x: int, y: int, duration: float = 0.0) -> None:
        if self.dry_run:
            log.info("[dry-run] move_to: (%s,%s) dur=%s", x, y, duration)
            return
        self._pyautogui.moveTo(x, y, duration=duration)

    def click(self, x: Optional[int] = None, y: Optional[int] = None, clicks: int = 1, button: str = "left") -> None:
        if self.dry_run:
            log.info("[dry-run] click: (%s,%s) clicks=%s button=%s", x, y, clicks, button)
            return
        self._pyautogui.click(x=x, y=y, clicks=clicks, button=button)

    def double_click(self, x: Optional[int] = None, y: Optional[int] = None, button: str = "left") -> None:
        self.click(x=x, y=y, clicks=2, button=button)

    def right_click(self, x: Optional[int] = None, y: Optional[int] = None) -> None:
        self.click(x=x, y=y, clicks=1, button="right")

    def drag_to(self, x: int, y: int, duration: float = 0.2, button: str = "left") -> None:
        if self.dry_run:
            log.info("[dry-run] drag_to: (%s,%s) dur=%s button=%s", x, y, duration, button)
            return
        self._pyautogui.dragTo(x, y, duration=duration, button=button)

    def scroll(self, clicks: int) -> None:
        if self.dry_run:
            log.info("[dry-run] scroll: %s", clicks)
            return
        self._pyautogui.scroll(clicks)

    def screenshot(self, path: Union[str, Path]) -> str:
        """
        Take screenshot to path. Returns string path.
        """
        p = str(path)
        if self.dry_run:
            log.info("[dry-run] screenshot: %s", p)
            return p
        img = self._pyautogui.screenshot()
        Path(p).parent.mkdir(parents=True, exist_ok=True)
        img.save(p)
        return p

    # ---------------------------
    # Copy workflows
    # ---------------------------

    def select_all_and_copy(self, timeout: Optional[float] = None) -> str:
        """
        Ctrl+A, Ctrl+C and verify clipboard changes (best-effort).
        """
        prev = self.get_clipboard_text()
        if self.dry_run:
            log.info("[dry-run] select_all_and_copy")
            return prev

        self.hotkey("ctrl", "a")
        _sleep(0.05)
        self.hotkey("ctrl", "c")

        new = self.wait_for_clipboard_change(prev, timeout=timeout)
        return new

    # ---------------------------
    # Image-based actions
    # ---------------------------

    def locate_center(self, image_path: Union[str, Path], confidence: Optional[float] = None, region: Optional[Tuple[int, int, int, int]] = None) -> Optional[Tuple[int, int]]:
        """
        Returns (x,y) center of image match or None.
        If confidence is provided, OpenCV is typically required by PyAutoGUI.
        If OpenCV is missing, we retry without confidence.
        """
        img = str(image_path)
        if self.dry_run:
            log.info("[dry-run] locate_center: %s conf=%s region=%s", img, confidence, region)
            return (0, 0)

        try:
            if confidence is None:
                box = self._pyautogui.locateOnScreen(img, region=region)
            else:
                box = self._pyautogui.locateOnScreen(img, confidence=confidence, region=region)
        except Exception as e:
            # Common: "confidence" requires opencv; retry without confidence.
            log.debug("locateOnScreen failed (maybe missing opencv). Retrying without confidence. err=%s", e)
            try:
                box = self._pyautogui.locateOnScreen(img, region=region)
            except Exception as e2:
                log.warning("locateOnScreen failed: %s", e2)
                return None

        if box is None:
            return None
        center = self._pyautogui.center(box)
        return (int(center.x), int(center.y))

    def wait_for_image(self, image_path: Union[str, Path], timeout: Optional[float] = None, poll: Optional[float] = None,
                       confidence: Optional[float] = None, region: Optional[Tuple[int, int, int, int]] = None) -> Optional[Tuple[int, int]]:
        to = timeout if timeout is not None else self.config.default_timeout
        pl = poll if poll is not None else self.config.default_poll
        t0 = _now()
        while _now() - t0 <= to:
            pos = self.locate_center(image_path, confidence=confidence, region=region)
            if pos is not None:
                return pos
            _sleep(pl)
        return None

    def click_image(self, image_path: Union[str, Path], timeout: Optional[float] = None, poll: Optional[float] = None,
                    confidence: Optional[float] = None, region: Optional[Tuple[int, int, int, int]] = None) -> Optional[Tuple[int, int]]:
        pos = self.wait_for_image(image_path, timeout=timeout, poll=poll, confidence=confidence, region=region)
        if pos is None:
            return None
        self.click(pos[0], pos[1])
        return pos


# ---------------------------
# Deterministic Script Runner
# ---------------------------

Condition = Dict[str, Any]
Step = Dict[str, Any]


class ScriptError(RuntimeError):
    pass


class ScriptRunner:
    """
    Execute a JSON script with deterministic control flow.

    Script format:
    {
      "vars": {"KEY": "VALUE", ...},    # optional
      "steps": [ { "action": "...", ... }, ... ]
    }

    Supported control-flow actions:
      - if: {action:"if", cond:{...}, then:[...], else:[...]}
      - repeat: {action:"repeat", times: N, steps:[...]}
      - wait_until: {action:"wait_until", cond:{...}, timeout:..., poll:...}
      - assert: {action:"assert", cond:{...}, message:"..."}
      - set_var: {action:"set_var", name:"X", value:"..."}  (supports ${} substitution)
      - log: {action:"log", message:"..."}

    Per-step reliability:
      - retries: integer (default 0)
      - retry_delay: seconds (default 0.25)

    Storage:
      - store: "var_name"   -> store return value into vars (if any)
    """

    def __init__(self, agent: SmartAgent):
        self.agent = agent
        self.vars: Dict[str, Any] = {}

    # ---- condition evaluation ----

    def eval_cond(self, cond: Condition) -> bool:
        ctype = cond.get("type")

        if ctype == "image_present":
            img = cond["image"]
            confidence = cond.get("confidence")
            region = cond.get("region")
            region_t = tuple(region) if isinstance(region, list) else region
            pos = self.agent.locate_center(img, confidence=confidence, region=region_t)
            return pos is not None

        if ctype == "clipboard_contains":
            needle = _safe_str(cond.get("text", ""))
            return needle in self.agent.get_clipboard_text()

        if ctype == "var_equals":
            name = cond["name"]
            value = cond.get("value")
            return self.vars.get(name) == value

        if ctype == "window_exists":
            title = _safe_str(cond.get("title", ""))
            needle = title.lower()
            for _, t in self.agent.list_windows():
                if needle in t.lower():
                    return True
            return False

        if ctype == "always":
            return True

        if ctype == "not":
            return not self.eval_cond(cond["cond"])

        if ctype == "and":
            return all(self.eval_cond(x) for x in cond.get("conds", []))

        if ctype == "or":
            return any(self.eval_cond(x) for x in cond.get("conds", []))

        raise ScriptError(f"Unknown condition type: {ctype}")

    # ---- execution ----

    def run(self, script: Dict[str, Any]) -> Dict[str, Any]:
        base_vars = script.get("vars", {}) or {}
        if not isinstance(base_vars, dict):
            raise ScriptError("'vars' must be an object/dict")
        self.vars = dict(base_vars)
        steps = script.get("steps", [])
        if not isinstance(steps, list):
            raise ScriptError("'steps' must be a list")
        self._run_steps(steps)
        return self.vars

    def _run_steps(self, steps: List[Step]) -> None:
        for step in steps:
            self._run_step(step)

    def _run_step(self, step: Step) -> None:
        if not isinstance(step, dict):
            raise ScriptError("Each step must be an object/dict")

        # Apply ${VAR} substitution to the entire step (except we keep action key stable)
        action = step.get("action")
        if not action:
            raise ScriptError("Step missing 'action'")
        if not isinstance(action, str):
            raise ScriptError("'action' must be a string")

        step_sub = _substitute_vars(step, self.vars)
        step_sub["action"] = action  # preserve

        retries = int(step_sub.get("retries", 0) or 0)
        retry_delay = float(step_sub.get("retry_delay", 0.25) or 0.25)

        last_err: Optional[Exception] = None
        for attempt in range(retries + 1):
            try:
                rv = self._dispatch(step_sub)
                store = step_sub.get("store")
                if store:
                    self.vars[str(store)] = rv
                return
            except Exception as e:
                last_err = e
                if attempt < retries:
                    log.warning("Step failed (attempt %s/%s) action=%s err=%s", attempt + 1, retries + 1, action, e)
                    _sleep(retry_delay)
                else:
                    break
        raise ScriptError(f"Step failed after {retries+1} attempts: {action}. Last error: {last_err}")

    def _dispatch(self, step: Step) -> Any:
        action = step["action"]

        # Control flow
        if action == "if":
            cond = step["cond"]
            then_steps = step.get("then", []) or []
            else_steps = step.get("else", []) or []
            ok = self.eval_cond(cond)
            self._run_steps(then_steps if ok else else_steps)
            return ok

        if action == "repeat":
            times = int(step.get("times", 1) or 1)
            inner = step.get("steps", []) or []
            for _ in range(times):
                self._run_steps(inner)
            return times

        if action == "wait_until":
            cond = step["cond"]
            timeout = float(step.get("timeout", self.agent.config.default_timeout) or self.agent.config.default_timeout)
            poll = float(step.get("poll", self.agent.config.default_poll) or self.agent.config.default_poll)
            t0 = _now()
            while _now() - t0 <= timeout:
                if self.eval_cond(cond):
                    return True
                _sleep(poll)
            return False

        if action == "assert":
            cond = step["cond"]
            msg = step.get("message", "Assertion failed")
            if not self.eval_cond(cond):
                raise ScriptError(str(msg))
            return True

        if action == "set_var":
            name = str(step["name"])
            value = step.get("value")
            self.vars[name] = value
            return value

        if action == "log":
            msg = str(step.get("message", ""))
            log.info("[script] %s", msg)
            return msg

        # Agent actions
        if action == "focus_window":
            title = str(step["title"])
            timeout = step.get("timeout")
            raise_on_fail = bool(step.get("raise_on_fail", False))
            return self.agent.focus_window(title, timeout=timeout, raise_on_fail=raise_on_fail)

        if action == "hotkey":
            keys = step.get("keys") or []
            return self.agent.hotkey(*list(keys))

        if action == "press":
            return self.agent.press(str(step["key"]))

        if action == "type_text":
            return self.agent.type_text(str(step.get("text", "")), interval=float(step.get("interval", 0.0) or 0.0))

        if action == "paste_text":
            return self.agent.paste_text(str(step.get("text", "")))

        if action == "move_to":
            return self.agent.move_to(int(step["x"]), int(step["y"]), duration=float(step.get("duration", 0.0) or 0.0))

        if action == "click":
            x = step.get("x")
            y = step.get("y")
            clicks = int(step.get("clicks", 1) or 1)
            button = str(step.get("button", "left"))
            return self.agent.click(None if x is None else int(x), None if y is None else int(y), clicks=clicks, button=button)

        if action == "double_click":
            x = step.get("x")
            y = step.get("y")
            button = str(step.get("button", "left"))
            return self.agent.double_click(None if x is None else int(x), None if y is None else int(y), button=button)

        if action == "right_click":
            x = step.get("x")
            y = step.get("y")
            return self.agent.right_click(None if x is None else int(x), None if y is None else int(y))

        if action == "drag_to":
            return self.agent.drag_to(int(step["x"]), int(step["y"]), duration=float(step.get("duration", 0.2) or 0.2), button=str(step.get("button", "left")))

        if action == "scroll":
            return self.agent.scroll(int(step.get("clicks", 0) or 0))

        if action == "sleep":
            secs = float(step.get("seconds", 0.0) or 0.0)
            _sleep(secs)
            return secs

        if action == "screenshot":
            path = step.get("path") or "screenshots/screen.png"
            return self.agent.screenshot(path)

        if action == "locate_center":
            img = step["image"]
            confidence = step.get("confidence")
            region = step.get("region")
            region_t = tuple(region) if isinstance(region, list) else region
            return self.agent.locate_center(img, confidence=confidence, region=region_t)

        if action == "wait_for_image":
            img = step["image"]
            timeout = step.get("timeout")
            poll = step.get("poll")
            confidence = step.get("confidence")
            region = step.get("region")
            region_t = tuple(region) if isinstance(region, list) else region
            return self.agent.wait_for_image(img, timeout=timeout, poll=poll, confidence=confidence, region=region_t)

        if action == "click_image":
            img = step["image"]
            timeout = step.get("timeout")
            poll = step.get("poll")
            confidence = step.get("confidence")
            region = step.get("region")
            region_t = tuple(region) if isinstance(region, list) else region
            return self.agent.click_image(img, timeout=timeout, poll=poll, confidence=confidence, region=region_t)

        if action == "get_clipboard":
            return self.agent.get_clipboard_text()

        if action == "set_clipboard":
            text = str(step.get("text", ""))
            self.agent.set_clipboard_text(text)
            return text

        if action == "select_all_and_copy":
            timeout = step.get("timeout")
            return self.agent.select_all_and_copy(timeout=timeout)

        raise ScriptError(f"Unknown action: {action}")


# ---------------------------
# CLI
# ---------------------------

def _cli_list_windows(agent: SmartAgent) -> int:
    wins = agent.list_windows()
    if not wins:
        print("No windows (or pywin32 not installed / not Windows).")
        return 1
    for hwnd, title in wins:
        print(f"{hwnd}\t{title}")
    return 0


def _cli_run(agent: SmartAgent, script_path: str, vars_kv: List[str]) -> int:
    script = _read_json(script_path)
    # Merge CLI vars into script vars
    script_vars = script.get("vars", {}) or {}
    if not isinstance(script_vars, dict):
        script_vars = {}
    for kv in vars_kv:
        if "=" not in kv:
            continue
        k, v = kv.split("=", 1)
        script_vars[k] = v
    script["vars"] = script_vars

    runner = ScriptRunner(agent)
    out_vars = runner.run(script)
    # Print final vars as JSON for chaining
    print(json.dumps(out_vars, ensure_ascii=False, indent=2))
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    parser = argparse.ArgumentParser(prog="windows_gui_automation_toolkit", add_help=True)
    parser.add_argument("--log-level", default="INFO", help="DEBUG, INFO, WARNING, ERROR")
    parser.add_argument("--dry-run", action="store_true", help="Print actions without controlling GUI")

    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list-windows", help="List visible windows (Windows + pywin32 required)")

    runp = sub.add_parser("run", help="Run a JSON automation script (LLM-free)")
    runp.add_argument("script", help="Path to script.json")
    runp.add_argument("--var", action="append", default=[], help="Inject variables KEY=VALUE (repeatable)")

    args = parser.parse_args(argv)

    _configure_default_logging(getattr(logging, str(args.log_level).upper(), logging.INFO))

    agent = SmartAgent(dry_run=bool(args.dry_run))

    if args.cmd == "list-windows":
        return _cli_list_windows(agent)
    if args.cmd == "run":
        return _cli_run(agent, args.script, list(args.var))

    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
