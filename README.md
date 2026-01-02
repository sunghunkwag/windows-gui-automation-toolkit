# Windows GUI Automation Toolkit

A small, practical toolkit for **Windows** GUI automation.

## Features
- Focus a window by title keyword (Win32 user32 APIs).
- Type text via clipboard paste (IME / non-ASCII friendly).
- Helper utilities for hotkeys and copy workflows.

## Install
```bash
pip install -U windows-gui-automation-toolkit
```

## Quick start
```python
from windows_gui_automation_toolkit.smart_agent import SmartAgent

agent = SmartAgent()
agent.wait_for_window("Chrome", timeout=5)
agent.type_text("https://github.com")
agent.hotkey("enter")
```

## Safety
This project controls keyboard/mouse input. Use on a test machine if possible.

## License
MIT
