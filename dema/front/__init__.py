from pathlib import Path

import ipyvuetify as v
from dema.front.app import App
from dema.front.waiter import Waiter
from IPython.display import HTML, display

display(HTML(f"<style>{(Path(__file__).parent / 'style.css').read_text()}</style>"))

v.theme.themes.light.primary = "#00205b"
v.theme.themes.dark.primary = "#383838"

# instantiate the app
waiter = Waiter()

__all__ = ["App"]
