from pathlib import Path

import ipyvuetify as v
from IPython.display import HTML, display

from datahasher.front.app import App, get_app_jupyter
from datahasher.front.waiter import Waiter

display(HTML(f"<style>{(Path(__file__).parent / 'style.css').read_text()}</style>"))

v.theme.themes.light.primary = "#00205b"
v.theme.themes.dark.primary = "#383838"

# instantiate the app
app = App()
waiter = Waiter()


# use to launch the app from jupyter
def launch_app(*, v_model: bool = True) -> None:
    display(get_app_jupyter(app, v_model=v_model))
