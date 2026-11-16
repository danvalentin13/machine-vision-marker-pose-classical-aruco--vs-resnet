"""live.py -- the same as live.py in the project folder, so that  python live.py  also works from inside code/."""
import os
import runpy

runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "live.py"), run_name="__main__")
