
import os
import sys
import runpy

here = os.path.dirname(os.path.abspath(__file__))
sys.argv = [sys.argv[0], "--models", "ResNet50"] + sys.argv[1:]
runpy.run_path(os.path.join(here, "run_grouped_experiments.py"), run_name="__main__")
