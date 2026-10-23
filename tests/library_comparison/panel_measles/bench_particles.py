"""4-town panel: runs ../measles/bench_particles.py with --panel (same CLI; results go to this folder's results/)."""

import os
import sys

script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "measles", "bench_particles.py")
os.execv(sys.executable, [sys.executable, os.path.normpath(script), "--panel", *sys.argv[1:]])
