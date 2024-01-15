import sys
from pathlib import Path


path = str(Path(__file__).parent)
if path not in sys.path:
    sys.path.append(str(Path(__file__).parent))
