"""
GrepRAG test package.
"""

from pathlib import Path
import sys

# Add parent directory to path for imports
sys_path_parent = Path(__file__).parent.parent
if str(sys_path_parent) not in sys.path:
    sys.path.insert(0, str(sys_path_parent))
