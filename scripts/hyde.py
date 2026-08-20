"""CLI entry point. Implementation lives in newbieduo.retrieval.hyde.

Kept so the documented command works after the move to the four-layer package;
this file deliberately contains no logic.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from newbieduo.retrieval.hyde import main  # noqa: E402

if __name__ == "__main__":
    main()
