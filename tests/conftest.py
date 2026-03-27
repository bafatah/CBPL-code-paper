from __future__ import annotations

from pathlib import Path
import sys
import types


ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

package = types.ModuleType("cbpl_paper")
package.__path__ = [str(ROOT)]
sys.modules.setdefault("cbpl_paper", package)
