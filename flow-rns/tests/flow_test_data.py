from __future__ import annotations

from pathlib import Path

from reinforcens.data import SPLIT_FILENAMES


def write_tiny_dataset(root: Path) -> Path:
    """Create the small click/exposure fixture used by Flow-RNS unit tests."""

    root.mkdir(parents=True, exist_ok=True)
    rows = {
        "train": [
            "0,1,1,event_click,0\n",
            "0,2|3|1,2,list_show,0\n",
            "0,6,3,event_click,1\n",
            "1,4,1,event_click,0\n",
            "1,5|7,2,list_show,0\n",
            "2,8,1,event_click,0\n",
        ],
        "validation": [
            "0,4,1,event_click,0\n",
            "0,5|7,2,list_show,0\n",
            "1,2,1,event_click,0\n",
            "1,3|6,2,list_show,0\n",
        ],
        "test": [
            "0,8,1,event_click,0\n",
            "0,0|9,2,list_show,0\n",
            "1,1,1,event_click,0\n",
            "1,8|9,2,list_show,0\n",
        ],
    }
    for split, values in rows.items():
        (root / SPLIT_FILENAMES[split]).write_text(
            "".join(values), encoding="utf-8"
        )
    return root
