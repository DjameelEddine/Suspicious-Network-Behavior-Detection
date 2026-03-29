from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from typing import Deque, Dict, Iterable, List, Optional

import pandas as pd


class EventStore:
    def __init__(self, max_events: int = 5000) -> None:
        self._events: Deque[Dict[str, object]] = deque(maxlen=max_events)

    def __len__(self) -> int:
        return len(self._events)

    def clear(self) -> None:
        self._events.clear()

    def add_event(self, event: Dict[str, object]) -> None:
        self._events.append(event)

    def add_events(self, events: Iterable[Dict[str, object]]) -> None:
        for event in events:
            self.add_event(event)

    def recent(self, limit: int = 200) -> List[Dict[str, object]]:
        if limit <= 0:
            return []
        return list(self._events)[-limit:]

    def as_dataframe(self, limit: Optional[int] = None) -> pd.DataFrame:
        events = list(self._events)
        if limit is not None and limit > 0:
            events = events[-limit:]
        if not events:
            return pd.DataFrame()
        return pd.DataFrame(events)

    def get_by_flow_id(self, flow_id: object) -> Optional[Dict[str, object]]:
        lookup = str(flow_id)
        for event in reversed(self._events):
            if str(event.get("flow_id")) == lookup:
                return event
        return None

    def export_csv(self, output_path: Path) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df = self.as_dataframe()
        df.to_csv(output_path, index=False)
        return output_path

    def export_json(self, output_path: Path) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        events = list(self._events)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(events, handle, indent=2, ensure_ascii=True)
        return output_path
