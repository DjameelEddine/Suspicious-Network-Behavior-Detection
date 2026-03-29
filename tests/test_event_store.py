from pathlib import Path

from src.event_store import EventStore


def test_event_store_add_and_lookup():
    store = EventStore(max_events=10)
    store.add_event({"flow_id": "flow-1", "status": "processed"})
    store.add_event({"flow_id": "flow-2", "status": "invalid"})

    event = store.get_by_flow_id("flow-2")
    assert event is not None
    assert event["status"] == "invalid"


def test_event_store_export_csv_json(tmp_path: Path):
    store = EventStore(max_events=10)
    store.add_event({"flow_id": "flow-1", "status": "processed", "confidence": 0.92})

    csv_path = store.export_csv(tmp_path / "events.csv")
    json_path = store.export_json(tmp_path / "events.json")

    assert csv_path.exists()
    assert json_path.exists()
