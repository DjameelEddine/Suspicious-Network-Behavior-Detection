from __future__ import annotations

from pathlib import Path

import pandas as pd


class DataSimulator:
    """Simulates live flow ingestion from a static CSV dataset."""

    def __init__(
        self,
        dataset_path: Path,
        batch_size: int = 10,
        rows_per_second: float = 5.0,
        shuffle: bool = False,
        random_seed: int = 42,
    ) -> None:
        self.dataset_path = Path(dataset_path)
        self.batch_size = max(1, int(batch_size))
        self.rows_per_second = max(0.1, float(rows_per_second))
        self.shuffle = shuffle
        self.random_seed = random_seed

        self._dataframe: pd.DataFrame | None = None
        self._cursor = 0
        self._status = "stopped"

    @property
    def status(self) -> str:
        return self._status

    @property
    def total_rows(self) -> int:
        if self._dataframe is None:
            return 0
        return len(self._dataframe)

    @property
    def cursor(self) -> int:
        return self._cursor

    def progress_ratio(self) -> float:
        if self.total_rows == 0:
            return 0.0
        return min(1.0, self._cursor / self.total_rows)

    def seconds_between_batches(self) -> float:
        return self.batch_size / max(self.rows_per_second, 0.1)

    def set_speed(self, rows_per_second: float) -> None:
        self.rows_per_second = max(0.1, float(rows_per_second))

    def set_batch_size(self, batch_size: int) -> None:
        self.batch_size = max(1, int(batch_size))

    def load(self, force_reload: bool = False) -> None:
        if self._dataframe is not None and not force_reload:
            return

        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {self.dataset_path.name}")

        dataframe = pd.read_csv(self.dataset_path, low_memory=False)
        dataframe.columns = [str(column).strip() for column in dataframe.columns]

        if self.shuffle:
            dataframe = dataframe.sample(frac=1.0, random_state=self.random_seed).reset_index(drop=True)

        if "flow_id" in dataframe.columns:
            dataframe = dataframe.rename(columns={"flow_id": "source_flow_id"})

        dataframe.insert(0, "flow_id", [f"flow-{index + 1}" for index in range(len(dataframe))])

        self._dataframe = dataframe
        self._cursor = 0

    def start(self) -> None:
        self.load()
        self._status = "running"

    def pause(self) -> None:
        if self._status == "running":
            self._status = "paused"

    def resume(self) -> None:
        if self._status == "paused":
            self._status = "running"

    def stop(self, reset_cursor: bool = False) -> None:
        self._status = "stopped"
        if reset_cursor:
            self._cursor = 0

    def reset(self) -> None:
        self._cursor = 0
        self._status = "stopped"

    def next_batch(self) -> pd.DataFrame:
        if self._status != "running":
            return pd.DataFrame()

        if self._dataframe is None:
            self.load()

        assert self._dataframe is not None

        if self._cursor >= len(self._dataframe):
            self._status = "completed"
            return pd.DataFrame()

        end_cursor = min(self._cursor + self.batch_size, len(self._dataframe))
        batch = self._dataframe.iloc[self._cursor:end_cursor].copy()
        self._cursor = end_cursor

        if self._cursor >= len(self._dataframe):
            self._status = "completed"

        return batch
