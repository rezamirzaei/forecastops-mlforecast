from __future__ import annotations

import pandas as pd
from typer.testing import CliRunner

from mlforecast_realworld import cli


class DummyPipeline:
    def __init__(self) -> None:
        self.processed_data_path = type("P", (), {"exists": lambda self: True})()
        self.model_path = type("P", (), {"exists": lambda self: True})()
        self.training_frame = pd.DataFrame({"unique_id": ["A"], "ds": [pd.Timestamp("2024-01-01")]})

    def prepare_training_data(self, download: bool = True):  # noqa: ARG002
        return pd.DataFrame({"unique_id": ["A"], "ds": [pd.Timestamp("2024-01-01")]})

    def fit(self, frame):  # noqa: ARG002
        return pd.DataFrame({"x": [1]})

    def save_model(self):
        return None

    def load_model(self):
        return None

    def _require_training_frame(self):
        return self.training_frame

    def run_full_pipeline(self, download: bool = True):  # noqa: ARG002
        return {"summary": {"rows": 1}}

    def forecast(self, horizon: int):  # noqa: ARG002
        return pd.DataFrame(
            {
                "unique_id": ["A"],
                "ds": [pd.Timestamp("2024-01-02")],
                "lin_reg": [1.0],
            }
        )

    def run_lightgbm_cv(self, frame):  # noqa: ARG002
        return pd.DataFrame(
            {
                "unique_id": ["A"],
                "ds": [pd.Timestamp("2024-01-02")],
                "lin_reg": [1.0],
            }
        )


def test_cli_commands(monkeypatch) -> None:
    monkeypatch.setattr(cli, "_get_pipeline", lambda: DummyPipeline())
    runner = CliRunner()
    for command in ["download", "train", "forecast", "run-all", "lgbm-cv"]:
        result = runner.invoke(cli.app, [command])
        assert result.exit_code == 0
