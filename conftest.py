from pathlib import Path
from typing import Any

from pytest import Config as Config
from pytest import hookimpl as hookimpl


@hookimpl(trylast=True)
def pytest_configure(config: Config) -> None:
    workerinput: Any = getattr(config, "workerinput", None)
    worker_id = "master"
    if isinstance(workerinput, dict):
        worker_id = workerinput.get("workerid")
    assert isinstance(worker_id, str)

    logging_plugin = config.pluginmanager.get_plugin("logging-plugin")
    assert logging_plugin is not None
    logging_plugin.set_log_path(str(Path(config.rootpath) / f"pytest-{worker_id}.log"))
