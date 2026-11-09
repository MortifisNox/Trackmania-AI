import pytest


@pytest.fixture(autouse=True)
def _no_personal_settings(tmp_path_factory, monkeypatch):
    """Tests must not pick up a tmai_settings.json in the repo folder, and they
    expect the Windows defaults (camera mode) whatever OS runs them. Tests of
    the Linux behaviour pass the system explicitly."""
    from tmai import launch

    monkeypatch.setattr(launch, "SETTINGS_FILE", tmp_path_factory.mktemp("settings") / "tmai_settings.json")
    monkeypatch.setattr(launch, "host_system", lambda: "Windows")
