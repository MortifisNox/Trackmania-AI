from tmai.game import paths


def _prefix_user(root, *parts):
    user = root.joinpath(*parts, "drive_c", "users", "steamuser")
    (user.parent / "Public").mkdir(parents=True, exist_ok=True)
    user.mkdir(parents=True, exist_ok=True)
    return user


def test_finds_the_steam_proton_prefix(tmp_path, monkeypatch):
    monkeypatch.delenv(paths.ENV_VAR, raising=False)
    user = _prefix_user(tmp_path, ".steam", "steam", "steamapps", "compatdata", "2225070", "pfx")
    (user / "OpenplanetNext").mkdir()
    assert paths.game_user_dir(tmp_path, "Linux") == user


def test_finds_a_prefix_in_another_steam_library(tmp_path, monkeypatch):
    monkeypatch.delenv(paths.ENV_VAR, raising=False)
    root = tmp_path / ".local" / "share" / "Steam" / "steamapps"
    root.mkdir(parents=True)
    lib = tmp_path / "games"
    (root / "libraryfolders.vdf").write_text(f'"libraryfolders" {{ "1" {{ "path" "{lib}" }} }}')
    user = _prefix_user(lib, "steamapps", "compatdata", "2225070", "pfx")
    (user / "Documents" / "Trackmania").mkdir(parents=True)
    assert paths.game_user_dir(tmp_path, "Linux") == user


def test_lutris_prefix_and_override(tmp_path, monkeypatch):
    monkeypatch.delenv(paths.ENV_VAR, raising=False)
    user = _prefix_user(tmp_path, "Games", "trackmania")
    (user / "OpenplanetNext").mkdir()
    assert paths.game_user_dir(tmp_path, "Linux") == user
    monkeypatch.setenv(paths.ENV_VAR, str(tmp_path / "elsewhere"))
    assert paths.game_user_dir(tmp_path, "Linux") == tmp_path / "elsewhere"


def test_nothing_found_points_at_the_usual_steam_place(tmp_path, monkeypatch):
    monkeypatch.delenv(paths.ENV_VAR, raising=False)
    found = paths.game_user_dir(tmp_path, "Linux")
    assert found.parts[-5:] == ("2225070", "pfx", "drive_c", "users", "steamuser")
