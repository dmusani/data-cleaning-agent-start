from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent.parent / "app.py")
PASSWORD = 1  # TextInput.PASSWORD in the Streamlit proto


@pytest.fixture(autouse=True)
def no_dotenv(monkeypatch):
    # app.py calls load_dotenv(); keep a real .env from leaking a key into tests.
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: False)


def password_inputs(at):
    return [t for t in at.text_input if t.proto.type == PASSWORD]


def test_missing_key_prompts_for_it(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    at = AppTest.from_file(APP, default_timeout=30).run()

    assert not at.exception
    assert len(at.warning) >= 1
    assert len(password_inputs(at)) == 1


def test_key_in_env_hides_prompt(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    at = AppTest.from_file(APP, default_timeout=30).run()

    assert not at.exception
    assert password_inputs(at) == []
    assert len(at.warning) == 0


def test_no_file_shows_prompt_to_upload(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    at = AppTest.from_file(APP, default_timeout=30).run()

    assert any("Upload a CSV" in i.value for i in at.info)
    assert len(at.tabs) == 0


# ---- Run-path tests: fake uploader and fake model, no network ----

import io

import streamlit as st
from fakes import BAD_CODE, GOOD_CODE, REPORT, ScriptedModel

CSV = b"a,b\n1,x\n1,x\n2,y\n"


class FakeUpload(io.BytesIO):
    name = "sample.csv"

    @property
    def size(self):
        return len(self.getvalue())


@pytest.fixture
def run_env(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(st, "file_uploader", lambda *a, **k: FakeUpload(CSV))
    # Never write generated code into the real logs/ folder.
    monkeypatch.setattr("data_cleaning_agent.data_cleaning_agent.LOG_PATH", str(tmp_path) + "/")

    def use_model(responses):
        monkeypatch.setattr(
            "langchain_openai.ChatOpenAI", lambda **kw: ScriptedModel(responses=responses)
        )

    return use_model


def click_clean(at):
    next(b for b in at.button if b.label == "Clean Data").click()
    return at.run()


def test_successful_run_shows_tabs_and_survives_rerun(run_env):
    run_env([GOOD_CODE, REPORT])

    at = click_clean(AppTest.from_file(APP, default_timeout=30).run())

    assert not at.exception
    assert [t.label for t in at.tabs] == ["Overview", "Quality", "Code", "Data"]
    assert len(at.metric) == 4
    assert any(REPORT in m.value for m in at.markdown)

    at.run()  # e.g. clicking a download button triggers a rerun
    assert [t.label for t in at.tabs] == ["Overview", "Quality", "Code", "Data"]


def test_failed_cleaning_shows_error_not_success(run_env):
    run_env([BAD_CODE, BAD_CODE, BAD_CODE])

    at = click_clean(AppTest.from_file(APP, default_timeout=30).run())

    assert not at.exception
    assert any("Cleaning failed" in e.value for e in at.error)
    assert len(at.tabs) == 0


def test_invalid_key_shows_friendly_message(run_env):
    run_env(["RAISE_AUTH"])

    at = click_clean(AppTest.from_file(APP, default_timeout=30).run())

    assert not at.exception
    # The key came from the environment, so the hint must point there, not at a sidebar field.
    assert any("OPENAI_API_KEY" in e.value for e in at.error)


def test_invalid_typed_key_says_to_reenter_it(run_env, monkeypatch):
    run_env(["RAISE_AUTH"])
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    at = AppTest.from_file(APP, default_timeout=30).run()
    password_inputs(at)[0].set_value("sk-bad")
    at = click_clean(at.run())

    assert not at.exception
    assert any("re-enter" in e.value.lower() for e in at.error)


def test_empty_csv_shows_error_and_no_button(run_env, monkeypatch):
    monkeypatch.setattr(st, "file_uploader", lambda *a, **k: FakeUpload(b""))

    at = AppTest.from_file(APP, default_timeout=30).run()

    assert not at.exception
    assert any("empty" in e.value.lower() for e in at.error)
    assert not any(b.label == "Clean Data" for b in at.button)


def test_zero_change_tiles_show_no_delta_arrow(run_env):
    unchanged = "```python\ndef data_cleaner(data_raw):\n    return data_raw\n```"
    run_env([unchanged, REPORT])

    at = click_clean(AppTest.from_file(APP, default_timeout=30).run())

    assert not at.exception
    assert len(at.metric) == 4
    assert all(not m.delta for m in at.metric)
