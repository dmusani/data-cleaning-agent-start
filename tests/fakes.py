"""Shared test doubles (no network, no OpenAI)."""

from langchain_core.language_models.fake_chat_models import FakeListChatModel

GOOD_CODE = (
    "```python\n"
    "def data_cleaner(data_raw):\n"
    "    return data_raw.drop_duplicates()\n"
    "```"
)
BAD_CODE = "```python\ndef data_cleaner(data_raw):\n    raise ValueError('bad')\n```"
REPORT = "- removed 1 duplicate"


class ScriptedModel(FakeListChatModel):
    """Fake chat model: replays `responses` in order and records every prompt.

    A response of "RAISE" raises RuntimeError("model down"); "RAISE_AUTH"
    raises an exception that looks like an invalid-API-key error.
    """

    calls: list = []

    def _call(self, messages, stop=None, run_manager=None, **kwargs):
        self.calls.append(str(messages))
        text = super()._call(messages, stop=stop, run_manager=run_manager, **kwargs)
        if text == "RAISE":
            raise RuntimeError("model down")
        if text == "RAISE_AUTH":
            raise Exception("Error code: 401 - Incorrect API key provided")
        return text
