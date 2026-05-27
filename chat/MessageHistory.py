import json, uuid, random
from typing import Union, Optional
from pathlib import Path
from datetime import datetime, UTC

from chat.HyperParams import ModelHyperParams



def utc_timestamp() -> str:
    return (
        datetime.now(UTC)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )

SEED = random.randint(0, 2**32 - 1)


class MessageHistory:
    model_hp: ModelHyperParams
    _session_content: dict = None

    def __init__(self, hp: ModelHyperParams):
        self.model_hp = hp

    def init_session(self, session_title="New Session", language="en"):
        if self._session_content is not None:
            raise Exception("MessageHistory already init")
        timestamp = utc_timestamp()
        self._session_content = {
            "version": "1.0",
            "session": {
                "id": "new_sess_{}".format(uuid.uuid4()),
                "title": session_title,
                "created_at": timestamp,
                "updated_at": timestamp
            },

            "metadata": {
                "prompt_name": None,
                "model": self.model_hp.model,
                "language": language,
                "hp": self.model_hp.record()
            },

            "messages": []
        }

    @classmethod
    def load(cls, data: Union[str, Path, dict]):
        if not isinstance(data, dict):
            with open(data, "r") as f:
                data = json.load(f)

        data["session"]["updated_at"] = utc_timestamp()

        metadata = data["metadata"]

        lang = metadata["language"]

        model = metadata["model"]

        hp_record = metadata["hp"]

        model_hp = ModelHyperParams.from_record(model, hp_record)

        obj = cls(hp=model_hp)

        obj._session_content = data

        return obj

    def _save_message(self, role, message):
        data = {
            "id": "msg_{:03d}".format(len(self._session_content["messages"])),
            "role": role,
            "content": message,
            "created_at": utc_timestamp(),
        }
        self._session_content["messages"].append(data)

    def set_system_prompt(self, sys_prompt: str, prompt_name=None):
        if prompt_name is not None:
            self._session_content["metadata"]["prompt_name"] = prompt_name
        self._session_content["messages"].append({
            "id": "sys_000",
            "role": "system",
            "content": sys_prompt,
            "created_at": utc_timestamp()
        })

    def user_message(self, message: str):
        self._save_message(role="user", message=message)

    def assistant_message(self, message: str):
        self._save_message(role="assistant", message=message)

    @property
    def session_content(self):
        return self._session_content

    @property
    def messages(self) -> list:
        return self._session_content["messages"]

    @property
    def sys_prompt(self) -> Optional[str]:
        if self.messages[0]["role"] != "system":
            return None
        return self.messages[0]["content"]

    @property
    def last_message(self) -> str:
        return self.messages[-1]["content"]

    def __len__(self):
        return len(self.messages)
