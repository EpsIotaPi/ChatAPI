import json
import os

_DEFAULT_PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "prompts")
_DEFAULT_PROMPTS_PATH = os.path.join(_DEFAULT_PROMPTS_DIR, "prompts.json")


class PromptManager(object):
    def __init__(self, file_path=None):
        if file_path is None:
            prompt_dir = os.getenv("PROMPTS_DIR") or _DEFAULT_PROMPTS_DIR
            file_path = os.path.join(prompt_dir, "prompts.json")
        self._base_dir = os.path.dirname(os.path.abspath(file_path))
        self._prompt_library = json.load(open(file_path, "r"))["prompts"]
        self.prompt_keys = list(self._prompt_library.keys())

    def get_prompt(self, name:str="chatapi", lang="en"):
        if name not in self.prompt_keys:
            raise KeyError("Prompt not found")
        return Prompt(name, self._prompt_library[name], lang, base_dir=self._base_dir)


class Prompt(object):
    json_object: bool

    def __init__(self, name:str, content: dict, lang="en", base_dir=None):
        self.name = name
        self.language = lang
        self._base_dir = base_dir or os.path.dirname(_DEFAULT_PROMPTS_PATH)
        self.description = content["description"]
        self.variables = content["variables"]
        self.language_list = list(content["languages"].keys())
        self._prompt = content["languages"]
        self._system_message = None
        self._user_message = None
        self.set_language(lang)

        self.json_object = content["json_object"] if "json_object" in content.keys() else False



    def set_language(self, lang):
        if lang not in self.language_list:
            raise KeyError("Language not found")
        self.language = lang
        self._system_message = None
        self._user_message = None

        if "system" in self._prompt[self.language].keys():
            self._system_message = self._prompt[self.language]["system"]
        elif "path_system" in self._prompt[self.language].keys():
            path = os.path.join(self._base_dir, self._prompt[self.language]["path_system"])
            with open(path, "r") as f:
                self._system_message = f.read()

        if "user" in self._prompt[self.language].keys():
            self._user_message = self._prompt[self.language]["user"]
        elif "path_user" in self._prompt[self.language].keys():
            path = os.path.join(self._base_dir, self._prompt[self.language]["path_user"])
            with open(path, "r") as f:
                self._user_message = f.read()


    def system_message(self):
        return self._system_message

    def user_message(self, **kwargs):
        if self._user_message is None:
            return None
        else:
            message = self._user_message
            for var in self.variables:
                if var not in kwargs:
                    raise KeyError(f"Variable {var} not found")
                placeholder = "{{" + var + "}}"
                message = message.replace(placeholder, kwargs[var])

            return message
