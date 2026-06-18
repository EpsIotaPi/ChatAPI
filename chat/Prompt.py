import json


class PromptManager(object):
    def __init__(self,  file_path="./chat/prompts/prompts.json"):
        self._prompt_library = json.load(open(file_path, "r"))["prompts"]
        self.prompt_keys = list(self._prompt_library.keys())

    def get_prompt(self, name:str="chat", lang="en"):
        if name not in self.prompt_keys:
            raise KeyError("Prompt not found")
        return Prompt(name, self._prompt_library[name], lang)


class Prompt(object):
    json_object: bool

    def __init__(self, name:str, content: dict, lang="en"):
        self.name = name
        self.language = lang
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
            with open(self._prompt[self.language]["path_system"], "r") as f:
                self._system_message = f.read()

        if "user" in self._prompt[self.language].keys():
            self._user_message = self._prompt[self.language]["user"]
        elif "path_user" in self._prompt[self.language].keys():
            with open(self._prompt[self.language]["path_user"], "r") as f:
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
