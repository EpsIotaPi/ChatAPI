import os, re


class Prompt:
    __lang_code_list = ["chi", "eng", "jpn"]

    def __init__(self, file_path):
        if not os.path.isfile(file_path):
            self.file_path = os.path.join("./prompts", file_path + ".txt")
        else:
            self.file_path = file_path
        self.__prompt = {}

        with open(self.file_path, "r") as f:
            language = None
            while True:
                line = f.readline()
                if not line:
                    break
                match = re.search(r"^={10}\s#([^\s=]+)\s={10}$", line)
                if match:
                    if language is not None:
                        if content[-1] != "\n":
                            content += "\n"
                        self.__prompt[language] = content

                    content = ""
                    language = self.__to_lang_code(match.group(1))
                else:
                    content += line
            if content[-1] != "\n":
                content += "\n"
            self.__prompt[language] = content

    def __to_lang_code(self, language: str) -> str:
        lang_code = language.lower()[:3]
        if lang_code not in self.__lang_code_list:
            print("ERROR: UNKNOWN LANGUAGE")

        return lang_code

    def __call__(self, language="eng") -> str:
        lang_code = self.__to_lang_code(language)
        if lang_code not in self.__lang_code_list:
            print("ERROR: UNKNOWN LANGUAGE")
        return self.__prompt[lang_code]
