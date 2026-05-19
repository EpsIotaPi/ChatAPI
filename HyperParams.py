from typing import Optional
from openai import Omit
from pyparsing import Dict


class ModelHyperParams:
    model: str
    base_url: str
    temperature: Optional[float] | Omit
    top_p: Optional[float] | Omit
    random_seed: Optional[int] | Omit

    """
    添加新的超参数时，需要更新：
    1. ModelHyperParams
    2. Conversation.from_hparams, Conversation.send
    """

    def __init__(self, model, temperature = None, top_p= None, random_seed = None):
        omit = Omit()
        self.model = model
        self.temperature = omit if temperature is None else temperature
        self.top_p =  omit if top_p is None else top_p
        self.random_seed = omit if random_seed is None else random_seed

    @classmethod
    def from_record(cls, model, hp_record: Dict):

        return cls(model,
                   temperature=hp_record["temperature"],
                   top_p=hp_record["top_p"],
                   random_seed=hp_record["top_p"])

    def record(self):
        records = {
            "temperature": None if isinstance(self.temperature, Omit) else self.temperature,
            "top_p": None if isinstance(self.top_p, Omit) else self.top_p,
            "random_seed": None if isinstance(self.random_seed, Omit) else self.random_seed,
        }

        return records