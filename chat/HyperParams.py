from typing import Optional
from pyparsing import Dict


class ModelHyperParams:
    model: str
    base_url: str
    temperature: Optional[float]
    top_p: Optional[float]
    random_seed: Optional[int]

    """
    添加新的超参数时，需要更新：
    1. ModelHyperParams
    2. Conversation.from_hparams, Conversation.send
    """

    def __init__(self, model, temperature = None, top_p= None, random_seed = None):
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.random_seed = random_seed

    @classmethod
    def from_record(cls, model, hp_record: Dict):

        return cls(model,
                   temperature=hp_record["temperature"],
                   top_p=hp_record["top_p"],
                   random_seed=hp_record["top_p"])

    def record(self):
        records = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "random_seed":self.random_seed,
        }

        return records