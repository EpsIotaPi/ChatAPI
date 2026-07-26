from dataclasses import dataclass, asdict, fields

@dataclass
class ModelHyperParams:
    # 添加新的超参数时，需要更新：OpenAIConnectionHandler.send

    model_alias: str

    temperature: float | None = None
    top_p: float| None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None

    random_seed: int | None = None
    max_completion_tokens: int | None = None
    reasoning_effort: str | None = None  # "none", "minimal", "low", "medium", "high", "xhigh", "max"

    @classmethod
    def from_record(cls, model_alias: str, hp_record: dict):
        # hp_record 的 key 需与本类字段名一致；用 .get() 兜底，避免记录里缺字段时报 KeyError
        hp_field_names = {f.name for f in fields(cls) if f.name != "model_alias"}
        kwargs = {name: hp_record.get(name) for name in hp_field_names}
        return cls(model_alias=model_alias, **kwargs)

    def record(self) -> dict:
        return asdict(self)
