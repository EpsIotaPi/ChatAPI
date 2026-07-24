import json
import threading
import pandas as pd
from tqdm.auto import tqdm

from concurrent.futures import ThreadPoolExecutor, as_completed

from chatapi.Conversation import Conversation
from chatapi.Prompt import PromptManager
from chatapi.HyperParams import ModelHyperParams
from chatapi.ConnectionHandler import OpenAIConnectionHandler, GoogleConnectionHandler
from chatapi.ConnectionParams import DeepSeekConnectionParams, GoogleConnectionParams, OpenAIConnectionParams

save_lock = threading.Lock()


row_list = pd.DataFrame([])
n = len(row_list)
max_workers = min(16, max(1, n))

connection_params = DeepSeekConnectionParams()
connection_params.deepseek_pro()
hp = ModelHyperParams(model=connection_params.model_alias, temperature=0, random_seed=114514)
prompt = PromptManager().get_prompt("AQ_Joint", lang="en")
connection_handler = OpenAIConnectionHandler(hp, connection_params, silence=True, stream=False)

save_file_name = "test"
save_file_name += hp.model


def single_conversation(idx_item):
    idx, item = idx_item
    content = item[1]
    argument = content["argument"]

    conversation = Conversation(connection_handler)
    conversation.init_session(prompt=prompt)

    annotation = json.loads(conversation.last_reply)
    with save_lock:
        conversation.save_to(f"./result/{save_file_name}.jsonl")

    result = {
        "id": idx + 1,
        "argument": argument,
        "annotator": hp.model,
        "combined_quality": 0,
        "logical_quality": annotation["Logical"],
        "dialectical_quality": annotation["Dialectical"],
        "rhetorical_quality": annotation["Rhetorical"]
    }

    return idx, result


if "__main__" == __name__:
    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(single_conversation, (idx, item)) for idx, item in enumerate(row_list.iterrows())
        ]
        for fut in tqdm(as_completed(futures), total=n, desc="Requesting Model", unit="Times"):
            idx, row = fut.result()
            results[idx] = row

    rows = [results[i] for i in range(n)]

    df = pd.json_normalize(rows)
    df.to_csv(f"./result/{save_file_name}.csv")
