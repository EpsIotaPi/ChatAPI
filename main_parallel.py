import json
import threading
import pandas as pd
from tqdm.auto import tqdm

from concurrent.futures import ThreadPoolExecutor, as_completed

from chat.Conversation import Conversation
from chat.Prompt import PromptManager
from chat.HyperParams import ModelHyperParams
from chat.Connection import DeepSeekConnection, GoogleConnection, OpenAIConnection

save_lock = threading.Lock()


row_list = pd.DataFrame([])
n = len(row_list)
max_workers = min(16, max(1, n))

save_file_name = "test"
connection = DeepSeekConnection()
hp = ModelHyperParams(model=connection.deepseek_flash, temperature=0, random_seed=114514)
prompt = PromptManager().get_prompt("AQ_Joint", lang="en")
save_file_name += hp.model


def single_conversation(idx_item):
    idx, item = idx_item
    content = item[1]
    argument = content["argument"]

    conversation = Conversation(hp, connection, silence=True)
    conversation.init_session(prompt=prompt, text=argument)

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
