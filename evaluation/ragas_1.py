from datasets import Dataset
from helper_functions import llm
from ragas import evaluate
from ragas.metrics import Faithfulness,ContextRelevance,ContextRecall,ResponseRelevancy
import json
from langchain_huggingface import HuggingFaceEmbeddings
from tqdm import tqdm
import time
import pandas as pd

with open(r'data/ragas_eval_filled.json', 'r') as file:
    eval_data = json.load(file) 

eval_dataset = Dataset.from_list(eval_data)

embedding_model = HuggingFaceEmbeddings(model_name ="models\local_all-MiniLM-L6-v2")
faithfulness_metric = Faithfulness()
context_relevance_metric = ContextRelevance()

batch_size = 4 
all_results = []

print(f"Starting evaluation in batches of {batch_size}...")

for i in tqdm(range(0, len(eval_dataset), batch_size)):
    batch = eval_dataset.select(range(i, min(i + batch_size, len(eval_dataset))))
    
    result = evaluate(
        dataset=batch,
        metrics=[
            faithfulness_metric,
            context_relevance_metric,
        ],
        llm=llm,  
        embeddings=embedding_model,
        raise_exceptions=False
    )
    
    all_results.append(result.to_pandas())

    print(f"Batch {i//batch_size + 1} complete. Waiting for 60 seconds...")
    time.sleep(60)
    
    
df_results = pd.concat(all_results, ignore_index=True)

df_results.to_csv("eval_results.csv")
print("\n--- Ragas Evaluation Results ---")
print(df_results)
print("--------------------------------")