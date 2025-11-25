import math
import time
import torch
from torch.utils.data import DataLoader, TensorDataset
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import math
from tqdm.auto import tqdm



MODEL_NAME = "gpt2"
DATASET_NAME = "wikitext"
DATASET_CONFIG = "wikitext-2-raw-v1"
DEVICE = "mps"
SEQ_LEN = 512
BATCH_SIZE = 4

def load_wikitext(tokenizer, ratio=0.5):
    dataset = load_dataset(DATASET_NAME, DATASET_CONFIG)
    text_texts = dataset['test']['text']
    full_text = "\n\n".join(text_texts[:int(len(text_texts)*ratio)])
    
    enc = tokenizer(full_text, return_tensors='pt')
    return enc["input_ids"][0]

def make_chunks(ids, seq_len):
    n_tokens = ids.size(0)
    n_chunks = n_tokens // seq_len
    trunc_size = n_chunks * seq_len
    ids = ids[:trunc_size]
    inputs = ids.view(n_chunks, seq_len)
    labels = inputs.clone()
    return inputs, labels
    
def compute_ppl(model, inputs, labels, device, batch_size):
    ds = TensorDataset(inputs, labels)
    dl = DataLoader(ds, batch_size=batch_size)
    
    total_nll = 0.0
    total_tokens = 0
    t0 = time.perf_counter()
    
    model.eval()
    with torch.no_grad():
        for batch_inputs, batch_labels in tqdm(dl, total=len(dl), desc="batches", leave=False):
            batch_inputs = batch_inputs.to(device)
            batch_labels = batch_labels.to(device)
            
            out = model(input_ids=batch_inputs, labels=batch_labels)
            loss = out.loss
            n_tokens_batch = batch_labels.numel()
            
            total_nll += loss.item() * n_tokens_batch
            total_tokens += n_tokens_batch
    
    t1 = time.perf_counter()
    elapsed = t1 - t0
    ppl = math.exp(total_nll / total_tokens)
    return ppl, elapsed

def main():
    print(f"Loading wikitext")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)

    print("Tokenizing dataset")
    ids = load_wikitext(tokenizer)
    print(f"Total tokens: {ids.size(0)}")

    print("Creating chunks")
    inputs, labels = make_chunks(ids, SEQ_LEN)

    print("Computing perplexity")
    ppl, elapsed = compute_ppl(model, inputs, labels, DEVICE, BATCH_SIZE)

    print(f"Perplexity: {ppl:.2f}")
    print(f"Elapsed time: {elapsed:.2f} seconds")

if __name__ == "__main__":
    main()