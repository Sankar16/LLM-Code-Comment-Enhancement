from rouge import Rouge
import os
import json
import time
import random
import re
import heapq
import argparse
import numpy as np
from collections import defaultdict
from sentence_transformers import SentenceTransformer, util
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# ==============================
#  MODEL INITIALIZATION
# ==============================
print("🔹 Initializing local model (microsoft/phi-2, CPU-safe)...")
model_name = "microsoft/phi-2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",  # use CPU-safe precision
    device_map=None
)

generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=-1  # run on CPU (or change to device_map={'': 'mps'} if you have Apple GPU)
)
print("✅ Model loaded successfully!")

# ==============================
#  PROMPT DEFINITIONS
# ==============================
zero_what = (
    "You are an expert Ruby programmer. "
    "Given the following Ruby method, describe its overall functionality clearly and concisely. "
    "Focus on what the method does (its purpose and behavior), not how it works internally.\n\n"
    "Ruby Method:\n'''{code}'''\n\nComment:"
)

zero_why = (
    "You are an expert Ruby programmer. "
    "Explain the reason or design rationale behind the following method. "
    "Why might a developer have written this method? Mention its intent or purpose in the broader program.\n\n"
    "Ruby Method:\n'''{code}'''\n\nComment:"
)

zero_use = (
    "You are an expert Ruby programmer. "
    "Describe how to use the following method — when and where it should be called, "
    "what inputs it expects, and what happens when it is used.\n\n"
    "Ruby Method:\n'''{code}'''\n\nComment:"
)

zero_done = (
    "You are an expert Ruby programmer. "
    "Describe the internal implementation details of the following method. "
    "Focus on how the method achieves its purpose — the logic, key steps, or flow of execution.\n\n"
    "Ruby Method:\n'''{code}'''\n\nComment:"
)

zero_property = (
    "You are an expert Ruby programmer. "
    "Describe the properties, preconditions, or postconditions of the following method. "
    "Mention any assumptions, validations, or guarantees related to inputs and outputs.\n\n"
    "Ruby Method:\n'''{code}'''\n\nComment:"
)

zero_others = (
    "You are an expert Ruby programmer. "
    "Given the following Ruby method, write a short and relevant comment summarizing its purpose.\n\n"
    "Ruby Method:\n'''{code}'''\n\nComment:"
)

PROMPT_MAP = {
    "what": zero_what,
    "why": zero_why,
    "how-to-use": zero_use,
    "how-it-is-done": zero_done,
    "property": zero_property,
    "others": zero_others
}

# ==============================
#  HELPER FUNCTIONS
# ==============================
def retry_with_exponential_backoff(func, initial_delay=1, exponential_base=2, jitter=True, max_retries=5):
    def wrapper(*args, **kwargs):
        delay = initial_delay
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                print(f"[LLM call] Attempt {attempt + 1} failed: {e}")
                time.sleep(delay)
                delay *= exponential_base * (1 + (random.random() if jitter else 0))
        raise Exception("Max retries exceeded")
    return wrapper


@retry_with_exponential_backoff
def generate_with_model(prompt, max_new_tokens=200):
    response = generator(prompt, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.7)
    return response[0]["generated_text"].split("Comment:")[-1].strip()


def tokenize(code_str):
    code_str = re.sub(r'#.*', '', code_str)  # remove comments
    code_str = re.sub(r'=begin.*?=end', '', code_str, flags=re.DOTALL)
    code_str = re.sub(r'[\.,;:\(\)\{\}\[\]\=\>\-\+\*\/\%\&\|\<\>\!]', ' ', code_str)
    code_str = re.sub(r'\s+', ' ', code_str)
    return re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*|[0-9]+|[^\w\s]+', code_str)


def count_common_elements(list1, list2):
    return len(set(list1).intersection(set(list2)))


def cal_similarity_token(code1, code2):
    return count_common_elements(tokenize(code1), tokenize(code2))


def read_data(file_address):
    ids, codes, comments, labels = [], [], [], []
    with open(file_address, "r") as file:
        lines = json.load(file)
    for sample in lines:
        ids.append(sample.get("id"))
        codes.append(sample.get("raw_code"))
        comments.append(sample.get("comment"))
        labels.append(sample.get("label").lower().replace("_", "-"))
    print(f"Loaded {len(ids)} samples from {file_address}")
    return ids, codes, comments, labels


def get_data():
    train_file = "labelled_data/labelled_vagrant.json"
    test_file = "labelled_data/labelled_ruby_dataset.json"
    _, tr_codes, tr_comments, tr_labels = read_data(train_file)
    _, te_codes, te_comments, te_labels = read_data(test_file)
    return tr_codes, tr_comments, tr_labels, te_codes, te_comments, te_labels


# ==============================
#  MAIN TEST LOOP
# ==============================
def test_sample_retrieve(test_codes, test_labels, training_codes, training_comments, category):
    ans_path = f"ans_{category}.txt"
    open(ans_path, "w", encoding="utf-8").close()  # reset file

    used_samples = 0
    for i, code in enumerate(test_codes):
        if test_labels[i] != category:
            continue
        used_samples += 1

        prompt_template = PROMPT_MAP.get(category, zero_others)
        new_prompt = prompt_template.format(code=code)

        try:
            cur_ans = generate_with_model(new_prompt)
            with open(ans_path, "a", encoding="utf-8") as fp:
                fp.write(cur_ans + "\n")
        except Exception as e:
            print(f"Error on sample {i}: {e}")
            continue

    print(f"[{category}] Used {used_samples} samples. Predictions written to {ans_path}.")


def compare_ans(category, te_comments, te_labels):
    ans_path = f"ans_{category}.txt"
    refs = [te_comments[i] for i in range(len(te_comments)) if te_labels[i] == category]

    if not os.path.exists(ans_path):
        print(f"No predictions file for {category}: {ans_path}")
        return refs, []

    with open(ans_path, "r", encoding="utf-8") as fp:
        cands = [line.strip() for line in fp.readlines() if line.strip()]

    print(f"[{category}] Gold refs: {len(refs)}, Predictions: {len(cands)}")
    return refs, cands


def test_metric(references, candidates):
    if not references or not candidates:
        print("Error: Empty references/candidates.")
        return
    sum_bleu, sum_rouge, sum_meteor, cnt = 0, 0, 0, 0
    rouge_score = Rouge()
    for ref, cand in zip(references, candidates):
        cnt += 1
        try:
            sum_bleu += sentence_bleu([ref], cand) * 100
        except:
            pass
        try:
            sum_rouge += rouge_score.get_scores(cand, ref)[0]["rouge-l"]["f"] * 100
        except:
            pass
        try:
            sum_meteor += meteor_score([ref], cand) * 100
        except:
            pass
    if cnt == 0:
        print("Error: No valid samples.")
        return
    print(f"BLEU: {sum_bleu/cnt:.2f}, ROUGE-L(F1): {sum_rouge/cnt:.2f}, METEOR: {sum_meteor/cnt:.2f}")


# ==============================
#  MAIN
# ==============================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieve", choices=["semantic", "token", "false"], required=True)
    parser.add_argument("--rerank", choices=["semantic", "token", "false"], required=True)
    args = parser.parse_args()

    tr_codes, tr_comments, tr_labels, te_codes, te_comments, te_labels = get_data()
    categories = ["what", "why", "how-to-use", "how-it-is-done", "property", "others"]

    if args.retrieve != "false" and args.rerank == "false":
        for cat in categories:
            print(f"\n=== Generating for category: {cat} ===")
            test_sample_retrieve(te_codes, te_labels, tr_codes, tr_comments, cat)

    for cat in categories:
        refs, cands = compare_ans(cat, te_comments, te_labels)
        test_metric(refs, cands)