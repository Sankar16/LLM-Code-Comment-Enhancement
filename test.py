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
from string import Template
from huggingface_hub import login
import torch

# ==============================
#  MODEL INITIALIZATION
# ==============================
'''
print("🔹 Initializing local model (microsoft/phi-2, CPU-safe)...")
model_name = "microsoft/phi-2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",  # use CPU-safe precision
    device_map=None
)

model_name = "meta-llama/Meta-Llama-3-8B-Instruct"  # ✅ GPU Llama model

print(f"🔹 Loading {model_name} ({'GPU' if torch.cuda.is_available() else 'CPU'})...")

tokenizer = AutoTokenizer.from_pretrained(model_name, token=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",  # ✅ auto places on GPU
    token=True
)
'''
model_name = "microsoft/Phi-3-mini-4k-instruct"

print(f"🔹 Loading {model_name} (GPU)...")

tokenizer = AutoTokenizer.from_pretrained(model_name, token=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    token=True
)

print("✅ Model loaded successfully!")

generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer
)
print("✅ Model loaded successfully!")
semantic_model = SentenceTransformer("all-MiniLM-L6-v2")

# ==============================
#  PROMPT DEFINITIONS
# ==============================
zero_what = (
    "You are an expert Ruby developer. Given a Ruby method, write a concise 1–3 line Ruby-style comment "
    "that accurately describes what the method does — its functionality or purpose. "
    "Focus on what it accomplishes or returns, not how it works internally. "
    "Use clear, natural Ruby documentation style, starting each line with '#'. "
    "Do not include markdown, code, or explanations.\n\n"
    "Example 1:\n"
    "Code:\n'''def update_profile(user, params)\n  user.update(params)\n  user.save!\nend'''\n"
    "Comment:\n# Updates the given user's profile details with the provided parameters\n# Saves the changes to the database\n\n"
    "Example 2:\n"
    "Code:\n'''def eligible_for_discount?(order)\n  order.total > 100 && order.customer.loyal?\nend'''\n"
    "Comment:\n# Returns true if the order exceeds $100 and the customer has loyalty status\n\n"
    "Now write the comment for this Ruby method:\n'''{code}'''\nComment:"
)

# ------------------------------

zero_why = (
    "You are an expert Ruby developer. Write a concise 1–3 line Ruby-style comment explaining *why* this method exists — "
    "the rationale or intent behind it. Describe what broader purpose it serves in the system, not how it works. "
    "Start each line with '#'. Avoid markdown, speculative reasoning, or unnecessary explanation.\n\n"
    "Example 1:\n"
    "Code:\n'''def archive_inactive_users\n  User.where(active: false).update_all(archived: true)\nend'''\n"
    "Comment:\n# Used to archive users who have been inactive for a long time\n# Keeps the active user dataset clean and efficient\n\n"
    "Example 2:\n"
    "Code:\n'''def refresh_cache\n  Rails.cache.clear\n  load_initial_data\nend'''\n"
    "Comment:\n# Refreshes system cache after configuration updates to maintain consistency\n\n"
    "Now write the comment for this Ruby method:\n'''{code}'''\nComment:"
)

# ------------------------------

zero_use = (
    "You are an expert Ruby developer. Write a concise 1–3 line Ruby-style comment describing when and how this method should be used. "
    "Focus on its intended context, inputs, and purpose. Start each line with '#'. "
    "Do not explain implementation details or include markdown.\n\n"
    "Example 1:\n"
    "Code:\n'''def send_welcome_email(user)\n  Mailer.welcome(user).deliver_later\nend'''\n"
    "Comment:\n# Called immediately after user registration to send a welcome email\n# Should be used only when a valid user record exists\n\n"
    "Example 2:\n"
    "Code:\n'''def generate_invoice(order)\n  Invoice.create_from_order(order)\nend'''\n"
    "Comment:\n# Use this method to generate an invoice after a successful checkout\n\n"
    "Now write the comment for this Ruby method:\n'''{code}'''\nComment:"
)

# ------------------------------

zero_done = (
    "You are an expert Ruby developer. Write a concise 1–3 line Ruby-style comment describing how the method works internally. "
    "Focus on the logic, flow, and operations performed — not its external purpose. "
    "Start each line with '#'. Avoid markdown, bullet points, or excessive detail.\n\n"
    "Example 1:\n"
    "Code:\n'''def calculate_average(scores)\n  scores.sum.to_f / scores.size\nend'''\n"
    "Comment:\n# Computes the average score by dividing the total sum by the count of elements\n\n"
    "Example 2:\n"
    "Code:\n'''def find_admin_users(users)\n  users.select { |u| u.role == 'admin' }\nend'''\n"
    "Comment:\n# Filters the list of users and returns only those with admin privileges\n\n"
    "Now write the comment for this Ruby method:\n'''{code}'''\nComment:"
)

# ------------------------------

zero_property = (
    "You are an expert Ruby developer. Write a concise 1–3 line Ruby-style comment describing the properties, "
    "preconditions, postconditions, or return value of the method. "
    "Focus on what must be true before calling it, or what it guarantees to return. "
    "Use Ruby comment style starting with '#', and avoid markdown or examples.\n\n"
    "Example 1:\n"
    "Code:\n'''def logged_in?\n  !@user.nil?\nend'''\n"
    "Comment:\n# Returns true if a user session is active, false otherwise\n\n"
    "Example 2:\n"
    "Code:\n'''def balance\n  @transactions.sum(&:amount)\nend'''\n"
    "Comment:\n# Calculates and returns the current account balance from all transactions\n# Assumes all transactions respond to :amount\n\n"
    "Now write the comment for this Ruby method:\n'''{code}'''\nComment:"
)

# ------------------------------

zero_others = (
    "You are an expert Ruby developer. Write a concise 1–3 line Ruby-style comment summarizing the general purpose of the method. "
    "Focus on what it achieves or manages. The comment must start with '#', be written in plain Ruby documentation style, "
    "and must not include markdown, examples, or code snippets.\n\n"
    "Example:\n"
    "Code:\n'''def clean_temp_files\n  FileUtils.rm_rf(Dir['tmp/*'])\nend'''\n"
    "Comment:\n# Deletes all temporary files from the tmp directory to free up space\n# Typically called as part of scheduled cleanup tasks\n\n"
    "Now write the comment for this Ruby method:\n'''{code}'''\nComment:"
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
    # response = generator(prompt, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.7)
    response = generator(
    prompt,
    max_new_tokens=max_new_tokens,
    temperature=0.3,        
    top_p=0.9,              
    do_sample=True,
    repetition_penalty=1.1, 
    eos_token_id=tokenizer.eos_token_id
)
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
    train_file = "labelled_train.json"
    test_file = "labelled_test.json"
    _, tr_codes, tr_comments, tr_labels = read_data(train_file)
    _, te_codes, te_comments, te_labels = read_data(test_file)
    return tr_codes, tr_comments, tr_labels, te_codes, te_comments, te_labels


# ==============================
#  MAIN TEST LOOP
# ==============================
'''
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
'''
def sanitize_output(text: str, max_lines: int = 3) -> str:
    """
    Cleans the model output to keep only valid Ruby-style comments.
    Ensures it starts with '#' and removes extra explanations or code.
    Limits to 1–3 concise comment lines.
    """
    if not text:
        return ""

    # Split and clean lines
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    # Keep only comment lines that start with '#'
    comment_lines = [line for line in lines if line.startswith("#")]

    # If none start with '#', try to fix common hallucinations (like “This method…”)
    if not comment_lines and lines:
        # Prefix with '#' if it looks like a plain sentence
        if re.match(r'^[A-Za-z]', lines[0]):
            comment_lines = [f"# {lines[0]}"]
        else:
            comment_lines = lines[:1]

    # Limit to 3 lines max, and trim long comments
    cleaned = "\n".join(comment_lines[:max_lines]).strip()
    return cleaned

def test_sample_retrieve(test_codes, test_labels, training_codes, training_comments, category, test_ids):
    ans_path = f"ans_{category}.json"
    results = []
    used_samples = 0

    for i, code in enumerate(test_codes):
        if test_labels[i] != category:
            continue
        used_samples += 1

        # Use string.Template to safely substitute variable without interfering with Ruby braces
        safe_code = code
        prompt_template = PROMPT_MAP.get(category, zero_others)

        # Convert prompt into a Template object (replace {code} → $code)
        prompt_template = prompt_template.replace("{code}", "$code")
        new_prompt = Template(prompt_template).safe_substitute(code=safe_code)

        try:
            cur_ans = generate_with_model(new_prompt)
            cleaned_ans = sanitize_output(cur_ans)  

            result = {
                "id": test_ids[i],
                "raw_code": code,
                "generated_comment": cleaned_ans,
                "old_comment": te_comments[i],
                "label": category
            }
            results.append(result)
        except Exception as e:
            print(f"⚠️ Error on sample {i}: {e}")
            continue

    with open(ans_path, "w", encoding="utf-8") as fp:
        json.dump(results, fp, indent=2, ensure_ascii=False)

    print(f"[{category}] ✅ Generated {len(results)} samples. Saved to {ans_path}.")

def compare_ans(category):
    ans_path = f"ans_{category}.json"
    if not os.path.exists(ans_path):
        print(f"No predictions file for {category}: {ans_path}")
        return [], []

    with open(ans_path, "r", encoding="utf-8") as fp:
        data = json.load(fp)

    references = [item["gold_comment"] for item in data]
    candidates = [item["generated_comment"] for item in data]

    print(f"[{category}] Gold refs: {len(references)}, Predictions: {len(candidates)}")
    return references, candidates



def test_metric(references, candidates):
    if not references or not candidates:
        print("Error: Empty references/candidates.")
        return
    sum_bleu, sum_rouge, sum_meteor, cnt = 0, 0, 0, 0
    rouge_score = Rouge()
    for ref, cand in zip(references, candidates):
        cnt += 1
        # --- BLEU ---
        try:
            sum_bleu += sentence_bleu([ref.split()], cand.split(), smoothing_function=smooth) * 100
        except:
            pass

        # --- ROUGE ---
        try:
            sum_rouge += rouge_calc.calc_score([cand], [ref]) * 100
        except:
            pass

        # --- METEOR ---
        try:
            sum_meteor += meteor_score([ref], cand) * 100
        except:
            pass
    if cnt == 0:
        print("Error: No valid samples.")
        return
    print(f"BLEU: {sum_bleu/cnt:.2f}, ROUGE-L(F1): {sum_rouge/cnt:.2f}, METEOR: {sum_meteor/cnt:.2f}")

def semantic_similarity(references, candidates):
    """
    Computes average semantic similarity between generated and gold comments
    using SentenceTransformer embeddings (cosine similarity).
    """
    if not references or not candidates:
        print("Error: Empty references/candidates.")
        return

    scores = []
    for ref, cand in zip(references, candidates):
        if not ref.strip() or not cand.strip():
            continue
        ref_emb = semantic_model.encode(ref, convert_to_tensor=True)
        cand_emb = semantic_model.encode(cand, convert_to_tensor=True)
        sim = util.pytorch_cos_sim(ref_emb, cand_emb).item()
        scores.append(sim)

    if scores:
        print(f"Semantic Similarity (avg cosine): {np.mean(scores):.3f}")
    else:
        print("No valid samples for semantic similarity.")

# ==============================
#  MAIN
# ==============================
'''
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
'''
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieve", choices=["semantic", "token", "false"], required=True)
    parser.add_argument("--rerank", choices=["semantic", "token", "false"], required=True)
    args = parser.parse_args()

    # Load data
    tr_ids, tr_codes, tr_comments, tr_labels = read_data("labelled_vagrant.json")
    te_ids, te_codes, te_comments, te_labels = read_data("labelled_test.json")
    categories = ["what", "why", "how-to-use", "how-it-is-done", "property", "others"]

    # Generate predictions
    if args.retrieve != "false" and args.rerank == "false":
        for cat in categories:
            print(f"\n=== Generating for category: {cat} ===")
            test_sample_retrieve(te_codes, te_labels, tr_codes, tr_comments, cat, te_ids)

    # Evaluate metrics
    for cat in categories:
        refs, cands = compare_ans(cat)
        test_metric(refs, cands)
        semantic_similarity(refs, cands)
