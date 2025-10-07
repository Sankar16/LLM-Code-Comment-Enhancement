from cbleu import *  # BLEU score calculation
from rouge import Rouge  # ROUGE score calculation
import os  # For interacting with the operating system
import argparse  # Import argparse for command-line argument parsing
import openai  # For OpenAI's GPT model (Codex)
import json  # For reading JSON files
import time  # For time tracking
import random  # For generating random values
import nltk  # For NLP tasks like BLEU score and n-gram calculation
import re  # For regular expressions (used in tokenization)
import numpy as np  # For numerical operations
from collections import defaultdict  # For defaultdict functionality
from sentence_transformers import SentenceTransformer, util  # For sentence embeddings and semantic similarity calculation
from nltk.translate.bleu_score import sentence_bleu  # BLEU score calculation
from nltk.util import ngrams  # For generating n-grams
from collections import Counter  # For counting items (used in BLEU scoring)

openai.api_key = os.getenv("OPENAI_API_KEY")

example_code = """
def start
  if !is_started?
    temp_exec = nil
    executor = get_external_executor
    if executor.nil?
      executor = temp_exec = create_executor
    else
      temp_exec = nil
    end
    future = executor.submit(create_task(temp_exec))
    return true
  end
  false
end
"""

exampler_what = "You are an expert Ruby programmer, please describe the functionality of the method:\n\"\"\"" + "Example Code1:\n" + example_code + "The comment is: Starts the background initialization"

exampler_why = "# You are an expert Ruby programmer, please explain the reason why the method is provided or the design rationale of the method:\n\"\"\"" + example_code + "The comment is: With this method, the initializer becomes active and invokes the initialize() method in a background task"

exampler_use = "# You are an expert Ruby programmer, please describe the usage or the expected setup of using the method:\n\"\"\"" + example_code + "The comment is: After the construction of a BackgroundInitializer() object, its start() method has to be called"

exampler_done = "# You are an expert Ruby programmer, please describe the implementation details of the method:\n\"\"\"" + example_code + "The comment is: Get an external executor to create a background task. If there is no executor, it creates a new one"

exampler_property = "# You are an expert Ruby programmer, please describe the assert properties of the method including pre-conditions or post-conditions of the method:\n\"\"\"" + example_code + "The comment is: Returns the flag indicating whether the initializer could be started successfully"

zero_what =  "You are an expert Ruby programmer, please describe the functionality of the method:\n\"\"\""

zero_property = "# You are an expert Ruby programmer, please describe the assert properties of the method including pre-conditions or post-conditions of the method:\n\"\"\""

zero_why = "# You are an expert Ruby programmer, please explain the reason why the method is provided or the design rationale of the method:\n\"\"\""

zero_use = "# You are an expert Ruby programmer, please describe the usage or the expected setup of using the method:\n\"\"\""

zero_done = "# You are an expert Ruby programmer, please describe the implementation details of the method:\n\"\"\""

model = SentenceTransformer("flax-sentence-embeddings/st-codesearch-distilroberta-base")

# define a retry decorator

def retry_with_exponential_backoff(
        func,
        initial_delay: float = 1,
        exponential_base: float = 2,
        jitter: bool = True,
        max_retries: int = 10,
        errors: tuple = (Exception,)  # Catch all exceptions (or more specific exceptions if desired)
):
    def wrapper(*args, **kwargs):
        # Initialize variables
        num_retries = 0
        delay = initial_delay

        while True:
            try:
                return func(*args, **kwargs)

            except errors as e:
                num_retries += 1

                if num_retries > max_retries:
                    raise Exception(
                        f"Maximum number of retries ({max_retries}) exceeded."
                    )

                delay *= exponential_base * (1 + jitter * random.random())
                time.sleep(delay)

            except Exception as e:
                raise e

    return wrapper

@retry_with_exponential_backoff
def completion_with_backoff(**kwargs):
    return openai.Completion.create(**kwargs)

def tokenize(code_str):
    # Remove comments in Ruby (single-line and multi-line comments)
    code_str = re.sub(r'#.*', '', code_str)  # single-line comments
    code_str = re.sub(r'=begin.*?=end', '', code_str, flags=re.DOTALL)  # multi-line comments

    # Remove punctuation specific to Ruby, like `=>`, and replace with space
    code_str = re.sub(r'[\.,\;\:\(\)\{\}\[\]\=\>\-\+\*\/\%\&\|\<\>\!]', ' ', code_str)

    # Remove extra spaces and split by whitespace
    code_str = re.sub(r'\s+', ' ', code_str)
    tokens = re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*|[0-9]+|[^\w\s]+', code_str)
    return tokens

def count_common_elements(list1, list2):
    set1 = set(list1)
    set2 = set(list2)
    common_elements = set1.intersection(set2)
    return len(common_elements)

def cal_similarity_token(code1, code2):
    list1, list2 = tokenize(code1), tokenize(code2)
    return count_common_elements(list1, list2)


def read_data(file_address):
    ids = []
    codes = []
    comments = []
    labels = []
    with open(file_address, "r") as file:
        #lines = file.readlines()
        lines = json.load(file)
    for sample in lines:
        #sample = json.loads(line)
        ids.append(sample.get("id"))
        codes.append(sample.get("raw_code"))
        comments.append(sample.get("comment"))
        labels.append(sample.get("label"))
    print('Number of samples is ' + str(len(ids)))
    print('Number of codes is ' + str(len(codes)))
    print('Number of comments is ' + str(len(comments)))
    print('Number of labels is ' + str(len(labels)))
    return ids, codes, comments, labels

def test_single_code(test_code):
    new_prompt = property_prompt + "\nFor the test code:\n" + test_code + " The comment is: "
    for i in range(100):
      response = completion_with_backoff(model="code-davinci-002", prompt=new_prompt, max_tokens=30)
      cur_ans = response["choices"][0]["text"].split('\n')[0]
      print(cur_ans)

def test_sample_retrieve(test_codes, test_labels, training_codes, training_comments, category, pattern):
    if pattern == 'semantic':
        sim_file = 'sim_semantic.txt'
    else:
        sim_file = 'sim_token.txt'
    
    with open(sim_file, 'r') as fpp:
        lines = fpp.readlines()
        for i in range(len(test_codes)):
            test_code = test_codes[i]
            if test_labels[i] != category:
                continue
            sim_ids = lines[i].split(" ")
            prompt_lists = {'what': zero_what, 'why': zero_why, 'use': zero_use, 'done': zero_done, 'property': zero_property}
            new_prompt = prompt_lists.get(category, zero_what)
            for i in range(10):
                new_prompt += ("\n#Example Code{}:\n".format(i) + training_codes[int(sim_ids[i])])
                new_prompt += ("\n# The comment is: " + training_comments[int(sim_ids[i])])
            new_prompt = new_prompt + "\n#For the test code:\n" + test_code + "\n# The comment is: "
            specified_times = 100
            for j in range(specified_times):
                try:
                    # Debugging: Print the request sent
                    print(f"Request sent to OpenAI: {new_prompt}")
                    
                    response = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant."},
                            {"role": "user", "content": new_prompt}
                        ]
                    )
                    
                    # Debugging: Print the response from OpenAI
                    print(f"Response from OpenAI: {response}")

                    if "choices" in response and len(response["choices"]) > 0:
                        cur_ans = response["choices"][0].get("message", {}).get("content", "No valid response")
                    else:
                        cur_ans = "No valid response"

                    # Append result to ans.txt
                    with open('ans.txt', 'a') as fp:
                        fp.write(cur_ans + '\n')

                except Exception as e:
                    print(f"Error during writing to ans.txt: {e}")
                    continue

def test_sample_rerank(test_codes, test_labels, training_codes, training_comments, category, pattern_rerank):
    with open('ans.txt', 'w') as fp, open('sim_semantic.txt', 'r') as fpp:
        lines = fpp.readlines()
        for i in range(len(test_codes)):
            test_code = test_codes[i]
            if test_labels[i] != category:
                continue
            prompt_lists = {'what': zero_what, 'why': zero_why, 'use': zero_use, 'done': zero_done,
                            'property': zero_property}
            new_prompt = prompt_lists.get(category, zero_what)
            sim_ids, random_ids = lines[i].split(" "), []
            val = 4236  # len for the retrieval set
            ### Generate 10 random values
            for k in range(10):
                random_ids.append(random.randint(1, val))
            for i in range(10):
                new_prompt += ("\n#Example Code{}:\n".format(i) + training_codes[int(random_ids[i])])
                new_prompt += ("\n# The comment is: " + training_comments[int(random_ids[i])])
            new_prompt = new_prompt + "\n#For the test code:\n" + test_code + "\n# The comment is: "
            specified_times, maxx, ans = 100, -1, ""
            code_tokens = tokenize(training_comments[int(sim_ids[0])])
            train_code_embedding = model.encode(training_comments[int(sim_ids[0])], convert_to_tensor=True)
            for j in range(specified_times):
                try:
                    response = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant."},
                            {"role": "user", "content": new_prompt}
                        ]
                    )
                    print("Response from OpenAI:", response)  # Debugging print
                    if "choices" in response and len(response["choices"]) > 0:
                        cur_ans = response["choices"][0].get("message", {}).get("content", "No valid response")
                    else:
                        cur_ans = "No valid response"
                    if pattern_rerank == 'token':
                       comment_list = tokenize(cur_ans)
                       intersect = count_common_elements(code_tokens, comment_list)
                       if intersect > maxx:
                           maxx = intersect
                           ans = cur_ans
                    else:
                        test_code_embedding = model.encode(cur_ans, convert_to_tensor=True)
                        hits = util.semantic_search(test_code_embedding, train_code_embedding)[0]
                        top_hit = hits[0]
                        intersect = top_hit['score']
                        if intersect > maxx:
                            maxx = intersect
                            ans = cur_ans
                except:
                    continue
            with open('ans.txt', 'a') as fp:
                fp.write(cur_ans + '\n')

def test_sample_retrieve_rerank(test_codes, test_labels, training_codes, training_comments, category, pattern_retrieve, pattern_rerank):
    if pattern_retrieve == 'semantic':
        file_sim = 'sim_semantic.txt'
    else:
        file_sim = 'sim_token.txt'
    with open('ans.txt', 'w') as fp, open(file_sim, 'r') as fpp:
        lines = fpp.readlines()
        for i in range(len(test_codes)):
            test_code = test_codes[i]
            if test_labels[i] != category:
                continue

            prompt_lists = {'what': zero_what, 'why': zero_why, 'use': zero_use, 'done': zero_done,
                            'property': zero_property}
            new_prompt = prompt_lists.get(category, zero_what)
            sim_ids, random_ids = lines[i].split(" "), []
            for i in range(10):
                new_prompt += ("\n#Example Code{}:\n".format(i) + training_codes[int(sim_ids[i])])
                new_prompt += ("\n# The comment is: " + training_comments[int(sim_ids[i])])
            new_prompt = new_prompt + "\n#For the test code:\n" + test_code + "\n# The comment is: "
            specified_times, maxx, ans = 100, -1, ""
            code_tokens = tokenize(training_comments[int(sim_ids[0])])
            train_code_embedding = model.encode(training_comments[int(sim_ids[0])], convert_to_tensor=True)
            for j in range(specified_times):
                try:
                    response = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant."},
                            {"role": "user", "content": new_prompt}
                        ]
                    )
                    print("Response from OpenAI:", response)  # Debugging print
                    if "choices" in response and len(response["choices"]) > 0:
                        cur_ans = response["choices"][0].get("message", {}).get("content", "No valid response")
                    else:
                        cur_ans = "No valid response"
                    if pattern_rerank == 'token':
                       comment_list = tokenize(cur_ans)
                       intersect = count_common_elements(code_tokens, comment_list)
                       if intersect > maxx:
                           maxx = intersect
                           ans = cur_ans
                    else:
                        test_code_embedding = model.encode(cur_ans, convert_to_tensor=True)
                        hits = util.semantic_search(test_code_embedding, train_code_embedding)[0]
                        top_hit = hits[0]
                        intersect = top_hit['score']
                        if intersect > maxx:
                            maxx = intersect
                            ans = cur_ans
                except:
                    continue
            with open('ans.txt', 'a') as fp:
                fp.write(cur_ans + '\n')

def test_sample_random(test_codes, test_labels, training_codes, training_comments, category):
  with open('ans.txt', 'w') as fp:
    for i in range(len(test_codes)):
      test_code = test_codes[i]
      if test_labels[i] != category:
          continue
      random_ids =  []
      val = 4236 # len for the retrieval set
      ### Generate 10 random values
      for k in range(10):
          random_ids.append(random.randint(1, val))
      prompt_lists = {'what': zero_what, 'why': zero_why, 'use': zero_use, 'done': zero_done, 'property': zero_property}
      new_prompt = prompt_lists.get(category, zero_what)
      for i in range(10):
          new_prompt += ("\n#Example Code{}:\n".format(i) + training_codes[int(random_ids[i])])
          new_prompt += ("\n# The comment is: " + training_comments[int(random_ids[i])])
      new_prompt = new_prompt + "\n#For the test code:\n" + test_code + "\n# The comment is: "
      #print(new_prompt)
      specified_times = 100
      for j in range(specified_times):
        try:
            response = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant."},
                            {"role": "user", "content": new_prompt}
                        ]
                    )
            print("Response from OpenAI:", response)  # Debugging print
            if "choices" in response and len(response["choices"]) > 0:
                cur_ans = response["choices"][0].get("message", {}).get("content", "No valid response")
            else:
                cur_ans = "No valid response"

                    # Writing to file (append mode)
                with open('ans.txt', 'a') as fp:
                    fp.write(cur_ans + '\n')
        except:
           continue

def get_data():
    training_codes, training_comments, training_labels, test_codes, test_comments, test_labels = [], [], [], [], [], []
    train_file_address = 'labelled_data/labelled_vagrant.json'
    cur_ids, cur_codes, cur_comments, cur_labels = read_data(train_file_address)
    training_codes += cur_codes
    training_comments += cur_comments
    training_labels += cur_labels
    test_file_address = 'labelled_data/labelled_ruby_dataset.json'
    cur_ids, cur_codes, cur_comments, cur_labels = read_data(test_file_address)
    test_codes += cur_codes
    test_comments += cur_comments
    test_labels += cur_labels
    print(f"Test Codes Length: {len(test_codes)}")
    print(f"Test Labels Length: {len(test_labels)}")
    print(f"Training Codes Length: {len(training_codes)}")
    print(f"Training Comments Length: {len(training_comments)}")
    return training_codes, training_comments, training_labels, test_codes, test_comments, test_labels

def find_similar_codes(training_code_list, test_code_list, training_label, test_label):
    training_dict = defaultdict(list)
    for i, (code, label) in enumerate(zip(training_code_list, training_label)):
        training_dict[label].append((code, i))

    result = []
    for test_code, test_label in zip(test_code_list, test_label):
        training_codes = training_dict[test_label]
        similarities = [cal_similarity_token(test_code, code) for code, _ in training_codes]
        top_indices = heapq.nlargest(10, range(len(similarities)), similarities.__getitem__)
        top_ids = [training_codes[i][1] for i in top_indices]
        result.append(top_ids)

    return result


def test_meteor(hypothesis, reference):
    import nltk
    from nltk.translate.meteor_score import meteor_score
    meteor = meteor_score(reference, hypothesis)
    return meteor


def generate_samples():
    training_codes, training_comments, training_labels, test_codes, test_comments, test_labels = get_data()
    test_sample(test_codes)

def compare_ans(category):
    training_codes, training_comments, training_labels, test_codes, test_comments, test_labels = get_data()
    ans_comments = []
    for i in range(len(test_comments)):
        if test_labels[i] == category:
            ans_comments.append(test_comments[i])

    print(f"Total ans_comments: {len(ans_comments)}")  # Debug print

    # Check if ans.txt exists before opening it
    if not os.path.exists('ans.txt'):
        print("'ans.txt' not found, creating a new file...")
        with open('ans.txt', 'w', encoding='utf-8') as fp:
            fp.write("Test comment content\n")
            # Optionally, write a placeholder or empty content to the file
            fp.write("")  # You can write something default here if needed.
    
    # Now open the file after ensuring it exists
    with open('ans.txt', 'r', encoding='utf-8') as fp:
        lines = fp.readlines()

    print(f"Lines read from ans.txt: {len(lines)}")  # Debug print
    return ans_comments, lines, test_labels

def code_sim_cal(code1, code2):
    model = SentenceTransformer("flax-sentence-embeddings/st-codesearch-distilroberta-base")
    code1_emb = model.encode(code1, convert_to_tensor=True)
    code2_emb = model.encode(code2, convert_to_tensor=True)
    hits = util.semantic_search(code1_emb, code2_emb)[0]
    top_hit = hits[0]
    sim_score = top_hit['score']
    return sim_score

def test_metric(references, candidates):
    sum_bleu, sum_rouge, sum_meteor = 0, 0, 0
    cnt = 0
    rouge_score = Rouge()

    # Check if references and candidates are non-empty
    if len(references) == 0 or len(candidates) == 0:
        print("Error: references or candidates are empty.")
        return

    for i in range(len(references)):
        candidate = candidates[i]
        reference = references[i]
        cnt += 1
        sum_bleu += (nltk_sentence_bleu(candidate, reference) * 100)
        sum_rouge += (rouge_score.calc_score(candidate, reference) * 100)
        sum_meteor += (test_meteor(candidate, [reference]) * 100)

    # Prevent division by zero
    if cnt == 0:
        print("Error: cnt is zero, no valid comparisons made.")
        return

    # Calculate the averages only if cnt > 0
    print(f"BLEU: {sum_bleu / cnt}")
    print(f"ROUGE: {sum_rouge / cnt}")
    print(f"METEOR: {sum_meteor / cnt}")
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("--mode", choices=['rerank', 'retrieve', 'random'], required=True, help="Mode of operation.")

    parser = argparse.ArgumentParser(description="Process some operations.")
    parser.add_argument("--rerank", choices=['semantic', 'token', 'false'], required=True, help="Mode of rerank.")
    parser.add_argument("--retrieve", choices=['semantic', 'token', 'false'], required=True, help="Mode of retrieve.")
    parser.add_argument("--random", action='store_true', help="If present, random mode is active.")
    args = parser.parse_args()
    training_codes, training_comments, training_labels, test_codes, test_comments, test_labels = get_data()
    categories = ['what', 'why', 'use', 'done', 'property']
    if args.random:
        for category in categories:
            test_sample_random(test_codes, test_labels, training_codes, training_comments, category)
    if args.retrieve != 'false' and args.rerank == 'false':
        for category in categories:
            test_sample_retrieve(test_codes, test_labels, training_codes, training_comments, category, args.retrieve)
    if args.retrieve == 'false' and args.rerank != 'false':
        for category in categories:
            test_sample_rerank(test_codes, test_labels, training_codes, training_comments, category, args.rerank)

    for category in categories:
        references, candidates, test_labels = compare_ans(category)
        test_metric(references, candidates)


