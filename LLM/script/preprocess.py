from sentence_transformers import SentenceTransformer, util
import time
import json
import re

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

def get_data():
    training_codes, training_comments, training_labels, test_codes, test_comments, test_labels = [], [], [], [], [], []
    train_file_address = 'data/labelled_train.json'
    cur_ids, cur_codes, cur_comments, cur_labels = read_data(train_file_address)
    training_codes += cur_codes
    training_comments += cur_comments
    training_labels += cur_labels
    test_file_address = 'data/labelled_test.json'
    cur_ids, cur_codes, cur_comments, cur_labels = read_data(test_file_address)
    test_codes += cur_codes
    test_comments += cur_comments
    test_labels += cur_labels
    return training_codes, training_comments, training_labels, test_codes, test_comments, test_labels

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

def pre_process_samples_token(test_codes, training_codes):
    test_codes_embeddings, training_codes_embeddings = [], []
    st = time.time()
    for i in range(len(test_codes)):
        test_code = test_codes[i]
        code1_emb = tokenize(test_code)
        test_codes_embeddings.append(code1_emb)
    ed = time.time()
    print('Test code embedding generate finish!')
    print(str(ed - st))
    for i in range(len(training_codes)):
        train_code = training_codes[i]
        code1_emb = tokenize(train_code)
        training_codes_embeddings.append(code1_emb)
    print('Training code embedding generate finish!')
    with open('sim_token.txt', 'w') as fp:
     for i in range(len(test_codes)):
        test_code_embedding = test_codes_embeddings[i]
        sim_scores = []
        for j in range(len(training_codes)):
            train_code_embedding = training_codes_embeddings[j]
            score = count_common_elements(test_code_embedding, train_code_embedding)
            sim_scores.append(score)
        sorted_indexes = [i for i, v in sorted(enumerate(sim_scores), key=lambda x: x[1], reverse=True)]
        for val in sorted_indexes[:10]:
            fp.write(str(val) + " ")
        fp.write('\n')

def pre_process_samples_semantic(test_codes, training_codes):
    model = SentenceTransformer("flax-sentence-embeddings/st-codesearch-distilroberta-base")
    test_codes_embeddings, training_codes_embeddings = [], []
    st = time.time()
    for i in range(len(test_codes)):
        test_code = test_codes[i]
        code1_emb = model.encode(test_code, convert_to_tensor=True)
        test_codes_embeddings.append(code1_emb)
    ed = time.time()
    print('Test code embedding generate finish!')
    print(str(ed - st))
    for i in range(len(training_codes)):
        train_code = training_codes[i]
        code1_emb = model.encode(train_code, convert_to_tensor=True)
        training_codes_embeddings.append(code1_emb)
    print('Training code embedding generate finish!')
    with open('sim_semantic.txt', 'w') as fp:
     for i in range(len(test_codes)):
        test_code_embedding = test_codes_embeddings[i]
        sim_scores = []
        for j in range(len(training_codes)):
            train_code_embedding = training_codes_embeddings[j]
            hits = util.semantic_search(test_code_embedding, train_code_embedding)[0]
            top_hit = hits[0]
            score = top_hit['score']
            sim_scores.append(score)
        sorted_indexes = [i for i, v in sorted(enumerate(sim_scores), key=lambda x: x[1], reverse=True)]
        for val in sorted_indexes[:10]:
            fp.write(str(val) + " ")
        fp.write('\n')

if __name__ == '__main__':

    training_codes, training_comments, training_labels, test_codes, test_comments, test_labels = get_data()
    pre_process_samples_token(test_codes, training_codes)
    pre_process_samples_semantic(test_codes, training_codes)