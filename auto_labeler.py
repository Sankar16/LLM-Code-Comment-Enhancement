import json
import time
from tqdm import tqdm
from openai import OpenAI

client = OpenAI(api_key="YOUR_KEY")

INPUT_PATH = "labelled_data.json"
OUTPUT_PATH = "auto_labeled.json"

PROMPT = """
You are an expert senior software engineer.

Your task: Classify whether a comment is USEFUL for understanding the Ruby method it belongs to.

Definition of a USEFUL comment:
- Explains the method’s purpose, behavior, or intention
- Adds context that is NOT obvious from the code
- Explains constraints, side effects, or usage
- Describes WHY the method exists

Definition of a NOT USEFUL comment:
- Repeats the code or states the obvious
- Is a Rubocop or linter directive
- Is vague, generic, or irrelevant
- Does not explain purpose, usage, or constraints

Output format (strict JSON ONLY):
{
  "usefulness": "useful" or "not_useful"
}

Ruby Method:
'''{raw_code}'''

Comment:
'''{comment}'''
"""

def classify(code, comment):
    message = PROMPT.format(raw_code=code, comment=comment)
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": message}],
        temperature=0
    )
    text = resp.choices[0].message.content.strip()
    return json.loads(text)["usefulness"]

if __name__ == "__main__":
    data = json.load(open(INPUT_PATH))
    output = []

    for sample in tqdm(data):
        try:
            result = classify(sample["raw_code"], sample["comment"])
            sample["usefulness"] = result
        except:
            sample["usefulness"] = "not_useful"  # fallback
        output.append(sample)
        time.sleep(0.1)

    json.dump(output, open(OUTPUT_PATH, "w"), indent=2)
    print(f"Saved labeled dataset → {OUTPUT_PATH}")