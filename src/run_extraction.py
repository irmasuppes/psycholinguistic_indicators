import argparse, json
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from utils import ollama_generate

def run_batch(input_file, output_file, model, prompt_file):
    df = pd.read_csv(input_file)
    posts = df["text"].tolist()
    labels = df[["label_stress","label_depression"]].to_dict("records")
    prompt_template = Path(prompt_file).read_text()
    think = True if 'cot' in prompt_file and 'qwen' in model else False

    out = []
    for i, (text, lbls) in tqdm(enumerate(zip(posts, labels)), total=len(posts)):
        temperature = float(0.2)
        data = ollama_generate(model, prompt_template, text, temperature, think)
        data["post_id"] = i
        data["text"] = text
        data["temperature"] = temperature
        data["think"] = think
        data["label_stress"] = lbls["label_stress"]
        data["label_depression"] = lbls["label_depression"]
        out.append(data)

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        for row in out:
            f.write(json.dumps(row) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="CSV with columns: text, label_stress, label_depression")
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", required=True)
    args = parser.parse_args()
    run_batch(args.input, args.output, args.model, args.prompt)
