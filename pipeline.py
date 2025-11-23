import os, subprocess

dataset = "data/dreaddit_depsev.csv"

models = ["qwen3:8b-q8_0", "mistral:7b-instruct-v0.3-q8_0"]

prompts = {
    "zero":"prompts/zero_shot.txt",
    "few":"prompts/few_shot.txt",
    "zero_cot":"prompts/hidden_cot_zero_shot.txt",
    "few_cot":"prompts/hidden_cot_few_shot.txt"
}

for model in models:
    for pname, pfile in prompts.items():
        out = f"outputs/{pname}/{model.replace(':','_')}_{pname}.jsonl"
        if not os.path.exists(out):
            cmd = [
                "python","src/run_extraction.py",
                "--input", dataset,
                "--output", out,
                "--model", model,
                "--prompt", pfile
            ]
            subprocess.run(cmd)
        # Evaluate on both tasks
        subprocess.run(["python","src/evaluate.py",out,"both","results"],  check=True)