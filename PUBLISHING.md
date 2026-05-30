# Publishing

This repository is prepared for GitHub as a code/notebook repository. Model weights, checkpoints, videos, GIFs, logs, and archives are listed in `HF_ARTIFACTS.txt` and ignored by Git.

## GitHub

Set the real repository URL, then push:

```bash
git remote add origin git@github.com:<github-user>/hicheel-agent.git
git push -u origin main
```

If you use HTTPS:

```bash
git remote add origin https://github.com/<github-user>/hicheel-agent.git
git push -u origin main
```

## Hugging Face

Install and log in:

```bash
python -m pip install -U huggingface_hub
huggingface-cli login
```

Upload the artifact list:

```bash
while IFS= read -r file; do
  case "$file" in ''|\#*) continue ;; esac
  file="${file#./}"
  huggingface-cli upload <hf-user>/hicheel-agent-artifacts "$file" "$file" --repo-type model
done < HF_ARTIFACTS.txt
```
