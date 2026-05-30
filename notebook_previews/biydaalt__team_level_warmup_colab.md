# biydaalt/team_level_warmup_colab.ipynb

Source notebook: [`biydaalt/team_level_warmup_colab.ipynb`](../biydaalt/team_level_warmup_colab.ipynb)

# Team-level MAPPO-like warmup

Зорилго: `magent2one4all.zip`-ээс warm-start хийгээд red team-ийн бүх agent дээр **shared policy / parameter-sharing PPO** сургана.

Энэ notebook нь `kaggle_train_mappo_like.py` runner-ийг дуудна. Runner нь:
- red agent бүрийг тусдаа training sample болгож өгнө
- нэг shared PPO policy сургана
- render хийхгүй
- env copy 1 ашиглана
- output-оо `/content/team_warmup_*` рүү хадгална

Upload хийх minimum folder:

```text
biydaalt/
  kaggle_train_mappo_like.py
  mappo_like_env.py
  opponents.py
  train.py
  battle_env.py
  model.py
  magent2one4all.zip
```

## 1. Dependencies

Энэ cell-ийг ажиллуулаад Colab `Runtime -> Restart session` хий. Restart хийсний дараа **энэ cell-ийг дахин ажиллуулахгүй**, 2-р cell-ээс үргэлжлүүл.

## Cell 3

```python
%pip install -q --force-reinstall "numpy==1.26.4"
%pip install -q --force-reinstall "gymnasium==1.1.1" "pettingzoo==1.24.3" "supersuit==3.9.3" "stable-baselines3[extra]==2.7.0" sb3-contrib pygame tensorboard magent2
print("Install done. Now restart runtime, then continue from Cell 2.")
```

## 2. Import test

## Cell 5

```python
import numpy as np
import torch
import gymnasium
import pettingzoo
import supersuit
import magent2
import stable_baselines3

print("numpy", np.__version__)
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("gymnasium", gymnasium.__version__)
print("pettingzoo", pettingzoo.__version__)
print("stable_baselines3", stable_baselines3.__version__)
print("OK")
```

## 3. Upload / unzip `biydaalt`

Хэрвээ `/content/biydaalt` folder аль хэдийн байгаа бол энэ cell юу ч хийхгүй. Байхгүй бол `biydaalt.zip` upload хийнэ.

## Cell 7

```python
from pathlib import Path
import os, zipfile

root = Path('/content/biydaalt')
if root.exists():
    print('Found:', root)
else:
    try:
        from google.colab import files
        print('Upload biydaalt.zip now...')
        uploaded = files.upload()
        for name in uploaded:
            if name.endswith('.zip'):
                with zipfile.ZipFile(name) as zf:
                    zf.extractall('/content')
                print('Extracted:', name)
    except Exception as e:
        print('Upload helper unavailable:', repr(e))

print('biydaalt exists:', root.exists())
if root.exists():
    print('\n'.join(str(p) for p in sorted(root.glob('*'))[:30]))
```

## 4. File check

## Cell 9

```python
from pathlib import Path

REQUIRED = [
    'kaggle_train_mappo_like.py',
    'mappo_like_env.py',
    'opponents.py',
    'train.py',
    'battle_env.py',
    'model.py',
    'magent2one4all.zip',
]
root = Path('/content/biydaalt')
missing = [name for name in REQUIRED if not (root / name).exists()]
print('root:', root)
print('missing:', missing)
assert not missing, 'Missing files: ' + ', '.join(missing)
```

## 5. Smoke test

Энэ 162 timestep test нь warm-start load, env step, save/eval ажиллаж байгааг шалгана. 1 минут орчим.

## Cell 11

```python
!python /content/biydaalt/kaggle_train_mappo_like.py \
  --strategy flee \
  --opponent-mode random \
  --total-timesteps 162 \
  --chunk-timesteps 162 \
  --n-steps 2 \
  --batch-size 162 \
  --n-epochs 1 \
  --max-steps 5 \
  --eval-episodes 1 \
  --device auto \
  --output-dir /content/team_warmup_smoke
```

## 6. Stage 1 warmup: random opponent

Энэ stage-ийн зорилго: red agents амьд үлдэх basic behavior сурах. Үр дүнгээс `survival_rate`, `avg_red_alive`, `avg_blue_alive`, `avg_reward` хар.

## Cell 13

```python
!python /content/biydaalt/kaggle_train_mappo_like.py \
  --strategy flee \
  --opponent-mode random \
  --total-timesteps 200000 \
  --chunk-timesteps 50000 \
  --n-steps 128 \
  --batch-size 2048 \
  --n-epochs 5 \
  --max-steps 500 \
  --eval-episodes 5 \
  --device auto \
  --output-dir /content/team_warmup_random
```

## 7. Stage 2 warmup: magent2 opponent

Stage 1 final checkpoint-оос үргэлжлүүлнэ. Random дээр survival тогтворжсоны дараа ажиллуул.

## Cell 15

```python
STAGE1_FINAL = '/content/team_warmup_random/checkpoints/ppo_kaggle_mappo_like_flee_final.zip'
!python /content/biydaalt/kaggle_train_mappo_like.py \
  --strategy flee \
  --opponent-mode magent2 \
  --resume {STAGE1_FINAL} \
  --total-timesteps 400000 \
  --chunk-timesteps 50000 \
  --n-steps 128 \
  --batch-size 2048 \
  --n-epochs 5 \
  --max-steps 500 \
  --eval-episodes 5 \
  --device auto \
  --output-dir /content/team_warmup_magent2
```

## 8. Zip outputs

Colab-аас download хийхэд зориулж outputs folder-уудыг zip болгоно.

## Cell 17

```python
!cd /content && zip -qr team_level_warmup_outputs.zip team_warmup_smoke team_warmup_random team_warmup_magent2 2>/dev/null || true
!ls -lh /content/team_level_warmup_outputs.zip
```

