# Kaggle MAPPO-like training

Kaggle notebook дээр эхний cell:

```bash
!pip install -q "stable-baselines3[extra]" sb3-contrib magent2 pettingzoo supersuit pygame tensorboard
```

`biydaalt` folder болон `magent2one4all.zip` файлыг Kaggle dataset болгож attach хийсэн бол:

```bash
!find /kaggle/input -name kaggle_train_mappo_like.py -o -name magent2one4all.zip
```

Run:

```bash
!python /kaggle/input/YOUR_DATASET_NAME/biydaalt/kaggle_train_mappo_like.py \
  --strategy flee \
  --opponent-mode random \
  --total-timesteps 200000 \
  --chunk-timesteps 50000 \
  --n-steps 128 \
  --batch-size 2048 \
  --n-epochs 5 \
  --max-steps 500 \
  --eval-episodes 5 \
  --device auto
```

Output:

```text
/kaggle/working/biydaalt_outputs/checkpoints/
/kaggle/working/biydaalt_outputs/logs/
```

Эхний run-д `--opponent-mode random` ашигла. Survival тогтворжсоны дараа:

```bash
--opponent-mode magent2
```

эсвэл checkpoint-үүд гарсны дараа:

```bash
--opponent-mode mixed
```

Kaggle-safe тохиргоо:

- env copy үргэлж `1`
- render disabled
- `SDL_VIDEODRIVER=dummy`
- output зөвхөн `/kaggle/working` рүү бичигдэнэ
