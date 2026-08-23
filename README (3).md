# Cartoonization of Images Using GANs

CartoonGAN (CVPR 2018) in TensorFlow 2, with a CycleGAN alternative and a deployable Streamlit app.

```
.
├── CartoonGAN_Cartoonization.ipynb   training + inference + sample results (55 cells, 19 sections)
├── app.py                            Streamlit front-end
├── requirements.txt                  deployment deps (tensorflow-cpu)
├── Dockerfile                        container for any cloud VM
└── exported_models/                  created by Section 13 of the notebook
    └── my_cartoongan_SavedModel/
```

---

## 1. Datasets

CartoonGAN trains on **unpaired** data — a folder of real photos and a folder of cartoon images. Nothing needs to correspond.

| Dataset | Link | Role | Size |
|---|---|---|---|
| **selfie2anime** ⭐ default | https://www.kaggle.com/datasets/arnaud58/selfie2anime | already split into `trainA`/`trainB`/`testA` | ~390 MB |
| Landscape Pictures | https://www.kaggle.com/datasets/arnaud58/landscape-pictures | `trainA` — real scenery (closest to the paper) | ~600 MB |
| Anime Images Dataset | https://www.kaggle.com/datasets/diraizel/anime-images-dataset | `trainB` — anime frames | ~2 GB |
| Safebooru | https://www.kaggle.com/datasets/alamson/safebooru | `trainB` — large anime corpus | large |
| photo2cartoon | https://www.kaggle.com/datasets/arnaud58/photo2cartoon | photo ↔ cartoon pair | ~1 GB |
| selfie2anime (no Kaggle account) | https://huggingface.co/datasets/huggan/selfie2anime | same as above | ~390 MB |
| CycleGAN official sets | http://efrosgans.eecs.berkeley.edu/cyclegan/datasets/ | for Section 15 | varies |
| Google Cartoon Set | https://google.github.io/cartoonset/ | synthetic cartoon faces | 10k / 100k |

Required layout (built automatically by notebook cell 3.3):

```
datasets/cartoon/
├── trainA/          real photos
├── trainB/          cartoon images
├── trainB_smooth/   trainB with blurred edges  ← generated in 3.5
└── testA/           8 held-out photos
```

`trainB_smooth` is not optional. The discriminator labels blurred-edge cartoons as **fake**, which is what forces the generator to draw sharp outlines instead of soft colour blobs.

---

## 2. Running the notebook

Open in Colab → **Runtime ▸ Change runtime type ▸ GPU (T4)** → run cells in order.

| Section | What it does | Time (T4) |
|---|---|---|
| 1–2 | Install, config | 2 min |
| 3 | Download dataset, build `trainB_smooth` | 10–20 min |
| 4 | `tf.data` pipeline | instant |
| 5–6 | Generator, PatchGAN discriminator, VGG19 content loss | 1 min |
| 8 | **Phase 1** — initialization, content loss only | ~20 min |
| 9 | **Phase 2** — adversarial training | 2–4 h |
| 11–12 | Inference + **sample results grid** | 2 min |
| 13 | Export SavedModel / .keras / .tflite | 2 min |
| 14 | Pretrained CartoonGAN — results with zero training | 3 min |
| 15 | CycleGAN alternative | optional |
| 16–17 | Streamlit app + deployment | 5 min |

**In a hurry?** Run Sections 1–4, then jump straight to **Section 14** — it loads pretrained Shinkai/Paprika weights from the reference repo and gives you presentable results immediately.

Edit before running:
- cell **3.2** — your Kaggle username + API key (https://www.kaggle.com/settings ▸ API)
- cell **16.3** — your ngrok token, only if you want a public link from Colab

---

## 3. Streamlit app

### A. Locally

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py                                   # http://localhost:8501
```

Keep `exported_models/` in the same folder as `app.py`. To use pretrained styles instead of your own model:

```bash
git clone --depth 1 https://github.com/mnicnc404/CartoonGan-tensorflow.git
```

The app auto-detects `light_shinkai_SavedModel` and `light_paprika_SavedModel` inside it.

### B. From Colab (no local install)

Run notebook cell 16.3 — it launches Streamlit in the background and opens an ngrok tunnel. Without an ngrok account:

```python
!npm install -g localtunnel
!streamlit run app.py --server.port 8501 &>/dev/null &
!npx localtunnel --port 8501
```

### C. Streamlit Community Cloud (free hosting)

1. Push `app.py`, `requirements.txt`, `sample_images/`, `exported_models/` to a **public** GitHub repo.
2. Model must be under 100 MB, or use Git LFS, or download weights at runtime inside `load_model()`.
3. share.streamlit.io → **New app** → pick repo, branch, `app.py` → Deploy.
4. Keep `tensorflow-cpu` in requirements — full `tensorflow` exceeds the memory limit.

### D. Hugging Face Spaces (better for large models)

```bash
huggingface-cli login
huggingface-cli repo create cartoonizer --type space --space_sdk streamlit
git clone https://huggingface.co/spaces/<username>/cartoonizer && cd cartoonizer
cp -r ../app.py ../requirements.txt ../exported_models ../sample_images .
git lfs install && git lfs track "*.pb" "variables/*"
git add . && git commit -m "CartoonGAN cartoonizer" && git push
```

### E. Docker

```bash
docker build -t cartoonizer .
docker run -p 8501:8501 cartoonizer
```

Deploys as-is to Cloud Run, Render, Railway or any VM.

### Performance

| Setting | Effect |
|---|---|
| `@st.cache_resource` on `load_model` | loads the model once — **don't remove it** |
| `max_dim = 512` | 2–4 s per image on cloud CPU; 1024 px can take 15 s+ |
| `.tflite` model | ~4× smaller and faster on CPU |
| warm-up inference at startup | first user request isn't slow |

---

## 4. Troubleshooting

| Symptom | Fix |
|---|---|
| Output ≈ input, barely stylized | lower `CONTENT_LAMBDA` (10 → 4), train longer |
| Output is colour noise | lower `D_LR`, raise `CONTENT_LAMBDA`, more init epochs |
| `d_loss → 0` early | discriminator won — `D_LR = 1e-5`, or update D every other step |
| Blurry, no cartoon edges | `trainB_smooth` missing — re-run cell 3.5 |
| Checkerboard artifacts | swap `Conv2DTranspose` for `UpSampling2D` + `Conv2D` |
| OOM | `BATCH_SIZE=2`, `IMG_SIZE=128`, or enable mixed precision (cell 1.2) |
| `tensorflow_addons` errors | not needed — the notebook defines its own `InstanceNormalization` |
| Streamlit "no model found" | `exported_models/` must sit next to `app.py` |

---

## 5. References

1. Chen, Lai, Liu — *CartoonGAN: Generative Adversarial Networks for Photo Cartoonization*, CVPR 2018. [paper](https://openaccess.thecvf.com/content_cvpr_2018/papers/Chen_CartoonGAN_Generative_Adversarial_CVPR_2018_paper.pdf)
2. Zhu, Park, Isola, Efros — *Unpaired Image-to-Image Translation Using Cycle-Consistent Adversarial Networks*, ICCV 2017. [arXiv](https://arxiv.org/abs/1703.10593)
3. Johnson, Alahi, Fei-Fei — *Perceptual Losses for Real-Time Style Transfer*, ECCV 2016. [arXiv](https://arxiv.org/abs/1603.08155)
4. Wang, Yu — *Learning to Cartoonize Using White-box Cartoon Representations*, CVPR 2020. [repo](https://github.com/SystemErrorWang/White-box-Cartoonization)
5. Reference implementation — https://github.com/mnicnc404/CartoonGan-tensorflow
6. Edge-smoothing script — https://github.com/taki0112/CartoonGAN-Tensorflow
