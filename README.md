# SDRN

#### Requirement:

```
  python==3.10.14
  torch==2.4.0
  numpy==1.26.4
```

#### Dataset:
14-Res, 14-Lap, 15-Res: Download from https://drive.google.com/drive/folders/1wWK6fIvfYP-54afGDRN44VWlXuUAHs-l?usp=sharing

#### Download BERT_Base:
https://github.com/google-research/bert



#### How to run:
```
Link Kaggle:
https://www.kaggle.com/code/iamnotfuc/sdrn-train
```

# OTE_MTL
## Usage

* Download pretrained GloVe embeddings with this [link](http://nlp.stanford.edu/data/wordvecs/glove.840B.300d.zip) and extract `glove.840B.300d.txt` into `glove/`.
* Train with command, optional arguments could be found in [train.py](/train.py), **--v2** denotes whether test on datav2
```bash
python train.py --model mtl --dataset rest14 [--v2]
```
* Infer with [infer.py](/infer.py)

#### How to run:
```
Link Kaggle:
https://www.kaggle.com/code/iamnotfuc/ote-mtl
```