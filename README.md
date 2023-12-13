# Learning Shared Features from Specific and Ambiguous Descriptions for Text-Based Person Search

## Prepare Datasets

We follow the [IRRA](https://github.com/anosorae/IRRA?tab=readme-ov-file#prepare-datasets) dataloader code. Just set the dataset path in `configs` `PATH.DATASETS` option.

## Training

Use the vscode [launch](https://code.visualstudio.com/docs/python/debugging) file or use the following bash command line:

```bash
python train.py --config-file configs/cuhk_pedes/two_stream.yaml --device-num 0
```

## Result

On CUHK-PEDES

| Method | Rank-1 | Rank-5 | Rank-10 |  mAp  |  mINP |
|:------:|:------:|:------:|:-------:|:-----:|:-----:|
|  LFSA  |  75.20 |  89.23 |  93.41  | 69.50 | 56.00 |

On ICFG-PEDES

| Method | Rank-1 | Rank-5 | Rank-10 |  mAp  |  mINP |
|:------:|:------:|:------:|:-------:|:-----:|:-----:|
|  LFSA  |  66.83 |  80.07 |  84.88  | 46.85 | 14.28 |

## Acknowledgments

The code structure is based on [TextReID](https://github.com/BrandonHanx/TextReID), the dataloader and optimization part refers the [IRRA](https://github.com/anosorae/IRRA), and the model code refers the [CLIP](https://github.com/openai/CLIP). Many thanks to them for their contributions in this field.
