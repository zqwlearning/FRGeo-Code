# Feature Recombination Geo-Localization

[[Project](https://zqwlearning.github.io/FRGeo/)], [[Paper, AAAI 2024](https://ojs.aaai.org/index.php/AAAI/article/view/28554)]

:mega: **It has been accepted by AAAI-24**

This is a PyTorch implementation of the “Aligning Geometric Spatial Layout in Cross-View Geo-Localization via Feature Recombination”.


<div style="text-align:center;">
    <image src="images/overall.jpg" style="width:100%;" style="height:100%;"/>
    <p>
        <strong>
              (a) Overview of our proposed FRGeo model. (b) Illustration of the proposed Feature Recombination Module (FRM). (c) Illustration of the proposed Weighted (B + 1)-tuple Loss (WBL).
        </strong>
    </p>
</div>

# Requirement

```python
- Python >= 3.8, numpy, matplotlib, pillow, ptflops, timm
- PyTorch >= 1.8.1, torchvision >= 0.11.1
```

# Dataset

Please download [CVUSA](http://mvrl.cs.uky.edu/datasets/cvusa/), [CVACT](https://github.com/Liumouliu/OriCNN) and [VIGOR](https://github.com/Jeff-Zilence/VIGOR). You may need to modify  dataset path in "dataloader".

# Model Zoo

| Dataset          | R@1   | R@5   | R@10  | R@1%  | Hit   | Download                                                     |
| ---------------- | ----- | ----- | ----- | ----- | ----- | ------------------------------------------------------------ |
| CVUSA            | 97.06 | 99.25 | 99.47 | 99.85 | -     | [model](https://drive.google.com/drive/folders/1TYHR5wWY3tvrORwzrN9JJ5TXWgVVepeY?usp=share_link) |
| CVACT            | 90.35 | 96.45 | 97.25 | 98.74 | -     | [model](https://drive.google.com/drive/folders/1c-0DQYGA9uLI96uBf5dWucWVAwnTdSmb?usp=share_link) |
| VIGOR Same-Area  | 71.26 | 91.38 | 94.32 | 99.52 | 82.41 | [model](https://drive.google.com/drive/folders/11skVOv7bTVtkSKdv6UfVfSlS3xLFJyh9?usp=share_link) |
| VIGOR Cross-Area | 37.54 | 59.58 | 67.34 | 94.28 | 40.66 | [model](https://drive.google.com/drive/folders/1R1tNAoGtB2E8nZ1uD8pH1gGEiwv58glu?usp=share_link) |

# Usage

## Training

To train our method on CVUSA for 50 epochs run:

```shell
python -u train.py --lr 0.0001 --batch-size 32 --dist-url 'tcp://localhost:8080' --multiprocessing-distributed --world-size 1 --rank 0  --epochs 50 --save_path ./result_cvusa --op sam --wd 0.03 --mining --dataset cvusa --cos --dim 3072 --asam --rho 2.5
```

To train our method on CVACT for 50 epochs run:

```shell
python -u train.py --lr 0.0001 --batch-size 32 --dist-url 'tcp://localhost:8080' --multiprocessing-distributed --world-size 1 --rank 0  --epochs 50 --save_path ./result_cvact --op sam --wd 0.03 --mining --dataset cvact --cos --dim 3072 --asam --rho 2.5
```

To train our method on VIGOR Same-Area for 50 epochs run:

```shell
python -u train.py --lr 0.00005 --batch-size 16 --dist-url 'tcp://localhost:8080' --multiprocessing-distributed --world-size 1 --rank 0  --epochs 50 --save_path ./result_vigor --op sam --wd 0.03 --mining --dataset vigor --cos --dim 3072 --asam --rho 2.5
```

To train our method on VIGOR Cross-Area for 50 epochs run:

```shell
python -u train.py --lr 0.00005 --batch-size 16 --dist-url 'tcp://localhost:8080' --multiprocessing-distributed --world-size 1 --rank 0  --epochs 50 --save_path ./result_vigor_cross --op sam --wd 0.03 --mining --dataset vigor --cos --dim 3072 --asam --rho 2.5 --cross
```

## Evaluation

You should organize the downloaded pre-trained models in the following way:

```shell
./result_cvusa
	model_best.pth.tar
	checkpoint.pth.tar
	...
./result_cvact
	model_best.pth.tar
	checkpoint.pth.tar
	...
./result_vigor
	model_best.pth.tar
	checkpoint.pth.tar
	...
./result_vigor_cross
	model_best.pth.tar
	checkpoint.pth.tar
	...
```

To evaluate our method on CVUSA  val run:

```shell
python -u train.py --dist-url 'tcp://localhost:8080' --multiprocessing-distributed --world-size 1 --rank 0  --save_path ./result_cvusa --dataset cvusa --dim 3072 -e
```

To evaluate our method on CVACT _val run:

```shell
python -u train.py --dist-url 'tcp://localhost:8080' --multiprocessing-distributed --world-size 1 --rank 0  --save_path ./result_cvact --dataset cvact --dim 3072 -e
```

To evaluate our method on VIGOR Same-Area run:

```shell
python -u train.py --dist-url 'tcp://localhost:8080' --multiprocessing-distributed --world-size 1 --rank 0 --save_path ./result_vigor --dataset vigor --dim 3072 -e
```

To evaluate our method on VIGOR Cross-Area run:

```shell
python -u train.py --dist-url 'tcp://localhost:8080' --multiprocessing-distributed --world-size 1 --rank 0 --save_path ./result_vigor_cross --dataset vigor --dim 3072 --cross -e
```

# Feature Embedding Visualization

<div style="text-align:center;">
    <image src="images/SNE.jpg" style="width:80%;" style="height:50%;"/>
    <p>
        <strong>
             Feature embedding visualization of cross-view feature representations learned by ou
        </strong>
    </p>
</div>


# Retrieval Example

<div style="text-align:center;">
    <image src="images/match_cvusa.jpg" style="width:80%;" style="height:50%;"/>
    <p>
        <strong>
            (a) CVUSA.
        </strong>
    </p>
    <image src="images/match_cvact.jpg" style="width:80%;" style="height:50%;"/>
    <p>
        <strong>
            (a) CVACT.
        </strong>
    </p>
    <image src="images/match_vigor.jpg" style="width:80%;" style="height:50%;"/>
    <p>
        <strong>
            (c) VIGOR.
        </strong>
    </p>
    <p>
        <strong>
            Cross-view image-based retrieval examples on (a) CVUSA, (b) CVACT and (c) VIGOR.
        </strong>
    </p>
</div>
# References and Acknowledgements

[TransGeo](https://github.com/Jeff-Zilence/TransGeo2022)，[ConvNeXt](https://github.com/facebookresearch/ConvNeXt)，[CVUSA](http://mvrl.cs.uky.edu/datasets/cvusa/)，[VIGOR](https://github.com/Jeff-Zilence/VIGOR)，[OriCNN](https://github.com/Liumouliu/OriCNN)，[Deit](https://github.com/facebookresearch/deit)，[MoCo](https://github.com/facebookresearch/moco)

Please contact us if you have any questions.

# Citation

```latex
@inproceedings{zhang2024aligning,
  title={Aligning Geometric Spatial Layout in Cross-View Geo-Localization via Feature Recombination},
  author={Zhang, Qingwang and Zhu, Yingying},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={38},
  number={7},
  pages={7251--7259},
  year={2024}
}
```