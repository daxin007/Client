# Cross-variable Linear Integrated Enhanced Transformer for Multivariate Long-Term Time Series Forecasting (Client)

This is the official repo for Cross-variable Linear Integrated Enhanced Transformer for Multivariate Long-Term Time Series Forecasting (Client). 

## Getting Started

1. Install Python >= 3.6, and install the dependencies by:

```
pip install -r requirements.txt
```

2. You can obtain all the nine datasets from [[Google Drive]](https://drive.google.com/drive/folders/13Cg1KYOlzM5C7K8gK8NfC-F3EYxkM3D2?usp=sharing), [[Tsinghua Cloud]](https://cloud.tsinghua.edu.cn/f/84fbc752d0e94980a610/) or [[Baidu Drive]](https://pan.baidu.com/s/1r3KhGd0Q9PJIUZdfEYoymg?pwd=i9iy) provided in TimesNet and put them into the folder `./dataset`.

3. You can reproduce the experiment results through through the training scripts `./scripts/`, and the name of our model's scripts is started with 'Client'.

```
# ETTh1
bash ./scripts/ETT_script/Client_ETTh1.sh
# ECL
bash ./scripts/ECL_script/Client.sh
```

4. You can visualize the predictions of Client through the notebook 'visualization.ipynb'.

5. The origin experimental results of mask series are shown in 'mask_result.csv', and the origin experimental results of LTSF are shown in 'result_of_Client.txt'.

## Citation

If you find our repo useful, please cite our paper:

```
@misc{gao2023client,
      title={Client: Cross-variable Linear Integrated Enhanced Transformer for Multivariate Long-Term Time Series Forecasting}, 
      author={Jiaxin Gao and Wenbo Hu and Yuntian Chen},
      year={2023},
      eprint={2305.18838},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

## Acknowledgement

We appreciate the following repos for their valuable code base or datasets:

https://github.com/thuml/Time-Series-Library

https://github.com/cure-lab/LTSF-Linear
