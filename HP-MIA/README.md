# Code for HP-MIA

HP-MIA: A novel membership inference attack scheme for high membership prediction precision

## Requirements
Please refer to `requirements.txt` to configure the environment.

## Test case 

This repository provides the functions for calculating the membership scores (`Compute_score.py`) as well as the architecture of the neural network models used for testing (`models.py`). 

As a test case for HP-MIA, two `ipynb` files provide the implementation on Purchase100. The `npz` file for Purchase100 is provided in the `data` folder. `Train_ref_Purchase.ipynb` provides codes to train the target model/shadow model/reference model. The target model and shadow model will be stored in `target_shadow` folder. 100 well-trained reference models for Purchase100 are stored in `ref_purchase` folder. 

In addtion, `attack_exp_purchase.ipynb` tested our Two-stage HP-MIA on Purchaser100. As a reference, we show the results of the calibrated attack (C-Loss) without using membership exclusion.

## Test on your own datasets/models

This repository only provided models trained on Purchase100. If you want to test on other datasets, please refer to `Train_ref_Purchase.ipynb` to train your own model. You can add your own neural network structure in `models.py` if you need to. 

The `data` folder is used to store the dataset files. Processing for seven common datasets (mnist, fashionmnist, emnist, cifar, cifar100, purchase and texas) is provided in `my_dataloader.py`. If you want to test other datasets, you need to refine the processing of the new datasets in this file. 

After completing the preparations for model and dataset loader, you can start your own experiments by simply modifying the variables `dataset` and `Net` in `Train_ref_Purchase.ipynb` and `attack_exp_purchase.ipynb` ！

## Acknowledgement
This implementation references codes from [ml-leaks-pytorch](https://github.com/GeorgeTzannetos/ml-leaks-pytorch), 
[membership-inference-evaluation](https://github.com/inspire-group/membership-inference-evaluation) and [Purchase100 and Texas100 dataset](https://github.com/xehartnort/Purchase100-Texas100-datasets). Thank the authors for their  work !
