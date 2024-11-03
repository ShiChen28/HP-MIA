# Code
The core of the code for HP-MIA (PyTorch implementation).

This repository provides the functions for calculating the membership scores (`Compute_score.py`) as well as the architecture of the neural network models used for testing (`models.py`). 

As a test case for HP-MIA, two ipynb files provide the implementation on Purchase100. `Train_ref_Purchase.ipynb` provides codes to train the target model/shadow model/reference model. The target model and shadow model are stored in `target_shadow`. 100 well-trained reference models for Purchase100 are stored in `ref_purchase`. In addtion, `attack_exp_purchase.ipynb` tested Two-stage HP-MIA on Purchaser100.

We have only provided models trained on Purchase100. If you want to test on other datasets, please refer to `Train_ref_Purchase.ipynb` to train your own model.

Please refer to `requirements.txt` to configure the environment.

This implementation references codes from [ml-leaks-pytorch](https://github.com/GeorgeTzannetos/ml-leaks-pytorch), 
[membership-inference-evaluation](https://github.com/inspire-group/membership-inference-evaluation) and [Purchase100 and Texas100 dataset](https://github.com/xehartnort/Purchase100-Texas100-datasets). Thank the authors for their  work !
