# Code
The core of the code for HP-MIA (PyTorch implementation).

This repository provides the functions for calculating the membership scores (`HP-MIA/Compute_score.py`) as well as the architecture of the neural network models used for testing (`HP-MIA/models.py`). 

As a reference case for HP-MIA, two ipynb files provide the implementation on Purchase100. `HP-MIA/Train_ref_Purchase.ipynb` provides codes to train the target model/shadow model/reference model. The target model and shadow model are stored in `HP-MIA/target_shadow`. 100 well-trained reference models for Purchase100 are stored in `HP-MIA/ref_purchase`. `HP-MIA/attack_exp_purchase.ipynb` tested Two-stage HP-MIA on Purchaser100.


Please refer to `HP-MIA/requirements.txt` to configure the environment.

This implementation references codes from [ml-leaks-pytorch](https://github.com/GeorgeTzannetos/ml-leaks-pytorch), 
[membership-inference-evaluation](https://github.com/inspire-group/membership-inference-evaluation) and [Purchase100 and Texas100 dataset](https://github.com/xehartnort/Purchase100-Texas100-datasets). Thank the authors for their  work !
