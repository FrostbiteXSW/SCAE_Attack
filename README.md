# **An Evasion Attack against Stacked Capsule Autoencoder**

This is the official source code of the paper: ["An Evasion Attack against Stacked Capsule Autoencoder"](https://arxiv.org/abs/2010.07230).

* **Author**: Jiazhu Dai, Siwei Xiong
* **Institution**: Shanghai University
* **Email**: daijz@shu.edu.cn (J. Dai)

Code files under /capsules folder were copied from [akosiorek/stacked_capsule_autoencoders](https://github.com/akosiorek/stacked_capsule_autoencoders) before 2020/9/24.

Executable .py files:

* *train.py*: Train the SCAE model and save it under /checkpoints folder. Dataset cache files are saved under /datasets folder.
* *test.py*: Test the model under /checkpoints folder, generate and save the K-Means classifier at the same place.
* *attack_gdu.py*: Launch the attack with GDU algorithm using the model under /checkpoints folder. Results are saved under /results/gdu folder.
* *attack_psc.py*: Launch the attack with PSC algorithm using the model under /checkpoints folder. Results are saved under /results/psc folder.
* *attack_opt.py*: Launch the attack with OPT algorithm using the model under /checkpoints folder. Results are saved under /results/opt folder.

Please feel free to use it as you like.