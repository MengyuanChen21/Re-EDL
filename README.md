# [TPAMI 2025] Revisiting Essential and Nonessential Settings of Evidential Deep Learning

> **Authors**: Mengyuan Chen, Junyu Gao, Changsheng Xu.

> **Affiliations**: Institute of Automation, Chinese Academy of Sciences

> **Paper**: https://arxiv.org/pdf/2410.00393

### Dependencies:
Here we list our used requirements and dependencies.
Theoretically, the specific versions of dependencies should not affect the performance of the method.
 - GPU: GeForce RTX 3090
 - Python: 3.8.5
 - PyTorch: 1.12.0
 - Pandas: 1.1.3
 - Scikit-learn: 1.0.1
 - Wandb: 0.12.6
 - Tqdm: 4.62.3

### Data preparation:
The required datasets (CIFAR-10/CIFAR-100/SVHN/GTSRB/LFWPeople/Places365/Food101) will be automatically downloaded if your server has an Internet connection.

### Pre-trained models:
The pre-trained models of EDL, I-EDL, R-EDL, and Re-EDL can be downloaded from [Google Disk](https://drive.google.com/file/d/1kaqxdDH30UZ47uuYfPZS798AUBn-yekw/view?usp=sharing).

They need to be unzipped and put in the directory './code_classical/saved_models/'.

### Test pre-trained models:
Just run:
   ```
   python main.py --configid "2_cifar10/cifar10-{method-name}-test" --suffix test
   ```
where "method-name" should be replaced by "edl", "iedl", "redl", or "reedl".

### Train from scratch:
   
Just run:
   ```
   python main.py --configid "2_cifar10/cifar10-{method-name}-train" --suffix test
   ```

