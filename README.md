# Mixed-Curvature VAE for Single-Cell Data Embeddings

This repository is adapted from [mvae](https://github.com/oskopek/mvae) to embed single-cell data into mixed-curvature spaces. The embedded data can be later on used for [Federated Learning applications](https://github.com/sauravpr/hyperbolic_federated_classification).

## Installation

The required packages are listed in `environment.yml`.

## Example Usage

Please see the shell files such as `run_sc.sh` for more details.

## Contact

Please contact Chao Pan (chaopan2@illinois.edu) and Saurav Prakash (sauravp2@illinois.edu) if you have any questions.

## Citation

Please consider citing our paper and the original mvae paper if you find our repository useful!

```
@article{
prakash2023federated,
title={Federated Classification in Hyperbolic Spaces via Secure Aggregation of Convex Hulls},
author={Prakash, Saurav and Sima, Jin and Pan, Chao and Chien, Eli and Milenkovic, Olgica},
journal={arXiv preprint arXiv:2308.06895},
year={2023},
url={https://arxiv.org/abs/2308.06895}
}
```

```
@inproceedings{
Skopek2020Mixed-curvature,
title={Mixed-curvature Variational Autoencoders},
author={Ondrej Skopek and Octavian-Eugen Ganea and Gary Bécigneul},
booktitle={International Conference on Learning Representations},
year={2020},
url={https://openreview.net/forum?id=S1g6xeSKDS}
}
```
