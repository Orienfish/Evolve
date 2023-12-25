# EVOLVE: Enhancing Unsupervised Continual Learning with Multiple Experts

This repo contains the implementation for paper:

Xiaofan Yu, Tajana Rosing, Yunhui Guo. "EVOLVE: Enhancing Unsupervised Continual Learning with Multiple Experts" in WACV 2024.

## File Structure

We emulate the hybrid mode of cloud and edge device in one environment.

* `OnlineContrast` holds the implementation of our method Evolve and CaSSLe. The code is implemented based on [SupContrast](https://github.com/HobbitLong/SupContrast).
* `UCL` holds the implementation of LUMP, PNN, SI, DER adapted from the original repo [UCL](https://github.com/divyam3897/UCL).

Dataset configuration for `OnlineContrast` and `UCL` is in `data_utils.py`. Model and training configuration is in `set_utils.py`. The shared evaluation functions are in `eval_utils.py`.

## Prerequisites

In each folder, set up the environment with `pip3 install -r requirements.txt`.

To use the pretrained models as teacher models, please download the zip file `pretrained_models.zip` from [here](https://drive.google.com/file/d/1dlH_-bBS6SXcQuuKDs03ID5n7tssj5RU/view?usp=sharing) and unzip it in the root directory.

### Dataset Preparation

We mainly focus on image classification while the methodology can be applied to more general scenarios. In this repo, we test with CIFAR-10 (10 classes), TinyImageNet (100 classes), CoRe50 (50 classes) and Stream-51 (51 classes).

For all methods in `OnlineContrast` and `UCL` folders, the shared root dataset directory is `datasets`. 

* All download and configuration should be completed automatically by the code.
* For TinyImageNet, sometimes the download cannot proceed normaly. If that is the case, please download the original [TinyImageNet](https://drive.google.com/file/d/1Sy3ScMBr0F4se8VZ6TAwDYF-nNGAAdxj/view). Unzip the file under `datasets/TINYIMG`.
* Our data loaders for CoRe50 and Stream51 is adapted from [Avalanche](https://github.com/ContinualAI/avalanche).

## Getting Started

We list the commands to fire our method and each baseline in the following lines.

### OnlineContrast

To run our method Evolve:

```bash
cd OnlineContrast/scripts
bash ./run-evolve.sh evolve <cifar10/tinyimagenet/core50/stream51> <iid/seq/seq-im> <trial#>
```

* We test with three types of data streams as discussed in the paper: (1) **iid**, (2) sequential classes (**seq**), and (3)sequential classes with imbalance lengths (**seq-im**). More details about data stream configuration are explained later.
* For all implementations, the last argument of `trial#` (e.g., `0,1,2`) determines the random seed configuration. Hence using the same `trial#` produces the same random selection.

You can run `run-cassle.sh` to run the corresponding baseline with similar argument format.

The linear evaluation of all methods is implemented in `main_linear.py` and `run_linear.sh`.

### UCL

We replace the original data loader with our own loader. For evaluation, we also adapt the original code with our own clustering and kNN classifier on the learned embeddings.

We run LUMP (`mixup` in argument), PNN, SI and DER as baselines to compare:

```bash
cd UCL
bash ./run-baseline.sh <mixup/pnn/si/der> <simclr/byol/simsiam/barlowtwins/vicreg> <cifar10/tinyimagenet/core50/stream51> <iid/seq/seq-im> <trial#>
```

## Data Stream Configuration

The single-pass and non-iid data streams are the key motivation of the paper, with more details discussed in our [precedent work](https://github.com/Orienfish/SCALE).

More implementation details can be found in `./data_utils.py` in the `SeqSampler` class.

## Citation

If you found the codebase useful, please consider cite our work.

## License

MIT

If you have any questions, please feel free to contact [x1yu@ucsd.edu](mailto:x1yu@ucsd.edu).
