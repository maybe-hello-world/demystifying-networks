# Demystifying Network Foundation Models

This repository contains the code and instructions to reproduce the results presented in the paper "Demystifying Network Foundation Models" (NeurIPS 2025).

## Requirements

This code was designed to be run on a machine with 4 GPUs A100 with 80Gb of memory. Please note, that some experiments may require a significant time to run (up to 12+ hours for silhouette score calculation).

## Preparation

### Precalculated embeddings

For most of the experiments (except input feature perturbation), we use the embeddings calculated using the frozen publicly available models used in the paper. These embeddings are included in the repository and all the paths in notebooks are set to use them. For most of the experiments that use precalculated embeddings, you will not need to prepare the full environment for all the models, but instead only prepare the default environment (see requirements.txt). 

The embeddings are available using the following link: https://1drv.ms/u/c/60b97bf95d6e1cc6/Ea_BaGgRvGdIhQMJ4gH_RU4BAc0URN1-0DMLQva0hWkBog?e=iAeyot

Please, download the embeddings and place them in the `data` folder. The folder `data` should include folders for each model, containing the embeddings for each dataset.

For reproducibility reasons, we also provide the code to recalculate the embeddings, which is located in the `src\embeddings_calculation` folder. Please, be advised that additional time and/or resources may be required to recalculate the embeddings (depending on your available hardware). See ```models\README.md``` for more details.

### Environment setup

Please, use the `requirements.txt` file to set up the environment. The code was tested on Python 3.9 and PyTorch 2.7.0. 

```bash
python -m venv env
source env/bin/activate
pip install -r requirements.txt
```

## Running the code
The folder `src` contains the code for all the experiments. Each experiment is located in a separate Jupyter notebook. 
All experiments in the `src\embeddings_calculation` and `src\perturb*.ipynb` will require full models and environments for reproduction. All other experiments could be run with precalculated embeddings.

## Citation
```
@article{beltiukov2025demyst,
 title={Demystifying Network Foundation Models},
 author={Beltiukov, Roman and Guthula, Satyandra and Guo, Wenbo and Willinger, Walter and Gupta, Arpit},
 journal={Advances in neural information processing systems (NeurIPS)},
 year={2025}
}
```
