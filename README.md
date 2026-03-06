# Property-aware Transport for Protein Optimization

We propose a conditional flow-matching framework for protein fitness prediction that learns a property-aware landscape on top of pretrained PLMs. By introducing an energy function tied to assay-specific fitness and a rank-consistent objective, it shapes the flow so that mutants are ordered coherently by their functional properties. In addition, the property-aware steering gate focuses learning on relevant positions, improving performance on diverse protein engineering tasks. Across ProteinGym and additional protein engineering benchmarks, RankFlow consistently matches or surpasses state-of-the-art supervised methods under the same training protocols, using far fewer trainable parameters than full PLM fine-tuning and offering a robust, transferable approach to protein fitness prediction.

## Requirements
- **Python**: 3.9  
- **PyTorch**: 2.4.1 with CUDA 11.8 (`pytorch`, `torchvision`, `torchaudio`, `pytorch-cuda=11.8`)  
- **Key libraries (pip)**:
  - `pytorch-lightning` / `lightning`
  - `torch-geometric`, `torch-scatter`, `torch-sparse`, `torch-cluster`
  - `fair-esm`
  - `openfold`
  - `hydra-core`, `omegaconf`, `hydra-optuna-sweeper`, `hydra-colorlog`
  - `optuna`
  - `numpy`, `scipy`, `pandas`

All exact versions are pinned in `environment.yml`.

## Installation

Create and activate the Conda environment:
```
>> conda env create --file environment.yml
>> conda activate rankflow
```

## Quick Start
Before training, place DMS and structure data in `./data`, and modify the data configuration file `./configs/data/proteingym.yaml`. Using a subfolder structure, we can, for example, organize the ProteinGym data as follows: create a main folder ProteinGym, and inside it place the reference file DMS_substitutions.csv along with two subfolders: DMS_ProteinGym_substitutions and ProteinGym_AF2_structures.

We can train and test RankFlow as follows.

```
>> bash ./scripts/run.sh
```

The program will start training with the default settings. To modify any hyperparameters, edit `configs/model/RankFlow.yaml` or `configs/rankflow.yaml`. In the example provided, we train on the assay `SPG1_STRSG_Olson_2014`. If you want to train on a different assay, update the assay_index field in `configs/data/proteingym.yaml`.
