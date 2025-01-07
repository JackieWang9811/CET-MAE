# CET-MAE
Contrastive EEG-Text Masked Autoencoder Learn Transferable representations for EEG-to-Text Decoding

## Setup the Environment
You can first setup a conda python environment to make everything works well and then setup the working environment.
```bash
conda create -n eeg2text python=3.8
conda activate eeg2text
cd PATH_TO_PROJECT/CET-MAE
pip install -r requirements.txt
```

## Data Preparation

### Download Datasets
Following [AAAI 2022 EEG-To-Text](https://github.com/MikeWangWZHL/EEG-To-Text), we employ ZuCo benchmark as our test bench.

1. Download ZuCo v1.0 'Matlab files' for 'task1-SR','task2-NR','task3-TSR' from [Zurich Cognitive Language Processing Corpus: A simultaneous EEG and eye-tracking resource for analyzing the human reading process](https://osf.io/q3zws/files/) under 'OSF Storage' root, unzip and move all .mat files to `./zuco_dataset/task1-SR/Matlab_files`, `./zuco_dataset/task2-NR/Matlab_files`, `./zuco_dataset/task3-TSR/Matlab_files` respectively.
2. Download ZuCo v2.0 'Matlab files' for 'task1-NR' from [ZuCo 2.0: A Dataset of Physiological Recordings During Natural Reading and Annotation](https://osf.io/2urht/files/) under 'OSF Storage' root, unzip and move all .mat files to `./zuco_dataset/task2-NR-2.0/Matlab_files`.

### Preprocess Datasets
```bash
python data_factory/data2pickle_v1.py
python data_factory/data2pickle_v2.py
```

## Over all Pretraining & EEG-Text Training
```bash
bash cet_mae_eeg2text_gpu2_7575.sh
```

## Contrastive EEG-Text Pretraining

```bash
python pre_train_eval_cet_mae_later_project_7575.py -c config/train_eval_cet_mae_gpu2_7575.yaml
```

## EEG-Text Training
```bash
python train_decoding_eeg_2_text_cet_mae.py -c config/train_eval_decoding_eeg_text_gpu2_7575.yaml
```


## Testing
```bash
python eval_decoding_eeg_2_text_cet_mae.py -c config/train_eval_decoding_eeg_text_gpu2_7575.yaml
```

## Reference
If you find this repo useful, please consider citing:
```bash
@inproceedings{wang-etal-2024-enhancing-eeg,
    title = "Enhancing {EEG}-to-Text Decoding through Transferable Representations from Pre-trained Contrastive {EEG}-Text Masked Autoencoder",
    author = "Wang, Jiaqi  and Song, Zhenxi  and Ma, Zhengyu  and Qiu, Xipeng  and Zhang, Min  and Zhang, Zhiguo",
    booktitle = "Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = aug,
    year = "2024",
    address = "Bangkok, Thailand",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.acl-long.393/",
    doi = "10.18653/v1/2024.acl-long.393",
    pages = "7278--7292"
}
```

   
