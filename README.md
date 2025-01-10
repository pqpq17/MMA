# [Paper] Integration of Multi-Source Medical Data for Medical Diagnosis Question Answering
The repository is for the paper "Integration of Multi-Source Medical Data for Medical Diagnosis Question Answering"

The main contributions can be summarized as follows:
(i) We explore a medical question-answering task based on multi-source data, named medical diagnosis question answering (MedDQA) task.
(ii) We introduce a new dataset constructed from real medical data as a foundation for future research.
(iii) We propose a multi-agent system where each agent can selectively handle specific data sources.

![](pics/workflow.png)


## Requirements

Install all required python dependencies:

```
pip install -r requirements.txt
```

## Dataset

Please check the Google Drive: https://drive.google.com/file/d/1zHmIb7Ej3lb3z6KO_pT9IfX0YBZJapFF/view?usp=sharing

Note that Our dataset is currently undergoing ethical review by the hospital. It will be made the whole dataset publicly available once the review process is completed.


## Implementation

### 1. Place the Dataset
Place the downloaded dataset in the `./MMA_Dataset` directory, including the `json_file` folder and `image` folder.

### 2. Configure the API Key
Add your API key to the `prompt_generator.py` file.

### 3. Run the Python File
Run the file using the following command:

```
python run.py
```

## Citation

If this repo is useful to you, please cite using this BibTeX.
```bibtex
@article{peng2024integration,
  title={Integration of Multi-Source Medical Data for Medical Diagnosis Question Answering},
  author={Peng, Qi and Cai, Yi and Liu, Jiankun and Zou, Quan and Chen, Xing and Zhong, Zheng and Wang, Zefeng and Xie, Jiayuan and Li, Qing},
  journal={IEEE Transactions on Medical Imaging},
  year={2024},
  publisher={IEEE}
}
```