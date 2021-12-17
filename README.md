# Title

## Introduction

### Requirements
This codebase is tested with:  
1.  h5py==3.1.0
2.  joypy==0.2.5
3.  matplotlib==3.4.2
4.  numpy==1.21.0
5.  pandas==1.2.3
6.  Pillow==8.4.0
7.  scikit_learn==1.0.1
8.  scipy==1.6.0
9.  seaborn==0.11.2
10. torch==1.8.1
11. torchvision==0.9.1
12. tqdm==4.60.0

To install all requirements, use "pip install -r requirements.txt"

## Running Experiments 
### Dataset Preparation
**MiniImageNet and CD-FSL:** Download the datasets for CD-FSL benchmark following step 1 and step 2 here: https://github.com/IBM/cdfsl-benchmark

**ImageNet:** https://www.kaggle.com/c/imagenet-object-localization-challenge/data

**Set datasets path:** Set the appropriate dataset pathes in "configs.py".

**Source dataset names:** "ImageNet", "miniImageNet"

**Target dataset names:** "EuroSAT", "CropDisease", "ChestX", "ISIC"

**All the dataset train/validation split files located at "datasets/split_seed_1" directory**

**All Baseline (miniImageNet) are trained Using an adapted version of the "https://github.com/cpphoo/STARTUP" repository**

|               | Baseline BN (Table 1)                                                                                                                                                                                                                    |
|---------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| To Train:     | https://anonymous.4open.science/r/STARTUP-7260/tree/main/teacher_miniImageNet                                                                                                                                           |
| To Fine-Tune: | python finetune.py --save_dir ./logs/baseline_teacher --target_dataset {Target dataset name} --subset_split datasets/split_seed_1/{Target dataset name} \_labeled_80.csv --embedding_load_path ./logs/baseline_teacher/checkpoint_best.pkl --freeze_backbone  |
|               | [Pre-Trained Dictionary](https://anonymous.4open.science/r/FN_Model_Zoo-BBEE/Dictionaries/baseline_teacher/checkpoint_best.pkl)                                                                                                          |

|               | Baseline FN (Table 1)                                                                                                                                                                                                                     |
|---------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| To Train:     | https://anonymous.4open.science/r/STARTUP-7260/tree/main/teacher_miniImageNet_na                                                                                                                                            |
| To Fine-Tune: | python finetune.py --save_dir ./logs/baseline_na_teacher --target_dataset {Target dataset name} --subset_split datasets/split_seed_1/{Target dataset name} \_labeled_80.csv --embedding_load_path ./logs/baseline_na_teacher/checkpoint_best.pkl --freeze_backbone  |
|               | [Pre-Trained Dictionary](https://anonymous.4open.science/r/FN_Model_Zoo-BBEE/Dictionaries/baseline_na_teacher/checkpoint_best.pkl)                                                                                                          |

|               | AdaBN BN (Table 1)                                                                                                                                                                                                                     |
|---------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| To Train:     | python AdaBN.py --dir ./logs/AdaBN/{dataset Name} --base_dictionary logs/baseline_teacher/checkpoint_best.pkl --target_dataset $target_testset --target_subset_split datasets/split_seed_1/$target_testset\_unlabeled_20.csv --bsize 256 --epochs 10 --model resnet10 &                                                                                                                                             |
| To Fine-Tune: | python finetune.py --save_dir ./logs/AdaBN/{Target dataset name} --target_dataset {Target dataset name} --subset_split datasets/split_seed_1/{Target dataset name} \_labeled_80.csv --embedding_load_path ./logs/AdaBN/{Target dataset name}/checkpoint_best.pkl --freeze_backbone  |
|               | Pre-Trained Dictionary (replace {Target_dataset_name}): https://anonymous.4open.science/r/FN_Model_Zoo-BBEE/Dictionaries/AdaBN/{Target_dataset_name}/checkpoint_best.pkl                                                                                                        |

|               | AdaBN FN (Table 1)                                                                                                                                                                                                                     |
|---------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| To Train:     | python AdaBN_na.py --dir ./logs/AdaBN_na/{dataset Name} --base_dictionary logs/baseline_na_teacher/checkpoint_best.pkl --target_dataset $target_testset --target_subset_split datasets/split_seed_1/$target_testset\_unlabeled_20.csv --bsize 256 --epochs 10 --model resnet10                                                                                                                                             |
| To Fine-Tune: | python finetune.py --save_dir ./logs/AdaBN_na/{Target dataset name} --target_dataset {Target dataset name} --subset_split datasets/split_seed_1/{Target dataset name} \_labeled_80.csv --embedding_load_path ./logs/AdaBN_na/{Target dataset name}/checkpoint_best.pkl --freeze_backbone  |
|               | Pre-Trained Dictionary (replace {Target_dataset_name}): https://anonymous.4open.science/r/FN_Model_Zoo-BBEE/Dictionaries/AdaBN_na/{Target_dataset_name}/checkpoint_best.pkl                                                                                                        |

|               | Baseline BN  (ImageNet) (Table 2)                                                                                                                                                                                                                    |
|---------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| To Train:     | python ImageNet.py --dir ./logs/ImageNet/ --arch resnet18 --data ./data/ILSVRC/Data/CLS-LOC --gpu 0                                                                                                                                             |
| To Fine-Tune: | python ImageNet_finetune.py --save_dir ./logs/ImageNet --target_dataset {Target dataset name} --subset_split datasets/split_seed_1/{Target dataset name} \_labeled_80.csv --embedding_load_path ./logs/baseline_teacher/checkpoint_best.pkl --freeze_backbone  |
|               | [Pre-Trained Dictionary](https://anonymous.4open.science/r/FN_Model_Zoo-BBEE/Dictionaries/ImageNet/checkpoint_best.pkl)                                                                                                          |

|               | Baseline FN  (ImageNet) (Table 2)                                                                                                                                                                                                                    |
|---------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| To Train:     | python ImageNet_na.py --dir ./logs/ImageNet_na/ --arch resnet18 --data ./data/ILSVRC/Data/CLS-LOC --gpu 0                                                                                                                                             |
| To Fine-Tune: | python ImageNet_finetune.py --save_dir ./logs/ImageNet_na --target_dataset {Target dataset name} --subset_split datasets/split_seed_1/{Target dataset name} \_labeled_80.csv --embedding_load_path ./logs/baseline_teacher/checkpoint_best.pkl --freeze_backbone  |
|               | [Pre-Trained Dictionary](https://anonymous.4open.science/r/FN_Model_Zoo-BBEE/Dictionaries/ImageNet_na/checkpoint_best.pkl)                                                                                                          |

|               | Baseline beta  (ImageNet) (Table 2)                                                                                                                                                                                                                    |
|---------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| To Train:     | python ImageNet_nw.py --dir ./logs/ImageNet_nw/ --arch resnet18 --data ./data/ILSVRC/Data/CLS-LOC --gpu 0                                                                                                                                             |
| To Fine-Tune: | python ImageNet_finetune.py --save_dir ./logs/ImageNet_nw --target_dataset {Target dataset name} --subset_split datasets/split_seed_1/{Target dataset name} \_labeled_80.csv --embedding_load_path ./logs/baseline_teacher/checkpoint_best.pkl --freeze_backbone  |
|               | [Pre-Trained Dictionary](https://anonymous.4open.science/r/FN_Model_Zoo-BBEE/Dictionaries/ImageNet_nw/checkpoint_best.pkl)                                                                                                          |

|               | Baseline gamma  (ImageNet) (Table 2)                                                                                                                                                                                                                    |
|---------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| To Train:     | python ImageNet_nb.py --dir ./logs/ImageNet_nb/ --arch resnet18 --data ./data/ILSVRC/Data/CLS-LOC --gpu 0                                                                                                                                             |
| To Fine-Tune: | python ImageNet_finetune.py --save_dir ./logs/ImageNet_nb --target_dataset {Target dataset name} --subset_split datasets/split_seed_1/{Target dataset name} \_labeled_80.csv --embedding_load_path ./logs/baseline_teacher/checkpoint_best.pkl --freeze_backbone  |
|               | [Pre-Trained Dictionary](https://anonymous.4open.science/r/FN_Model_Zoo-BBEE/Dictionaries/ImageNet_nb/checkpoint_best.pkl)                                                                                                          |

|               | Near-domain few-shot evaluation (Table 4)|
|---------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|               | Baseline BN  |
| To Fine-Tune: | python finetune.py --save_dir ./logs/eval/baseline_teacher --target_dataset ImageNet_test --subset_split datasets/split_seed_1/ImageNet_val_labeled.csv --embedding_load_path ./logs/baseline_teacher/checkpoint_best.pkl --freeze_backbone  |
|               | [Pre-Trained Dictionary](https://anonymous.4open.science/r/FN_Model_Zoo-BBEE/Dictionaries/baseline_teacher/checkpoint_best.pkl) |
|               | Baseline FN  |
| To Fine-Tune: | python finetune.py --save_dir ./logs/eval/baseline_na_teacher --target_dataset ImageNet_test --subset_split datasets/split_seed_1/ImageNet_val_labeled.csv --embedding_load_path ./logs/baseline_na_teacher/checkpoint_best.pkl --freeze_backbone  |
|               | [Pre-Trained Dictionary](https://anonymous.4open.science/r/FN_Model_Zoo-BBEE/Dictionaries/baseline_teacher/checkpoint_best.pkl) |
|               | AdaBN BN  |
| To Adapt:     |  python AdaBN.py --dir ./logs/AdaBN_teacher/miniImageNet --base_dictionary logs/baseline_teacher/checkpoint_best.pkl --target_dataset ImageNet_test --target_subset_split datasets/split_seed_1/ImageNet_val_labeled.csv --bsize 256 --epochs 10 --model resnet10 |
| To Fine-Tune: | python finetune.py --save_dir ./logs/AdaBN_teacher/miniImageNet --target_dataset ImageNet_test --subset_split datasets/split_seed_1/ImageNet_val_labeled.csv --embedding_load_path ./logs/AdaBN_teacher/miniImageNet/checkpoint_best.pkl --freeze_backbone |
|               | [Pre-Trained Dictionary](https://anonymous.4open.science/r/FN_Model_Zoo-BBEE/Dictionaries/AdaBN_teacher/miniImageNet/checkpoint_best.pkl) |
|               | AdaBN FN  |
| To Adapt:     |  python AdaBN.py --dir ./logs/AdaBN_na_teacher/miniImageNet --base_dictionary logs/baseline_na_teacher/checkpoint_best.pkl --target_dataset ImageNet_test --target_subset_split datasets/split_seed_1/ImageNet_val_labeled.csv --bsize 256 --epochs 10 --model resnet10 |
| To Fine-Tune: | python finetune.py --save_dir ./logs/AdaBN_na_teacher/miniImageNet --target_dataset ImageNet_test --subset_split datasets/split_seed_1/ImageNet_val_labeled.csv --embedding_load_path ./logs/AdaBN_na_teacher/miniImageNet/checkpoint_best.pkl --freeze_backbone |
|               | [Pre-Trained Dictionary](https://anonymous.4open.science/r/FN_Model_Zoo-BBEE/Dictionaries/AdaBN_na_teacher/miniImageNet/checkpoint_best.pkl) |
