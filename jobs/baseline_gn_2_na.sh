#!/bin/bash
#SBATCH --mail-user=ar.aamer@gmail.com
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL
#SBATCH --job-name=baseline_gn32na_seed2
#SBATCH --output=%x-%j.out
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=32
#SBATCH --mem=127000M
#SBATCH --time=1-00:00
#SBATCH --account=rrg-ebrahimi

nvidia-smi

source ~/my_env9/bin/activate

echo "------------------------------------< Data preparation>----------------------------------"
echo "Copying the source code"
date +"%T"
cd $SLURM_TMPDIR
cp -r ~/scratch/FeatureNorm .

echo "Copying the datasets"
date +"%T"
cp -r ~/scratch/CD-FSL_Datasets .

echo "creating data directories"
date +"%T"
cd FeatureNorm
cd data
unzip -q $SLURM_TMPDIR/CD-FSL_Datasets/miniImagenet.zip

mkdir ChestX-Ray8 EuroSAT ISIC2018 plant-disease

cd EuroSAT
unzip ~/scratch/CD-FSL_Datasets/EuroSAT.zip
cd ..

cd ChestX-Ray8
unzip ~/scratch/CD-FSL_Datasets/ChestX-Ray8.zip
mkdir images
find . -type f -name '*.png' -print0 | xargs -0 mv -t images
cd ..

cd ISIC2018
unzip ~/scratch/CD-FSL_Datasets/ISIC2018.zip
unzip ~/scratch/CD-FSL_Datasets/ISIC2018_GroundTruth.zip
cd ..

cd plant-disease
unzip ~/scratch/CD-FSL_Datasets/plant-disease.zip

echo "----------------------------------< End of data preparation>--------------------------------"
date +"%T"
echo "--------------------------------------------------------------------------------------------"

echo "---------------------------------------<Run the program>------------------------------------"
date +"%T"
cd $SLURM_TMPDIR

cd FeatureNorm

python baseline_mod.py --dir ./logs/baseline_gn32na_seed2 --bsize 128 --epochs 1000 --model resnet10_GN_na --seed 2 

python finetune.py --save_dir ./logs/baseline_gn32na_seed2 --target_dataset EuroSAT --subset_split datasets/split_seed_1/EuroSAT_labeled_80.csv --embedding_load_path ./logs/baseline_gn32na_seed2/checkpoint_best.pkl --seed 2 --model resnet10_GN_na --freeze_backbone &
python finetune.py --save_dir ./logs/baseline_gn32na_seed2 --target_dataset CropDisease --subset_split datasets/split_seed_1/CropDisease_labeled_80.csv --embedding_load_path ./logs/baseline_gn32na_seed2/checkpoint_best.pkl --seed 2 --model resnet10_GN_na --freeze_backbone &
python finetune.py --save_dir ./logs/baseline_gn32na_seed2 --target_dataset ISIC --subset_split datasets/split_seed_1/ISIC_labeled_80.csv --embedding_load_path ./logs/baseline_gn32na_seed2/checkpoint_best.pkl --seed 2 --model resnet10_GN_na --freeze_backbone &
wait
python finetune.py --save_dir ./logs/baseline_gn32na_seed2 --target_dataset ChestX --subset_split datasets/split_seed_1/ChestX_labeled_80.csv --embedding_load_path ./logs/baseline_gn32na_seed2/checkpoint_best.pkl --seed 2 --model resnet10_GN_na --freeze_backbone

echo "-----------------------------------<End of run the program>---------------------------------"
date +"%T"
echo "--------------------------------------<backup the result>-----------------------------------"
date +"%T"
cd $SLURM_TMPDIR
cp -r $SLURM_TMPDIR/FeatureNorm/logs/baseline_gn32na_seed2/ ~/scratch/FeatureNorm/logs/

