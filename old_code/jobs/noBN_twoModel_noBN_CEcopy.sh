#!/bin/bash
#SBATCH --mail-user=ar.aamer@gmail.com
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL
#SBATCH --job-name=noBN_twoModel_noBN_CEcopy
#SBATCH --output=%x-%j.out
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=32
#SBATCH --mem=127000M
#SBATCH --time=2-00:00
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

python noBN_twoModel_noBN_CEcopy.py --dir ./logs/noBN_twoModel_noBN_CEcopy --bsize 128 --epochs 1000 --model resnet10

python finetune.py --save_dir ./logs/noBN_twoModel_noBN_CEcopy --target_dataset EuroSAT --subset_split datasets/split_seed_1/EuroSAT_labeled_80.csv --embedding_load_path ./logs/noBN_twoModel_noBN_CEcopy/checkpoint_best.pkl --freeze_backbone &
python finetune.py --save_dir ./logs/noBN_twoModel_noBN_CEcopy --target_dataset CropDisease --subset_split datasets/split_seed_1/CropDisease_labeled_80.csv --embedding_load_path ./logs/noBN_twoModel_noBN_CEcopy/checkpoint_best.pkl --freeze_backbone &
python finetune.py --save_dir ./logs/noBN_twoModel_noBN_CEcopy --target_dataset ISIC --subset_split datasets/split_seed_1/ISIC_labeled_80.csv --embedding_load_path ./logs/noBN_twoModel_noBN_CEcopy/checkpoint_best.pkl --freeze_backbone &
wait
python finetune.py --save_dir ./logs/noBN_twoModel_noBN_CEcopy --target_dataset ChestX --subset_split datasets/split_seed_1/ChestX_labeled_80.csv --embedding_load_path ./logs/noBN_twoModel_noBN_CEcopy/checkpoint_best.pkl --freeze_backbone

echo "-----------------------------------<End of run the program>---------------------------------"
date +"%T"
echo "--------------------------------------<backup the result>-----------------------------------"
date +"%T"
cd $SLURM_TMPDIR
cp -r $SLURM_TMPDIR/FeatureNorm/logs/noBN_twoModel_noBN_CEcopy/ ~/scratch/FeatureNorm/logs/

