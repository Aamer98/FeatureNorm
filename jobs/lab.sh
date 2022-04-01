#!/bin/bash
#SBATCH --mail-user=Moslem.Yazdanpanah@gmail.com
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL
#SBATCH --job-name=lab
#SBATCH --output=%x-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --mem=127000M
#SBATCH --time=3-00:00
#SBATCH --account=rrg-ebrahimi

nvidia-smi

module load python/3.7
source ~/ENV/bin/activate

echo "------------------------------------< Data preparation>----------------------------------"
echo "Copying the source code"
date +"%T"
cd $SLURM_TMPDIR
cp -r ~/scratch/BMS .

echo "Copying the datasets"
date +"%T"
cp -r ~/scratch/CD-FSL_Datasets .

echo "creating data directories"
date +"%T"
cd BMS
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

cd BMS


python lab.py

echo "-----------------------------------<End of run the program>---------------------------------"
date +"%T"
echo "--------------------------------------<backup the result>-----------------------------------"
date +"%T"
cd $SLURM_TMPDIR
cp -r $SLURM_TMPDIR/BMS/lab/ ~/scratch/BMS/
