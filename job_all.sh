#!/bin/bash
#SBATCH --mail-user=Moslem.Yazdanpanah@gmail.com
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL
#SBATCH --job-name=student
#SBATCH --output=%x-%j.out
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=40
#SBATCH --mem=127000M
#SBATCH --time=2-05:15
#SBATCH --account=rrg-ebrahimi

nvidia-smi

source ~/ENV/bin/activate

cd $SLURM_TMPDIR

cp -r ~/scratch/STARTUP .
cd STARTUP
cd src

mkdir dataset
cd dataset
unzip ~/scratch/CD-FSL_Datasets/miniImagenet.zip

mkdir ChestX-Ray8 EuroSAT ISIC2018 plant-disease

# cd EuroSAT
# unzip ~/scratch/CD-FSL_Datasets/EuroSAT.zip
# cd ..

# cd ChestX-Ray8
# unzip ~/scratch/CD-FSL_Datasets/ChestX-Ray8.zip
# mkdir images
# find . -type f -name '*.png' -print0 | xargs -0 mv -t images

# cd ..
# cd ISIC2018
# unzip ~/scratch/CD-FSL_Datasets/ISIC2018.zip
# unzip ~/scratch/CD-FSL_Datasets/ISIC2018_GroundTruth.zip

# cd ..
cd plant-disease
unzip ~/scratch/CD-FSL_Datasets/plant-disease.zip


cd $SLURM_TMPDIR

cd STARTUP
cd src

# cd student_STARTUP
# bash run.sh
# cd $SLURM_TMPDIR
# zip -r ~/scratch/student_models.zip $SLURM_TMPDIR/STARTUP/src/student_STARTUP/

# cd evaluation
# bash run.sh
# cd $SLURM_TMPDIR
# zip -r ~/scratch/STARTUP/evaluation.zip $SLURM_TMPDIR/STARTUP/src/evaluation/

cd student_STARTUP
bash run_no_taskx.sh
cd $SLURM_TMPDIR
zip -r ~/scratch/student_models_taskx.zip $SLURM_TMPDIR/STARTUP/src/student_STARTUP/miniImageNet_source_no_taskx/


# cd evaluation
# bash run_no_taskx.sh
# cd $SLURM_TMPDIR
# zip -r ~/scratch/STARTUP/evaluation_no_taskx_80.zip $SLURM_TMPDIR/STARTUP/src/evaluation/

