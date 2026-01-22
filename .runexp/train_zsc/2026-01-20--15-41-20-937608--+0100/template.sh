#!/bin/bash

#SBATCH --job-name=train_zsc
#SBATCH --output=/gpfs/home/acad/ulg-sysmod/mamodei/repos/cbir-mm-fgw/.runexp/train_zsc/2026-01-20--15-41-20-937608--+0100/%A_%a_%x.out
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
#SBATCH --mem=60GB
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --account=ariacpg
#SBATCH --array=0-8%4

commands=(
  '/gpfs/home/acad/ulg-sysmod/mamodei/repos/cbir-mm-fgw/.venv/bin/python3 /gpfs/home/acad/ulg-sysmod/mamodei/repos/cbir-mm-fgw/.venv/bin/cbir_mm_fgw_train_zscore_corr /gpfs/home/acad/ulg-sysmod/mamodei/repos/cbir-mm-fgw/.runexp/train_zsc/2026-01-20--15-41-20-937608--+0100/train_zscore_all-0.yaml --runexp-no-dry-run'
  '/gpfs/home/acad/ulg-sysmod/mamodei/repos/cbir-mm-fgw/.venv/bin/python3 /gpfs/home/acad/ulg-sysmod/mamodei/repos/cbir-mm-fgw/.venv/bin/cbir_mm_fgw_train_zscore_corr /gpfs/home/acad/ulg-sysmod/mamodei/repos/cbir-mm-fgw/.runexp/train_zsc/2026-01-20--15-41-20-937608--+0100/train_zscore_all-1.yaml --runexp-no-dry-run'
  '/gpfs/home/acad/ulg-sysmod/mamodei/repos/cbir-mm-fgw/.venv/bin/python3 /gpfs/home/acad/ulg-sysmod/mamodei/repos/cbir-mm-fgw/.venv/bin/cbir_mm_fgw_train_zscore_corr /gpfs/home/acad/ulg-sysmod/mamodei/repos/cbir-mm-fgw/.runexp/train_zsc/2026-01-20--15-41-20-937608--+0100/train_zscore_all-2.yaml --runexp-no-dry-run'
  '/gpfs/home/acad/ulg-sysmod/mamodei/repos/cbir-mm-fgw/.venv/bin/python3 /gpfs/home/acad/ulg-sysmod/mamodei/repos/cbir-mm-fgw/.venv/bin/cbir_mm_fgw_train_zscore_corr /gpfs/home/acad/ulg-sysmod/mamodei/repos/cbir-mm-fgw/.runexp/train_zsc/2026-01-20--15-41-20-937608--+0100/train_zscore_all-3.yaml --runexp-no-dry-run'
  '/gpfs/home/acad/ulg-sysmod/mamodei/repos/cbir-mm-fgw/.venv/bin/python3 /gpfs/home/acad/ulg-sysmod/mamodei/repos/cbir-mm-fgw/.venv/bin/cbir_mm_fgw_train_zscore_corr /gpfs/home/acad/ulg-sysmod/mamodei/repos/cbir-mm-fgw/.runexp/train_zsc/2026-01-20--15-41-20-937608--+0100/train_zscore_all-4.yaml --runexp-no-dry-run'
  '/gpfs/home/acad/ulg-sysmod/mamodei/repos/cbir-mm-fgw/.venv/bin/python3 /gpfs/home/acad/ulg-sysmod/mamodei/repos/cbir-mm-fgw/.venv/bin/cbir_mm_fgw_train_zscore_corr /gpfs/home/acad/ulg-sysmod/mamodei/repos/cbir-mm-fgw/.runexp/train_zsc/2026-01-20--15-41-20-937608--+0100/train_zscore_all-5.yaml --runexp-no-dry-run'
  '/gpfs/home/acad/ulg-sysmod/mamodei/repos/cbir-mm-fgw/.venv/bin/python3 /gpfs/home/acad/ulg-sysmod/mamodei/repos/cbir-mm-fgw/.venv/bin/cbir_mm_fgw_train_zscore_corr /gpfs/home/acad/ulg-sysmod/mamodei/repos/cbir-mm-fgw/.runexp/train_zsc/2026-01-20--15-41-20-937608--+0100/train_zscore_all-6.yaml --runexp-no-dry-run'
  '/gpfs/home/acad/ulg-sysmod/mamodei/repos/cbir-mm-fgw/.venv/bin/python3 /gpfs/home/acad/ulg-sysmod/mamodei/repos/cbir-mm-fgw/.venv/bin/cbir_mm_fgw_train_zscore_corr /gpfs/home/acad/ulg-sysmod/mamodei/repos/cbir-mm-fgw/.runexp/train_zsc/2026-01-20--15-41-20-937608--+0100/train_zscore_all-7.yaml --runexp-no-dry-run'
  '/gpfs/home/acad/ulg-sysmod/mamodei/repos/cbir-mm-fgw/.venv/bin/python3 /gpfs/home/acad/ulg-sysmod/mamodei/repos/cbir-mm-fgw/.venv/bin/cbir_mm_fgw_train_zscore_corr /gpfs/home/acad/ulg-sysmod/mamodei/repos/cbir-mm-fgw/.runexp/train_zsc/2026-01-20--15-41-20-937608--+0100/train_zscore_all-8.yaml --runexp-no-dry-run'
)
export RUNEXP_EXEC_KEY="2026-01-20--15-41-20-937608--+0100--${SLURM_ARRAY_TASK_ID}"


# ------------------------------------------------------------------------------
# Setting up the environment
# ------------------------------------------------------------------------------

echo "----------------- Environment ------------------"
module purge
module load EasyBuild/2022a
module load CUDA/11.7.0
module list

# ------------------------------------------------------------------------------
# Printing some information
# ------------------------------------------------------------------------------

echo "------------------- Job info -------------------"
echo "job_id             : $SLURM_JOB_ID"
echo "jobname            : $SLURM_JOB_NAME"
echo "queue              : $SLURM_JOB_PARTITION"
echo "qos                : $SLURM_JOB_QOS"
echo "account            : $SLURM_JOB_ACCOUNT"
echo "submit dir         : $SLURM_SUBMIT_DIR"
echo "number of mpi tasks: $SLURM_NTASKS tasks"
echo "OMP_NUM_THREADS    : $OMP_NUM_THREADS"
echo "number of gpus     : $SLURM_GPUS_ON_NODE"
echo "Executable         : $EXEC"

echo "------------------- Node list ------------------"
echo $SLURM_JOB_NODELIST

echo "---------------- Checking limits ---------------"
ulimit -a

# ------------------------------------------------------------------------------
# And finally running the code
# ------------------------------------------------------------------------------

echo "--------------- Running the code ---------------"

echo -n "This run started on: "
date

set -x

source ./env.sh
${commands[@]:$SLURM_ARRAY_TASK_ID:1}
res=$?

echo -n "This run completed on: "
date
exit $res

