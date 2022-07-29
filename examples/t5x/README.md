# T5X Integration

This example demonstrates how to use a Rax ranking loss to optimize a ranking
task with a T5X model.

> ⚠️ **Disclaimer**: Fine-tuning large language models such as T5 is often done
> on TPU machines running on Google Cloud. The Rax developers are **not**
> responsible for any billing costs incurred by running this example code.

## Prerequisites

First, make sure you are able to run the basic T5X german-to-english translation
example on a Cloud TPU VM as described
[here](https://github.com/google-research/t5x/blob/84528ed20b4bf55b05a593b7a4fbf0bc8b4d6c01/README.md#example-english-to-german-translation).
This involves setting up a google cloud account with an appropriate project,
permissions and billing.

## Setup a Cloud TPU VM

First, you should launch a Cloud TPU VM:

```shell
GCP_ZONE="us-central1-a"
GCP_VM_NAME="rax-t5x"
GCP_ACCELERATOR="v3-8"

# Authenticate
gcloud auth login

# Create TPU VM
gcloud alpha compute tpus tpu-vm create $GCP_VM_NAME \
  --zone $GCP_ZONE \
  --accelerator-type $GCP_ACCELERATOR \
  --version tpu-vm-base

# Connect to TPU VM
gcloud alpha compute tpus tpu-vm ssh $GCP_VM_NAME \
  --zone $GCP_ZONE
```

> ⚠️ **Remember to shut down your Cloud TPU VM after you finish training**. You
> will continue to be billed if you do not.

## Install code and dependencies

Inside the new Cloud TPU VM, run the following to set up T5X:

```shell
# Clone and set up T5X
export PATH="$HOME/.local/bin:$PATH"
git clone --branch=main https://github.com/google-research/t5x
cd t5x

# Install T5X dependencies
python3 -m pip install -e '.[tpu]' -f \
  https://storage.googleapis.com/jax-releases/libtpu_releases.html
pip uninstall -y jax && pip install jax[tpu] -f \
  https://storage.googleapis.com/jax-releases/libtpu_releases.html
pip uninstall -y seqio seqio-nightly && pip install -U seqio-nightly

# Install Rax
pip install git+https://github.com/google/rax.git
```

Clone the Rax repo to get the T5X example code and copy it into the T5X examples
dir:

```shell
# Copy the Rax example code to the T5X example dir.
git clone https://github.com/google/rax
mkdir -p t5x/examples/rax
cp rax/examples/t5x/* t5x/examples/rax/
```

## Start training

Now you are ready to fine-tune a T5 model on the MS-MARCO QNA v2.1 task with the
Rax softmax loss:

```shell
# Train model.
NOW=$(date +%Y%m%d)
export GOOGLE_CLOUD_BUCKET_NAME=...
export TFDS_DATA_DIR=gs://$GOOGLE_CLOUD_BUCKET_NAME/t5x/data
export MODEL_DIR="gs://$GOOGLE_CLOUD_BUCKET_NAME/t5x/output/${NOW}_softmax"

# Pre-download dataset.
tfds build "huggingface:ms_marco/v2.1" --data_dir=$TFDS_DATA_DIR

# Start training.
python3 ./t5x/train.py \
  --gin_file="t5x/examples/rax/msmarco.gin" \
  --tfds_data_dir="${TFDS_DATA_DIR}" \
  --gin.MODEL_DIR="'${MODEL_DIR}'" \
  --gin.USE_CACHED_TASKS="False"
```

To train with a different loss, for example the pointwise sigmoid loss, you can
run:

```shell
# Create config for sigmoid loss
cat <<EOF > t5x/examples/rax/msmarco_sigmoid.gin
from __gin__ import dynamic_registration
import rax
include 't5x/examples/rax/msmarco.gin'
LOSS_FN = @rax.pointwise_sigmoid_loss
EOF

# Train model
export MODEL_DIR="gs://$GOOGLE_CLOUD_BUCKET_NAME/t5x/output/${NOW}_sigmoid"
python3 ./t5x/train.py \
  --gin_file="t5x/examples/rax/msmarco_sigmoid.gin" \
  --tfds_data_dir="${TFDS_DATA_DIR}" \
  --gin.MODEL_DIR="'${MODEL_DIR}'" \
  --gin.USE_CACHED_TASKS="False"
```

## Cleanup

You should **shut down and delete** your Cloud TPU VM once you finish training.
You will continue to be billed if you do not do this! Instructions for deleting
your Cloud TPU VM are [here](https://cloud.google.com/tpu/docs/run-calculation-jax#cleaning_up) or run the following:

```shell
gcloud alpha compute tpus tpu-vm delete $GCP_VM_NAME \
  --zone $GCP_ZONE
```

