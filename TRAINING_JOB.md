# BERT-TRM Training Job

## Job Submitted
- **Job ID**: 355812
- **Job Name**: BERT-TRM
- **Status**: Pending (in queue)
- **Resources**: 4 GPUs, 16 CPUs, 256GB RAM, 48 hours
- **Conda Environment**: torch2.8

## What the Job Does

The batch script (`train_bert_batch.sh`) performs the following steps:

1. **Data Preparation** (if needed):
   - Downloads Wikipedia articles
   - Builds Masked Language Model (MLM) dataset
   - Creates vocabulary of 10,000 tokens
   - Sequences with max length 256 tokens
   - 15% masking probability

2. **Model Training**:
   - Architecture: BERT with recursive reasoning
   - 4 layers, 256 hidden size, 8 attention heads
   - Multi-GPU distributed training (4 GPUs)
   - Global batch size: 256
   - Learning rate: 1e-4 with warmup
   - Checkpoints saved every 5,000 steps

## Monitoring the Job

### Check job status:
```bash
squeue -u jerryma
```

### Monitor training progress in real-time:
```bash
cd /projects/jdworak/jerry_ma_AD_ML/idea_density/lms/bert-trm
tail -f output_bert_trm_355812.log
```

### Use the monitoring script:
```bash
./monitor_training.sh
```

## Output Files

- **Training output**: `output_bert_trm_355812.log`
- **Error log**: `error_bert_trm_355812.log`
- **Model checkpoints**: `outputs/bert/*/checkpoint_*.pt`

## Managing the Job

### Cancel the job:
```bash
scancel 355812
```

### Check detailed job info:
```bash
scontrol show job 355812
```

## After Training Completes

The trained model checkpoints will be saved in:
```
outputs/bert/<timestamp>/checkpoint_*.pt
```

You can test the trained model using:
```bash
python test_bert_model.py --checkpoint <path_to_checkpoint>
```

## Configuration

The training uses these key parameters:
- **Global batch size**: 256 sequences per batch
- **Epochs**: 100,000 (training continues until time limit or convergence)
- **Evaluation interval**: Every 5,000 steps
- **Learning rate**: 1e-4 with cosine decay
- **Weight decay**: 0.01
- **Warmup steps**: 10,000

## Notes

- The job will run for up to 48 hours
- Checkpoints are saved at evaluation intervals
- WandB logging is disabled (can be enabled by commenting WANDB_MODE=disabled)
- All required libraries are pre-installed in the torch2.8 conda environment
