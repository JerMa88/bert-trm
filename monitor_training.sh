#!/bin/bash
# Monitor script for BERT-TRM training job

echo "======================================================================"
echo "BERT-TRM Training Job Monitor"
echo "======================================================================"
echo

# Find the most recent job
JOBID=$(squeue -u jerryma -n BERT-TRM -h -o "%i" | head -1)

if [ -z "$JOBID" ]; then
    echo "No active BERT-TRM job found. Checking recent job outputs..."
    echo
    
    # Find most recent output file
    LATEST_OUTPUT=$(ls -t /projects/jdworak/jerry_ma_AD_ML/idea_density/lms/bert-trm/output_bert_trm_*.log 2>/dev/null | head -1)
    
    if [ -n "$LATEST_OUTPUT" ]; then
        echo "Most recent output file: $LATEST_OUTPUT"
        echo "======================================================================"
        echo "Last 50 lines of output:"
        echo "======================================================================"
        tail -50 "$LATEST_OUTPUT"
    else
        echo "No output files found yet."
    fi
else
    echo "Active Job ID: $JOBID"
    echo "Job Status:"
    squeue -j $JOBID
    echo
    
    # Check if output file exists yet
    OUTPUT_FILE="/projects/jdworak/jerry_ma_AD_ML/idea_density/lms/bert-trm/output_bert_trm_${JOBID}.log"
    
    if [ -f "$OUTPUT_FILE" ]; then
        echo "======================================================================"
        echo "Last 50 lines of output:"
        echo "======================================================================"
        tail -50 "$OUTPUT_FILE"
    else
        echo "Output file not created yet. Job may be queued."
    fi
fi

echo
echo "======================================================================"
echo "To monitor in real-time, use:"
echo "  tail -f output_bert_trm_*.log"
echo
echo "To cancel the job, use:"
echo "  scancel $JOBID"
echo "======================================================================"
