#!/bin/bash

# Check if the input file is provided
if [ $# -ne 1 ]; then
    echo "Usage: $0 <input_file>"
    exit 1
fi

input_file=$1
output_file="run_metrics.csv"

# Add the headers and extract data using sed
{
  echo "Avg_prompt_throughput,Avg_generation_throughput,Running,Swapped,Pending,GPU_KV_cache_usage,CPU_KV_cache_usage"
  sed -nE 's/^.*Avg prompt throughput: ([0-9.]+) tokens\/s, Avg generation throughput: ([0-9.]+) tokens\/s, Running: ([0-9]+) reqs, Swapped: ([0-9]+) reqs, Pending: ([0-9]+) reqs, GPU KV cache usage: ([0-9.]+)%, CPU KV cache usage: ([0-9.]+)%.*/\1,\2,\3,\4,\5,\6,\7/p' "$input_file"
} > "$output_file"

echo "Metrics extracted to $output_file"
