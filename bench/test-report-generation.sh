#!/bin/bash

# Test script to verify the benchmark report generation
# This simulates what the CI will do

cd "$(dirname "$0")/../build" || exit 1

ITERATIONS=2  # Use fewer iterations for testing

# Initialize results directory
mkdir -p benchmark-results

echo "Running benchmark configurations (this may take a while)..."

# Run all combinations
for n in 100 200 400; do
  for d in 2 4 8; do
    for bench in kriging nugget noise; do
      echo "Running: n=$n d=$d benchmark=$bench iterations=$ITERATIONS"
      bash ../bench/bench.sh iterations=$ITERATIONS n=$n d=$d $bench > "benchmark-results/${bench}_n${n}_d${d}.log" 2>&1
    done
  done
done

echo ""
echo "Generating markdown report..."

REPORT="benchmark-results/BENCHMARK_REPORT.md"

# Header
echo "# C++ Kriging Benchmarks Report" > "$REPORT"
echo "" >> "$REPORT"
echo "**Date:** $(date -u '+%Y-%m-%d %H:%M:%S UTC')" >> "$REPORT"
echo "**Platform:** $(uname -s) $(uname -r)" >> "$REPORT"
echo "**Build Type:** Release" >> "$REPORT"
echo "**Iterations per test:** $ITERATIONS" >> "$REPORT"
echo "**Test function:** f(x) = sum_i sin(2*pi*x_i)" >> "$REPORT"
echo "" >> "$REPORT"

# Table of Contents
echo "## Table of Contents" >> "$REPORT"
echo "" >> "$REPORT"
echo "- [Summary Tables](#summary-tables)" >> "$REPORT"
echo "  - [Kriging](#kriging)" >> "$REPORT"
echo "  - [NuggetKriging](#nuggetkriging)" >> "$REPORT"
echo "  - [NoiseKriging](#noisekriging)" >> "$REPORT"
echo "- [Detailed Results](#detailed-results)" >> "$REPORT"
echo "" >> "$REPORT"

# Summary Tables
echo "## Summary Tables" >> "$REPORT"
echo "" >> "$REPORT"

# Function to extract fit time from log
extract_fit_mean() {
  local logfile=$1
  grep "^fit" "$logfile" | awk '{print $3}'
}

extract_predict_mean() {
  local logfile=$1
  grep "^predict" "$logfile" | awk '{print $3}'
}

extract_update_mean() {
  local logfile=$1
  grep "^update[^_]" "$logfile" | awk '{print $3}'
}

# Kriging Summary
echo "### Kriging" >> "$REPORT"
echo "" >> "$REPORT"
echo "| n   | d | fit (ms) | predict (ms) | update (ms) |" >> "$REPORT"
echo "|-----|---|----------|--------------|-------------|" >> "$REPORT"
for n in 100 200 400; do
  for d in 2 4 8; do
    logfile="benchmark-results/kriging_n${n}_d${d}.log"
    if [ -f "$logfile" ]; then
      fit=$(extract_fit_mean "$logfile")
      pred=$(extract_predict_mean "$logfile")
      upd=$(extract_update_mean "$logfile")
      echo "| $n | $d | $fit | $pred | $upd |" >> "$REPORT"
    fi
  done
done
echo "" >> "$REPORT"

# NuggetKriging Summary
echo "### NuggetKriging" >> "$REPORT"
echo "" >> "$REPORT"
echo "| n   | d | fit (ms) | predict (ms) | update (ms) |" >> "$REPORT"
echo "|-----|---|----------|--------------|-------------|" >> "$REPORT"
for n in 100 200 400; do
  for d in 2 4 8; do
    logfile="benchmark-results/nugget_n${n}_d${d}.log"
    if [ -f "$logfile" ]; then
      fit=$(extract_fit_mean "$logfile")
      pred=$(extract_predict_mean "$logfile")
      upd=$(extract_update_mean "$logfile")
      echo "| $n | $d | $fit | $pred | $upd |" >> "$REPORT"
    fi
  done
done
echo "" >> "$REPORT"

# NoiseKriging Summary
echo "### NoiseKriging" >> "$REPORT"
echo "" >> "$REPORT"
echo "| n   | d | fit (ms) | predict (ms) | update (ms) |" >> "$REPORT"
echo "|-----|---|----------|--------------|-------------|" >> "$REPORT"
for n in 100 200 400; do
  for d in 2 4 8; do
    logfile="benchmark-results/noise_n${n}_d${d}.log"
    if [ -f "$logfile" ]; then
      fit=$(extract_fit_mean "$logfile")
      pred=$(extract_predict_mean "$logfile")
      upd=$(extract_update_mean "$logfile")
      echo "| $n | $d | $fit | $pred | $upd |" >> "$REPORT"
    fi
  done
done
echo "" >> "$REPORT"

# Detailed Results
echo "## Detailed Results" >> "$REPORT"
echo "" >> "$REPORT"

for bench in kriging nugget noise; do
  BENCH_NAME="${bench^}"
  if [ "$bench" = "nugget" ]; then BENCH_NAME="NuggetKriging"; fi
  if [ "$bench" = "noise" ]; then BENCH_NAME="NoiseKriging"; fi

  echo "### $BENCH_NAME" >> "$REPORT"
  echo "" >> "$REPORT"

  for n in 100 200 400; do
    for d in 2 4 8; do
      logfile="benchmark-results/${bench}_n${n}_d${d}.log"
      if [ -f "$logfile" ]; then
        echo "#### Configuration: n=$n, d=$d" >> "$REPORT"
        echo "" >> "$REPORT"
        echo "\`\`\`" >> "$REPORT"
        cat "$logfile" | grep -A 100 "n=$n d=$d" | grep -v "Results saved" >> "$REPORT"
        echo "\`\`\`" >> "$REPORT"
        echo "" >> "$REPORT"
      fi
    done
  done
done

echo ""
echo "Report generated at: $REPORT"
echo ""
echo "Preview:"
echo "=========================================="
cat "$REPORT"
