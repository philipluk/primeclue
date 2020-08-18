#!/usr/bin/env bash

# Median:
# tail -n1 results_breast_cancer_* | cut -d' ' -f9 | sort -n | grep '0\.' | head -n 10 | tail -n1
run() {
  TIME=10
  for RUN in {1..19}; do
    date
    cargo run --release --example time_limit "/home/lw/projects/primeclue-hpl/test_data/primeclue_data/$1" "$TIME" > "results_$1_$RUN.txt"
  done
}

rm -Rf results_*
set -e
run breast_cancer
run companies
run credit_card_default
run crime
run heart
run hepatitis
run adult
run banknote
run online_news
