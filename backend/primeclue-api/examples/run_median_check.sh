#!/usr/bin/env bash

run() {
  date
  cargo run --release --example median_check "/home/lw/projects/primeclue/test_data/primeclue_data/$1" 10
}

set -e
run breast_cancer
run companies
run credit_card_default
run crime
run heart
run hepatitis
run stocks
run adult
run banknote
run online_news
