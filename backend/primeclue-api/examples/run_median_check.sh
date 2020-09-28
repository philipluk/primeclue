#!/usr/bin/env bash

run() {
  date
  cargo run --release --example median_check "../../../test_data/primeclue_data/$1" 60
}

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
run fizz_buzz
