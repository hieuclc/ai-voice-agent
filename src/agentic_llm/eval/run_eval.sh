python ../agent/evaluate_rag.py --law-file law_questions.json \
                                --adm-file admission_questions.json\
                                --tour-file tour_questions.json \
                                --output ./results/eval_results.csv \
                                --concurrency 32 \
                                --from-raw law:results/eval_results_law_raw.json tour:results/eval_results_tour_raw.json admission:results/eval_results_admission_raw.json
