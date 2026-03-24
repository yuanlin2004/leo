#leo evaluate \
#--environment appworld \
#--appworld-root /tmp/appworld-data \
#--task-id 82e2fac_1 \
#--experiment-name my-appworld-run-v2 \
#--output-root /tmp/leo-appworld-runs \
#--provider openrouter \
#--max-iterations 20

rm -rf /tmp/leo-appworld-runs/appworld-test-normal-base-openrouter-minmax27
#PYTHONPATH=src python -m leo.cli.main evaluate \
leo evaluate \
  --environment appworld \
  --appworld-root /Users/yuan/Documents/GitHub/appworld \
  --experiment-name appworld-test-normal-base-openrouter-minmax27 \
  --output-root /tmp/leo-appworld-runs \
  --dataset test_normal \
  --task-limit 5 \
  --max-iterations 30 \
  --agent react \
  --provider openrouter \
  --model minimax/minimax-m2.7 \
   2>&1 | tee artifacts/log


#  --extra-sys-prompt artifacts/code-dict-man.txt \
#  --log-level concise 2>&1 | tee log
#  --dataset test_normal \
#  --task-limit 5 \
#  --task-id 82e2fac_1 \
#  --task-limit 5 \
#  --log-level concise 2>&1 | tee log
#  --log-level trace 2>&1 | tee log
#  --extra-sys-prompt sys.txt \
#  --task-id 325d6ec_3\
#  --task-id 82e2fac_1 \
#  --extra-sys-prompt sys.txt \
#  --provider ollama \
#  --model 'gpt-oss:20b' \
#  --provider openrouter \
#  --model minimax/minimax-m2.5 \
#  --model minimax/minimax-m2.7 \
#  --model openai/gpt-5.3-codex\
#  --model minimax/minimax-m2.5 \
#  --model openai/gpt-oss-120b \
#  --model openai/gpt-5.3-codex\
#  --model openai/gpt-oss-20b \

#PYTHONPATH=src python -m leo.cli.main evaluate --agent plan-execute --environment appworld --appworld-root /tmp/appworld-data --task-id 82e2fac_1 --experiment-name appworld-plan-execute-rerun-82e2fac_1-20260315 --output-root /tmp/leo-appworld-runs --max-iterations 20
