export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.3 && \
python ../../train_rlpd_copy.py "$@" \
    --exp_name=ram_insertion \
    --checkpoint_path=first_run \
    --learner \