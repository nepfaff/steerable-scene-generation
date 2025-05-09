import os

is_rank_zero = os.getenv("LOCAL_RANK", "0") == "0"
