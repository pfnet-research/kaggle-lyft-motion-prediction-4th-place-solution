import sys
import os
sys.path.append(os.pardir)
sys.path.append(os.path.join(os.pardir, os.pardir))
from lib.transforms.single_agent import transform_single_agent
from lib.transforms.multi_agent import transform_multi_agent, collate_fn_multi_agent
from lib.transforms.rnn_head_single_agent import transform_rnn_head_single_agent
from lib.transforms.yaw import transform_yaw

pred_mode_to_transform = {
    "single": transform_single_agent,
    "multi": transform_single_agent,
    "multi_agent": transform_multi_agent,
    "rnn_head_multi": transform_rnn_head_single_agent,
    "multi_deep_ensemble": transform_single_agent,
    "yaw": transform_yaw,
}

pred_mode_to_collate_fn = {
    "single": None,  # Use default
    "multi": None,  # Use default
    "multi_agent": collate_fn_multi_agent,
    "rnn_head_multi": None,  # Use default
    "multi_deep_ensemble": None,  # Use default
    "yaw": None,  # Use default
}
