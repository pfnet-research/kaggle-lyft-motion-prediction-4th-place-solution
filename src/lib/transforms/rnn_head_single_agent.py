def transform_rnn_head_single_agent(batch):
    """Used in pred_mode='rnn_head_multi', predict only single agent."""
    return (
        batch["image"],
        batch["history_positions"],
        batch["history_availabilities"],
        batch["target_positions"],
        batch["target_availabilities"]
    )
