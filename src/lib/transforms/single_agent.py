def transform_single_agent(batch):
    """Used in pred_mode='single' or 'multi', predict only single agent. This is baseline."""
    return batch["image"], batch["target_positions"], batch["target_availabilities"]
