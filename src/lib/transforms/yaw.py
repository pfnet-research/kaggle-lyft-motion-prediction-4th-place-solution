def transform_yaw(batch):
    """Used in pred_mode='single' or 'multi', predict only single agent. This is baseline."""
    return batch["image"], batch["yaw"]
