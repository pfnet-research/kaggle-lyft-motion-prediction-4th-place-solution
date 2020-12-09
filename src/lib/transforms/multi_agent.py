import torch


def transform_multi_agent(batch):
    """Used in pred_mode='single' or 'multi', predict only single agent. This is baseline."""
    image = torch.as_tensor(batch["image"])
    centroid_pixel = torch.as_tensor(batch["centroid_pixel"])
    target_positions = torch.as_tensor(batch["target_positions"])
    target_availabilities = torch.as_tensor(batch["target_availabilities"])
    return image, centroid_pixel, target_positions, target_availabilities


def collate_fn_multi_agent(data_list):
    image = torch.stack([d[0] for d in data_list], dim=0)
    centroid_pixel = torch.cat([d[1] for d in data_list], dim=0).type(torch.long)
    target_positions = torch.cat([d[2] for d in data_list], dim=0)
    target_availabilities = torch.cat([d[3] for d in data_list], dim=0)
    batch_agents = torch.cat([torch.full((d[1].shape[0],), i, dtype=torch.long) for i, d in enumerate(data_list)])
    return image, centroid_pixel, batch_agents, target_positions, target_availabilities
