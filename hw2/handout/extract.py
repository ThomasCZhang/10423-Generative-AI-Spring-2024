import torch

def cosine_schedule(timesteps, s=0.008):
    """
    Defines the cosine schedule for the diffusion process,
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    Args:
        timesteps: The number of timesteps.
        s: The strength of the schedule.
    Returns:
        The computed alpha.
    """
    steps = timesteps + 1
    x = torch.linspace(0, steps, steps)
    alphas_cumprod = torch.cos(((x / steps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    alphas = alphas_cumprod[1:] / alphas_cumprod[:-1]
    return torch.clip(alphas, 0.001, 1)

def extract(a, t, x_shape):
    """
    Extracts the tensor at the given time step.
    Args:
        a: The total time steps.
        t: The time step to extract.
        x_shape: The reference shape.
    Returns:
        The extracted tensor.
    """
    b, *_ = t.shape
    out = a.gather(-1, t)
    print(*((1,) * (len(x_shape) - 1)))
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

img = torch.ones((10,10))
b = img.shape[0]
a = cosine_schedule(10)
t = torch.full([b], 3)
x_shape = img.shape
print(extract(a, t, x_shape))