import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from inspect import isfunction

def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


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
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

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


# normalization functions
def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5


# DDPM implementation
class Diffusion(nn.Module):
    def __init__(
        self,
        model,
        *,
        image_size,
        channels=3,
        timesteps=1000,
    ):
        super().__init__()
        self.channels = channels
        self.image_size = image_size
        self.model = model
        self.num_timesteps = int(timesteps)

        """
        Initializes the diffusion process.
            1. Setup the schedule for the diffusion process.
            2. Define the coefficients for the diffusion process.
        Args:
            model: The model to use for the diffusion process.
            image_size: The size of the images.
            channels: The number of channels in the images.
            timesteps: The number of timesteps for the diffusion process.
        """
        ## TODO: Implement the initialization of the diffusion process ##
        # 1. define the scheduler here
        # 2. pre-compute the coefficients for the diffusion process
        
        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        )

        alpha = torch.concat([torch.tensor([1]), cosine_schedule(self.num_timesteps)])
        alpha = alpha.to(self.device)
        sqrt_alpha = torch.sqrt(alpha)
        one_minus_alpha = 1. - alpha
        # sqrt_one_minus_alpha = torch.sqrt(one_minus_alpha)
        
        alpha_bar = torch.cumprod(alpha,-1)
        sqrt_alpha_bar = torch.sqrt(alpha_bar)
        one_minus_alpha_bar = 1.-alpha_bar
        sqrt_one_minus_alpha_bar = torch.sqrt(one_minus_alpha_bar)

        # Reverse process coefficients
        self.x0_hat_coef1 = 1/sqrt_alpha_bar[1:] # From t = 1 to T
        self.x0_hat_coef2 = sqrt_one_minus_alpha_bar[1:] # From t = 1 to T

        self.mu_tilda_coef1 = sqrt_alpha[1:]*(one_minus_alpha_bar[:-1])/(one_minus_alpha_bar[1:])
        self.mu_tilda_coef2 = sqrt_alpha_bar[:-1]*(one_minus_alpha[1:])/(one_minus_alpha_bar[1:])

        self.sigmas = torch.sqrt(one_minus_alpha_bar[:-1]/one_minus_alpha_bar[1:]*one_minus_alpha[1:])

        # Forward procees coefficients
        self.xt_coef1 = sqrt_alpha_bar
        self.xt_coef2 = sqrt_one_minus_alpha_bar   
        # ###########################################################


    # backward diffusion
    @torch.no_grad()
    def p_sample(self, x, t, t_index):
        """
        Samples from the reverse diffusion process at time t_index.
        Args:
            x: The initial image.
            t: a tensor of the time index to sample at.
            t_index: a scalar of the index of the time step.
        Returns:
            The sampled image.
        """
        ####### TODO: Implement the p_sample function #######
        # sample x_{t-1} from the gaussian distribution wrt. posterior mean and posterior variance
        # Hint: use extract function to get the coefficients at time t
        eta = self.model.forward(x, t)
        
        x0_coef1 = extract(self.x0_hat_coef1, t, x.shape)
        x0_coef2 = extract(self.x0_hat_coef2, t, x.shape)

        mu_coef1 = extract(self.mu_tilda_coef1, t, x.shape)
        mu_coef2 = extract(self.mu_tilda_coef2, t, x.shape)

        x0_hat = x0_coef1*(x - x0_coef2*eta)
        x0_hat = torch.clamp(x0_hat, -1, 1)
        mu_tilda = mu_coef1*x+mu_coef2*x0_hat

        if t_index == 0:
            return mu_tilda
        else:
            z = torch.randn(x.shape, device = self.device)
            sigma = extract(self.sigmas, t, x.shape)
            return mu_tilda + sigma*z
        # ####################################################

    @torch.no_grad()
    def p_sample_loop(self, img):
        """
        Samples from the noise distribution at each time step.
        Args:
            img: The initial image that randomly sampled from the noise distribution.
        Returns:
            The sampled image.
        """
        b = img.shape[0]
        device = img.device
        #### TODO: Implement the p_sample_loop function ####
        # 1. loop through the time steps from the last to the first
        # 2. inside the loop, sample x_{t-1} from the reverse diffusion process
        # 3. clamp and unnormalize the generted image to valid pixel range 
        # Hint: to get time index, you can use torch.full()
        for t in range(self.num_timesteps-1, -1, -1):
            img = self.p_sample(img, torch.full((b, ) , t, device = device), t)
        img = torch.clamp(img, -1, 1)
        img = unnormalize_to_zero_to_one(img)
        return img
        # ####################################################

    @torch.no_grad()
    def sample(self, batch_size):
        """
        Samples from the noise distribution at each time step.
        Args:
            batch_size: The number of images to sample.
        Returns:
            The sampled images.
        """
        self.model.eval()
        #### TODO: Implement the p_sample_loop function ####
        img = torch.randn((batch_size, self.channels, self.image_size, self.image_size), device = self.device)
        img = self.p_sample_loop(img)
        return img

    # forward diffusion
    def q_sample(self, x_0, t, noise=None):
        """
        Samples from the noise distribution at time t. Apply alpha interpolation between x_start and noise.
        Args:
            x_0: The initial image.
            t: The time index to sample at.
            noise: The noise tensor to sample from. If None, noise will be sampled.
        Returns:
            The sampled image.
        """
        ###### TODO: Implement the q_sample function #######
        if noise is None:
            noise = torch.randn(x_0.shape)

        xt_coef1 = extract(self.xt_coef1, t, x_0.shape)
        xt_coef2 = extract(self.xt_coef2, t, x_0.shape)

        x_t = xt_coef1*x_0 + xt_coef2*noise        
        return x_t

    def p_losses(self, x_0, t, noise):
        """
        Computes the loss for the forward diffusion.
        Args:
            x_0: The initial image.
            t: The time index to compute the loss at.
            noise: The noise tensor to use. If None, noise will be sampled.
        Returns:
            The computed loss.
        """
        ###### TODO: Implement the p_losses function #######
        # define loss function wrt. the model output and the target
        # Hint: you can use pytorch built-in loss functions: F.l1_loss
        if noise is None:
            noise = torch.randn(x_0.shape)

        # q sample to get x_t
        # get eta_theta using unet(x_t, t)
        x_t = self.q_sample(x_0, t, noise)

        
        pred_noise = self.model.forward(x_t, t)
        loss = F.l1_loss(noise, pred_noise)

        return loss
        # ####################################################

    def forward(self, x_0, noise):
        """
        Computes the loss for the forward diffusion.
        Args:
            x_0: The initial image.
            noise: The noise tensor to use.
        Returns:
            The computed loss.
        """
        b, c, h, w, device, img_size, = *x_0.shape, x_0.device, self.image_size
        assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
        ###### TODO: Implement the forward function #######
        t = torch.randint(1, self.num_timesteps, (b,), device = device)
        loss = self.p_losses(x_0, t, noise)
        # print('FID Loss: ', loss)
        return loss
