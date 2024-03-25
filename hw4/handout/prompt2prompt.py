import torch
from diffusers import DiffusionPipeline
from typing import Union, Tuple, List, Dict, Optional, Callable
import torch
import abc
from tqdm.notebook import tqdm
import ptp_utils
import seq_aligner
import numpy as np
from PIL import Image
import cv2
from IPython.display import display


device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')


@torch.no_grad()
def text2image_ldm(
    model,
    prompt:  List[str],
    controller,
    num_inference_steps: int = 50,
    guidance_scale: Optional[float] = 7.,
    generator: Optional[torch.Generator] = None,
    latent: Optional[torch.FloatTensor] = None,
):
    """
    Processes the LDM pipeline using huggingface diffuser's API and ptp_utils with controller for AttentionControl.
    """
    ####TODO####

    # Hint: You will need to use ptp_utils.register_attention_control, ptp_utils.init_latent, ptp_utils.diffusion_step, 
    #       ptp_utils.latent2image, along with huggingface diffuser's API

    ptp_utils.register_attention_control(model, controller=controller)

    height = 256
    width = 256
    batch_size = len(prompt)
    # raise NotImplementedError

    # uncond_input = model.tokenizer(["" for _ in range(batch_size)]) # Returns a dictionary of lists
    uncond_input = model.tokenizer(["" for _ in range(batch_size)], padding='max_length', max_length = 77, return_tensors = 'pt').to(model.device) 
    uncond_embeddings = model.bert(uncond_input['input_ids'])[0]

    text_input = model.tokenizer(prompt, padding='max_length', max_length = 77, return_tensors = 'pt').to(model.device)
    text_embeddings = model.bert(text_input['input_ids'])[0]
    latent, latents = ptp_utils.init_latent(latent, model, height, width, generator=generator, batch_size=batch_size)
    latents.to(model.device)
    context = torch.cat([uncond_embeddings, text_embeddings])

    model.scheduler.set_timesteps(num_inference_steps)
    for t in tqdm(model.scheduler.timesteps):
        latents = ptp_utils.diffusion_step(model, controller, latents, context, t, guidance_scale)

    image = ptp_utils.latent2image(model.vqvae, latents)

    ####END_TODO####

    return image, latent 


class AttentionControl(abc.ABC):
    """
    Base class for EmptyControl, AttentionStore, AttentionReplace. The control mechanisms 
    are implemented in the forward method for each controller. 
    """

    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0

    def step_callback(self, x_t):
        return x_t

    def between_steps(self):
        return

    @abc.abstractmethod
    def forward (self, attn, is_cross: bool, place_in_unet: str):
        raise NotImplementedError

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        h = attn.shape[0]
        attn[h // 2:] = self.forward(attn[h // 2:], is_cross, place_in_unet)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()
        return attn

    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0


class EmptyControl(AttentionControl):
    """
    Empty controller that simply returns the attention in forward pass and does nothing.
    """

    def forward (self, attn, is_cross: bool, place_in_unet: str):
        return attn


class AttentionStore(AttentionControl):
    """
    Creates a bookkeeping list that tracks the attention weights for each attention layer 
    in different parts of the UNet. This class is only used for visualizing the cross attention 
    for each input word and is only used in the Cross Attention Visualization section. 
    """

    def __init__(self):
        super(AttentionStore, self).__init__()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

    @staticmethod
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [],  "mid_self": [],  "up_self": []}

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        if attn.shape[1] <= 16 ** 2:  # avoid memory overhead
            key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
            self.step_store[key].append(attn)
        return attn

    def between_steps(self):
        if len(self.attention_store) == 0:
            self.attention_store = self.step_store
        else:
            for key in self.attention_store:
                for i in range(len(self.attention_store[key])):
                    self.attention_store[key][i] += self.step_store[key][i]
        self.step_store = self.get_empty_store()

    def get_average_attention(self):
        average_attention = {key: [item / self.cur_step for item in self.attention_store[key]] for key in self.attention_store}
        return average_attention


    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

    


class AttentionReplace(AttentionControl, abc.ABC):
    """
    Controller for replacing the attention of an input word from the input prompt with the corresponding 
    word from the modified prompt. 
    """

    def __init__(self, prompts, tokenizer, num_steps: int,
                 cross_replace_steps: Union[float, Tuple[float, float], Dict[str, Tuple[float, float]]],
                 self_replace_steps: Union[float, Tuple[float, float]]):
        super(AttentionReplace, self).__init__()
        self.batch_size = len(prompts)
        if type(self_replace_steps) is float:
            self_replace_steps = 0, self_replace_steps

        ####TODO####
        
        # Hint: You will need to use ptp_utils.get_time_words_attention_alpha and seq_aligner.get_replacement_mapper.
            
        # raise NotImplementedError
        
        device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        if torch.backends.mps.is_available():
            device = torch.device('mps')

        self.cross_replace_alpha = ptp_utils.get_time_words_attention_alpha(prompts, num_steps, cross_replace_steps, tokenizer).to(device)
        self.num_self_replace = (self_replace_steps[0]*num_steps, self_replace_steps[1]*num_steps)
        self.mapper = seq_aligner.get_replacement_mapper(prompts, tokenizer).to(device)

        ####END_TODO####

    def step_callback(self, x_t):
        return x_t

    def replace_self_attention(self, attn_base, att_replace):

        ####TODO####
        # raise NotImplementedError

        if att_replace.shape[-1] < 256:
            attn = attn_base.unsqueeze(0).expand(att_replace.shape[0], *attn_base.shape) # Expand attn_base
        else:
            attn = att_replace 
        
        return attn
        
        ####END_TODO####

    def replace_cross_attention(self, attn_base, attn_replace):

        ####TODO####

        # raise NotImplementedError
    
        attn = torch.einsum('h i j, b j d -> b h i d', attn_base, self.mapper)

        return attn
    
        ####END_TODO####

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        if is_cross or (self.num_self_replace[0] <= self.cur_step < self.num_self_replace[1]):

            ####TODO####
            # First dimension of attn is number of batches * number of heads.
            num_heads = attn.shape[0] // self.batch_size 
            attn = attn.reshape(self.batch_size, num_heads, *attn.shape[1:])
            attn_base, attn_replace = attn[0], attn[1:]
            if is_cross:
                alpha_t = self.cross_replace_alpha[self.cur_step]
                attn[1:] = self.replace_cross_attention(attn_base, attn_replace) * alpha_t + attn_replace * (1-alpha_t)
            else:
                attn[1:] = self.replace_self_attention(attn_base, attn_replace)

            attn = attn.reshape(num_heads*self.batch_size, *attn.shape[2:])
            # raise NotImplementedError

            ####END_TODO####

        return attn


def text_under_image(image: np.ndarray, text: str, text_color: Tuple[int, int, int] = (0, 0, 0)):
    """
    Takes an image and a text string, then adds the text under the image.
    """
    h, w, c = image.shape
    offset = int(h * .2)
    img = np.ones((h + offset, w, c), dtype=np.uint8) * 255
    font = cv2.FONT_HERSHEY_SIMPLEX
    img[:h] = image
    textsize = cv2.getTextSize(text, font, 1, 2)[0]
    text_x, text_y = (w - textsize[0]) // 2, h + offset - textsize[1] // 2
    cv2.putText(img, text, (text_x, text_y ), font, 1, text_color, 2)
    return img


def create_image_grid(images, num_rows=1, offset_ratio=0.02, on_ipynb=True, save_path="output.jpg"):
    """
    Displays a list or a batch of images in a grid format.
    """
    if type(images) is list:
        num_empty = len(images) % num_rows
    elif images.ndim == 4:
        num_empty = images.shape[0] % num_rows
    else:
        images = [images]
        num_empty = 0

    empty_images = np.ones(images[0].shape, dtype=np.uint8) * 255
    images = [image.astype(np.uint8) for image in images] + [empty_images] * num_empty
    num_items = len(images)

    h, w, c = images[0].shape
    offset = int(h * offset_ratio)
    num_cols = num_items // num_rows
    image_ = np.ones((h * num_rows + offset * (num_rows - 1),
                      w * num_cols + offset * (num_cols - 1), 3), dtype=np.uint8) * 255
    for i in range(num_rows):
        for j in range(num_cols):
            image_[i * (h + offset): i * (h + offset) + h:, j * (w + offset): j * (w + offset) + w] = images[
                i * num_cols + j]

    pil_img = Image.fromarray(image_)
    
    if on_ipynb:
        display(pil_img)
    else:
        pil_img.save(save_path)


def aggregate_attention(prompts: List[str], attention_store: AttentionStore, res: int, from_where: List[str], is_cross: bool, select: int):
    """
    Obtains the aggregated attention maps, averaged over attentions in different parts of the UNet.
    """
    out = []
    attention_maps = attention_store.get_average_attention()
    num_pixels = res ** 2
    for location in from_where:
        for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
            if item.shape[1] == num_pixels:
                cross_maps = item.reshape(len(prompts), -1, res, res, item.shape[-1])[select]
                out.append(cross_maps)
    out = torch.cat(out, dim=0)
    out = out.sum(0) / out.shape[0]
    return out.cpu()


def show_cross_attention(tokenizer, prompts: List[str], attention_store: AttentionStore, res: int, from_where: List[str], select: int = 0):
    """
    Visualizes the cross attention w.r.t. each input words.
    """
    tokens = tokenizer.encode(prompts[select])
    decoder = tokenizer.decode
    attention_maps = aggregate_attention(prompts, attention_store, res, from_where, True, select)
    images = []
    for i in range(len(tokens)):
        image = attention_maps[:, :, i]
        image = 255 * image / image.max()
        image = image.unsqueeze(-1).expand(*image.shape, 3)
        image = image.numpy().astype(np.uint8)
        image = np.array(Image.fromarray(image).resize((256, 256)))
        image = text_under_image(image, decoder(int(tokens[i])))
        images.append(image)
    create_image_grid(np.stack(images, axis=0))


def run_and_display(prompts, controller, ldm, num_diffusion_steps=50, guidance_scale=5., latent=None, callback:Optional[Callable[[np.ndarray], np.ndarray]] = None, generator=None, on_ipynb=True):
    """
    Runs the prompt-to-prompt pipeline and displays the results. 
    """
    images, x_t = text2image_ldm(ldm, prompts, controller, latent=latent, num_inference_steps=num_diffusion_steps, guidance_scale=guidance_scale, generator=generator)
    if callback is not None:
        images = callback(images)
    save_path = " ".join(prompts) + ".jpg"
    create_image_grid(images, on_ipynb=on_ipynb, save_path=save_path)
    return images, x_t



        
if __name__ == '__main__':
    # See run_in_colab.ipynb for more examples of how to run Prompt to Prompt.
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    if torch.backends.mps.is_available():
        device = torch.device('mps')

    model_id = "CompVis/ldm-text2im-large-256"
    NUM_DIFFUSION_STEPS = 50
    GUIDANCE_SCALE = 5.
    MAX_NUM_WORDS = 77
    # load model and scheduler
    ldm = DiffusionPipeline.from_pretrained(model_id).to(device)
    tokenizer = ldm.tokenizer

    g_cpu = torch.Generator().manual_seed(42)
    prompts = ["A painting of a squirrel eating a burger"]
    controller = AttentionStore()
    images, latent = run_and_display(prompts, controller, ldm, num_diffusion_steps=NUM_DIFFUSION_STEPS, guidance_scale=GUIDANCE_SCALE, generator=g_cpu, on_ipynb=False)

    prompts = ["A painting of a squirrel eating a burger",
                "A painting of a lion eating a burger",
                "A painting of a cat eating a burger",
                "A painting of a deer eating a burger",
              ]
    controller = AttentionReplace(prompts, tokenizer, NUM_DIFFUSION_STEPS, cross_replace_steps=.8, self_replace_steps=.2)
    _ = run_and_display(prompts, controller, ldm, num_diffusion_steps=NUM_DIFFUSION_STEPS, guidance_scale=GUIDANCE_SCALE, latent=latent, on_ipynb=False)
