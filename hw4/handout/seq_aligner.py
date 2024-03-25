import torch
import numpy as np


### Replacement Task
def get_word_inds(text: str, word_place: int, tokenizer):
    """
    Splits the text into words. If 'word_place' is a string, it finds all occurrences of the word in the text and stores their indices. 
    If 'word_place' is an integer, it wraps it in a list for consistent processing. 
    Encodes the text into tokens and decodes each token back into string form to identify the boundaries of each word in the tokenized version. 
    It iterates over these tokens, matching them to the specified word indices ('word_place') and collecting the corresponding token indices in the output list 'out'.
    """
    split_text = text.split(" ")
    if type(word_place) is str:
        word_place = [i for i, word in enumerate(split_text) if word_place == word]
    elif type(word_place) is int:
        word_place = [word_place]
    out = []
    if len(word_place) > 0:
        words_encode = [tokenizer.decode([item]).strip("#") for item in tokenizer.encode(text)][1:-1]
        cur_len, ptr = 0, 0
        for i in range(len(words_encode)):
            cur_len += len(words_encode[i])
            if ptr in word_place:
                out.append(i + 1)
            if cur_len >= len(split_text[ptr]):
                ptr += 1
                cur_len = 0
    return np.array(out)


def get_replacement_mapper_(x: str, y: str, tokenizer, max_len=77):
    """
    Splits both input strings x and y into words and constructs a mapping matrix of size max_len x max_len. 
    """
    words_x = x.split(' ')
    words_y = y.split(' ')   
    if len(words_x) != len(words_y):
        raise ValueError(f"attention replacement edit can only be applied on prompts with the same length"
                         f" but prompt A has {len(words_x)} words and prompt B has {len(words_y)} words.") 

    mapper = np.zeros((max_len, max_len))
    ####TODO####
    for i in range(len(words_x)):
        x_idxs = get_word_inds(x, i, tokenizer)
        y_idxs = get_word_inds(y, i, tokenizer)
        mesh_x, mesh_y = np.meshgrid(x_idxs, y_idxs)
        ratio = min(1/len(x_idxs), 1/len(y_idxs))
        mapper[mesh_x.flatten(), mesh_y.flatten()] = ratio

    # raise NotImplementedError()
    ####END_TODO####
    mapper = torch.from_numpy(mapper).float()
    return mapper


def get_replacement_mapper(prompts, tokenizer, max_len=77):
    """
    Returns stacked PyTorch tensor containing all the mapping matrices, where each matrix 
    corresponds to the mapping from the first prompt to one of the subsequent prompts.
    The max_len=77 because that is the maximum length of the CLIP text encoder.
    """
    x_seq = prompts[0]
    mappers = []
    for i in range(1, len(prompts)):
        mapper = get_replacement_mapper_(x_seq, prompts[i], tokenizer, max_len)
        mappers.append(mapper)
    return torch.stack(mappers)

if __name__ == '__main__':
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    model_id = "CompVis/ldm-text2im-large-256"
    NUM_DIFFUSION_STEPS = 50
    GUIDANCE_SCALE = 5.
    MAX_NUM_WORDS = 77
    # load model and scheduler
    import diffusers
    ldm = diffusers.DiffusionPipeline.from_pretrained(model_id).to(device)
    tokenizer = ldm.tokenizer
    get_replacement_mapper_('a big dog hat', 'a big cat hat', tokenizer)
    get_replacement_mapper_('a big doggyish hat', 'a big cat hat', tokenizer)
    get_replacement_mapper_('a big cat hat', 'a big doggyish hat', tokenizer)
    # print(get_replacement_mapper(prompts, tokenizer))