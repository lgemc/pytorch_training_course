import torch

from transformers import GPT2LMHeadModel, GPT2Config

class FineTuneModel(GPT2LMHeadModel):
    def __init__(self,
                 config: GPT2Config,
                 model_name="gpt2",
                 vocab_size=50257,
                 device=None
                 ):
        super().__init__(config)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.model.resize_token_embeddings(vocab_size)
        self._device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self._device)