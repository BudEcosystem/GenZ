from transformers import AutoModelForCausalLM, AutoTokenizer

class Causal:

    def __init__(self, model_args):
        self.model_name = model_args.model_name
        self.device_map = 'auto'
        self.trust_remote_code = model_args.trust_remote_code
        self.pad_token_id = model_args.pad_token_id

    def load_pretrained(self):
        
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=self.trust_remote_code)

        tokenizer.bos_token = tokenizer.eos_token if tokenizer.bos_token is None else tokenizer.bos_token
        tokenizer.pad_token_id = tokenizer.pad_token_id if self.pad_token_id is None else self.pad_token_id

        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map=self.device_map,
        )        
        
        return model, tokenizer
