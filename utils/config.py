from dataclasses import dataclass, field
from typing import Optional

@dataclass
class ModelArguments:

    model_name: str = field(
        metadata={"help": "Path to pretrained model or model from huggingface"}
    )
    model_type: Optional[str] = field(
        default="causal", 
        metadata={"help": "Supported models: causal"}
    ) 
    trust_remote_code: Optional[bool] = field(
        default=False, 
        metadata={"help": "Allow loading remote code for tokenizer"}
    ) 
    pad_token_id: Optional[int] = field(
        default=None, 
        metadata={"help": "Allow adding pad token id, if not set in the tokenizer config"}
    ) 


@dataclass
class DataArguments:
    
    data_path: str = field(
        metadata={"help": "path of the json file or dataset from huggingface"}
    ) 
    split: Optional[str] = field(
        default="train", 
        metadata={"help": "path of the json file or dataset from huggingface"}
    ) 
    max_length: Optional[int] = field(
        default=512, 
        metadata={"help": "The cut off length for the tokenizer"}
    ) 
    prompt_template: Optional[str] = field(
        default="default", 
        metadata={"help": "Prompt template to be used for the model input"}
    ) 
    split_ratio: Optional[int] = field(
        default=0.05, 
        metadata={"help": "Test/Validation split ratio"}
    ) 
    prompt_column: Optional[str] = field(
        default="prompt", 
        metadata={"help": "Column name for prompt column"}
    )
    response_column: Optional[str] = field(
        default="response", 
        metadata={"help": "Column name for response column"}
    )
    history_column: Optional[str] = field(
        default="history", 
        metadata={"help": "Column name for history column"}
    ) 



