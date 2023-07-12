
from transformers import Seq2SeqTrainer

from utils.common import parse_args, load_data
from utils.preprocess_data import prepare_data
from models import get_model
from utils.data_collator import DynamicDataCollatorWithPadding


def main():

    model_args, data_args, training_args = parse_args()
    model_class = get_model(model_args)

    model, tokenizer = model_class.load_pretrained()
    dataset = load_data(data_args)

    trainer_data = prepare_data(dataset, data_args, tokenizer)
    data_collator = DynamicDataCollatorWithPadding(tokenizer)
    
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=data_collator,
        **trainer_data
    )

    train_result = trainer.train()
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()
    trainer.save_model()


if __name__ == "__main__":
    main()
