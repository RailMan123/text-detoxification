import warnings

from datasets import load_from_disk, load_metric
import transformers
import numpy as np
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer

warnings.filterwarnings('ignore')

# selecting model checkpoint
model_checkpoint = "ceshine/t5-paraphrase-paws-msrp-opinosis"
transformers.set_seed(42)
raw_datasets = load_from_disk("../../data/interim")

# Load the BLUE metric
metric = load_metric("sacrebleu", split='train')

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

# prefix for model input
prefix = "turn toxic to neutral"

max_input_length = 128
max_target_length = 128
toxic_sent = "toxic_comment"
target_sent = "neutral_comment"


def preprocess_function(examples):
    inputs = [prefix + ex for ex in examples[toxic_sent]]
    targets = [ex for ex in examples[target_sent]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

    # Setup the tokenizer for targets
    labels = tokenizer(targets, max_length=max_target_length, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


train_size, val_size, test_size = 20000, 2000, 2000
cropped_datasets = raw_datasets
cropped_datasets['train'] = raw_datasets['train'].select(range(train_size))
cropped_datasets['validation'] = raw_datasets['validation'].select(range(val_size))
cropped_datasets['test'] = raw_datasets['test'].select(range(test_size))
tokenized_datasets = cropped_datasets.map(preprocess_function, batched=True)


# create a model for the pretrained model
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

# defining the parameters for training
batch_size = 32
model_name = model_checkpoint.split("/")[-1]

args = Seq2SeqTrainingArguments(
    f"{model_name}-finetuned-{toxic_sent}-to-{target_sent}",
    evaluation_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.00,
    save_total_limit=3,
    num_train_epochs=10,
    predict_with_generate=True,
    fp16=True,
    report_to='tensorboard',
)
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)


# simple postprocessing for text
def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels


# compute metrics function to pass to trainer
def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": result["score"]}

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result


trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)
trainer.train()
# saving model
trainer.save_model('../../models/best')
