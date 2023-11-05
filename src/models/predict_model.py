import warnings
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
warnings.filterwarnings('ignore')

# selecting model checkpoint
model_checkpoint = "../../models/best"

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

# prefix for model input
prefix = "turn toxic to neutral"

# loading the model and run inference for it
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
model.eval()
model.config.use_cache = False


def neutralize(model, inference_request, tokenizer=tokenizer):
    input_ids = tokenizer(inference_request, return_tensors="pt").input_ids
    outputs = model.generate(input_ids=input_ids)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True, temperature=0))


print("Enter a sentence you want to neutralize:")
inp = input()
inp = inp.strip().lower()
inference_request = prefix + inp
neutralize(model, inference_request, tokenizer)
