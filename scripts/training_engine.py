from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import get_chat_template
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
import torch

#downloading a pre-trained Llama-3.1 model
#4-bit version to mthematically reduce/compress vram usage
# round off the long complex decimal numbers in the neural network into smaller and simpler numbers (4-bit)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit", #using unsloth to fetch the model, neural network, we will be using. As well as fetching the tokeniser, the translator/dictonary from human words.
    load_in_4bit = True, #execution of quantization
)

#low-rank adaptation 
model = FastLanguageModel.get_peft_model(  #Parameter-Efficient Fine-Tuning, take base model we just loaded, freeze it, and use it in the LoRA setup
    model, #passeing frozen Llama-3.1 model into the PEFT function
    r = 16, #rank
    # apply LoRA to all linear transformation layers in the transformer architecture to maximize the model's ability to learn complex driving tasks.
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 16, #  multiplier to determin how strongly the new driving rules override the original brain
    lora_dropout = 0,#disables random connection dropping. required to be 0 for maximum math speed and memory efficiency
    bias = "none", #telling math engine to only train the "weights" and ignore the "biases", y=mx+b (ignore the b)
    use_gradient_checkpointing = "unsloth", # used to store only mathematical  checkpoints instead of  whole equation to fit modell on a single GPU
    random_state = 42, #random seed, ensures that if we run script twice, the initial mathematical starting points are the same
)

# set dictonary/tranlator rules to work for llama
#llama doesnt read json, instead has own rules to separate text
tokenizer = get_chat_template(tokenizer, chat_template = "llama-3.1")

# function to format dataset by applying the llama rules to each json message with a loop
def formatting_prompts_func(examples):
    convos = examples["messages"]
    texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False) for convo in convos]
    return { "text" : texts }

# get and apply the dataset to the formatting function
training_dataset_path = "data/captures/training/training_dataset.jsonl"
training_dataset = load_dataset("json", data_files=training_dataset_path, split="train")
training_dataset = training_dataset.map(formatting_prompts_func, batched = True)

trainer = SFTTrainer( #start the Hugging Face training engine  built for instruction-following models
    model = model, #passes the loaded llm
    tokenizer = tokenizer, #passes the loaded translator/dictonary
    train_dataset = training_dataset, #pass the formatted training dataset
    dataset_text_field = "text", #tells the trainer to look for column named "text" in dataset
    max_seq_length = 2048, #maximum number of tokens, words/numbers the model is allowed to read in a single example
    dataset_num_proc = 2, #make it use 2 CPU cores to prepare the text data, making the pipeline run slightly faster
    packing = False, #packing set to false so every driving frame strictly isolated so the model doesn't confuse between different steps 
    args = TrainingArguments(  #configuration block for the training math
        per_device_train_batch_size = 2, #number of examples the GPU processes at the exact same time
        gradient_accumulation_steps = 4, #tells AI to process 4 batches in a row ,storing the math in short-term memory, before updating weights
        warmup_steps = 5, #first 5 steps, the AI will learn very slowly

        #max_steps = 60, # Hard Stop! for a quick 2-minute test run
        num_train_epochs = 3, #x full runs of the dataset, for actual  training run
        
        learning_rate = 2e-4, #how much of change to make when amistake is made
        fp16 = not is_bfloat16_supported(), 
        bf16 = is_bfloat16_supported(), #detect what gpu cluster is being used so to decide what learning algorithm to use
        logging_steps = 1,
        optim = "adamw_8bit", #mathematical algorithm used to update the weights. Using 8-bit compressed version of the AdamW optimizer to save even more GPU memory
        weight_decay = 0.01, #penalizes the AI if its math numbers get too big, preventing it from "overfitting"
        lr_scheduler_type = "linear", #gradually reducing the learning_rate in a straight line as the training nears the end
        seed = 42,
        output_dir = "outputs", #folder where the script will save backup checkpoints of the brain in case the cluster loses power
    ),
)

#runniing the trainer
print("Training...")
trainer_stats = trainer.train()

print("Exporting to a .gguf file...")
#model.save_pretrained_gguf("carla_driver_model", tokenizer, quantization_method = "q4_k_m")


model.save_pretrained("carla_driver_adapter")
tokenizer.save_pretrained("carla_driver_adapter")
print("File saved. ")