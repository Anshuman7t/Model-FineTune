# Model-FineTune

# Fine-Tuning a Language Model Using Custom Data For Chat and Text Generation

Before starting, ensure you have the following installed:

* Python (version 3.8 or later)
* pandas, datasets, transformers, torch, accelerate, bitsandbytes and peft libraries
* Access to a GPU for faster training (recommended) (Also you can use Google Colab for all this)

# Data Preparation

Data should be in a structured format, such as a CSV or Excel file, containing questions and answers.

# Step 1: Load and Clean Data
#code:
import pandas as pd

file_path = '/content/path'
df = pd.read_excel(file_path)

cleaned_data = []

for index, row in df.iterrows():
    questions = row['Question'].split('\n')  #Split multiple questions in one cell
    answer = row['Answer'].strip()  #Get the corresponding answer

    for question in questions:
        question = question.strip()  #Clean up any extra whitespace
        if question:  #Ensure the question is not empty
            cleaned_data.append({'Question': question, 'Answer': answer})

#Convert the cleaned data to a DataFrame and save it
cleaned_df = pd.DataFrame(cleaned_data)
cleaned_df.to_csv('./cleaned_dataset.csv', index=False)

# Step 2: Format Data for Model Input

#for llama model:

with open('./formatted_output.txt', 'w', encoding='utf-8') as f:
    for index, row in cleaned_df.iterrows():
        formatted_text = f"<s>[INST] {row['Question']} [/INST] {row['Answer']} </s>\n"
        f.write(formatted_text)

#for gpt2 model:

with open('./formatted_output.txt', 'w', encoding='utf-8') as f:
    for index, row in cleaned_df.iterrows():
        question = row['Question'].strip()
        answer = row['Answer'].strip()
        formatted_text = f"Q: {question}\nA: {answer}\n"
        f.write(formatted_text)

# Load the Pre-trained Model and Tokenizer
model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path = "openai-community/gpt2",
                                                   quantization_config = BitsAndBytesConfig(load_in_4bit = True, bnb_4bit_compute_dtype = getattr(torch, "float16"), bnb_4bit_quant_type = "nf4"))
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path = "openai-community/gpt2", trust_remote_code = True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

We load the GPT-2 model and its tokenizer. We use 4-bit quantization to reduce the model's memory footprint, making it easier to fine-tune even on limited hardware.
Why use 4-bit quantization?
* Reduces memory usage, allowing for the fine-tuning of large models on smaller GPUs.

# Load the Dataset for Training

dataset = load_dataset('text', data_files='./formatted_output.txt')

# Define Training Arguments

training_arguments = TrainingArguments(output_dir = "./results", per_device_train_batch_size = 4, max_steps = 10000, save_steps=250, logging_steps=100, learning_rate=5e-5)

Here we specify the training parameters:
* Output Directory: Where the model checkpoints and logs are saved.
* Batch Size: The number of samples processed before the model's internal parameters are updated. We use 4 to balance between speed and memory usage.
* Max Steps: The maximum number of training steps. We use 10,000 to allow the model enough time to learn from the data.
* Save Steps: How often to save the model. We save every 250 steps to ensure we have frequent checkpoints.
* Logging Steps: How often to log the progress.
* Learning Rate: The rate at which the model learns. We use 5e-5, which is a standard value for fine-tuning tasks.

# Fine-Tune the Model with LoRA (Low-Rank Adaptation)

llama_sft_trainer = SFTTrainer(
    model=llama_model,
    args=training_arguments,
    train_dataset=dataset['train'],
    tokenizer=llama_tokenizer,
    peft_config=LoraConfig(task_type="CAUSAL_LM", r=8, lora_alpha=16, lora_dropout=0.1),
    dataset_text_field="text"
)
llama_sft_trainer.train()

LoRA (Low-Rank Adaptation), which allows us to train efficiently by modifying only a small part of the model:
* r (rank): We set it to 8, which balances between the complexity of the model's modifications and the memory usage. Lowering this reduces memory use but may impact performance.
* lora_alpha: Set to 16, this controls the scale of the updates. Lower values mean smaller updates and slower learning, but too high can lead to overfitting.
* lora_dropout: Set to 0.1 to prevent overfitting by randomly dropping some of the updates during training.
