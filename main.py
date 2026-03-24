
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import Dataset

# Dummy data for demonstration
data = [
    {"text": "Hello, how are you?"},
    {"text": "I am doing great, thanks!"},
    {"text": "What is your favorite color?"},
    {"text": "Blue is a beautiful color."},
    {"text": "The quick brown fox jumps over the lazy dog."},
    {"text": "Never underestimate the power of a good book."},
    {"text": "Artificial intelligence is transforming industries."},
    {"text": "Machine learning models require vast amounts of data."},
    {"text": "Natural Language Processing is a subfield of AI."},
    {"text": "Computer vision enables machines to see and interpret the world."},
    {"text": "Deep learning has achieved remarkable success in various domains."},
    {"text": "Reinforcement learning involves agents learning from interactions."},
    {"text": "Generative AI can create realistic images and text."},
    {"text": "The future of AI is exciting and full of possibilities."},
    {"text": "Data science combines statistics, computer science, and domain knowledge."},
    {"text": "Big data analytics helps uncover hidden patterns and insights."},
    {"text": "Cloud computing provides scalable and flexible infrastructure."},
    {"text": "Quantum computing promises to revolutionize computation."},
    {"text": "Cybersecurity is crucial for protecting digital assets."},
    {"text": "Blockchain technology offers decentralized and secure transactions."},
    {"text": "Edge computing brings computation closer to data sources."},
    {"text": "Robotics is advancing rapidly with AI integration."},
    {"text": "Virtual reality creates immersive digital experiences."},
    {"text": "Augmented reality overlays digital information onto the real world."},
    {"text": "The Internet of Things connects everyday devices to the internet."},
    {"text": "Smart cities leverage technology to improve urban living."},
    {"text": "Sustainable technology aims to minimize environmental impact."},
    {"text": "Biotechnology is revolutionizing medicine and agriculture."},
    {"text": "Space exploration continues to push the boundaries of human knowledge."},
    {"text": "Renewable energy sources are vital for a sustainable future."},
]

dataset = Dataset.from_list(data)

# 1. Load pre-trained model and tokenizer
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Add a padding token if the tokenizer doesn't have one
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})
    model.resize_token_embeddings(len(tokenizer))

# 2. Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=64)

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# 3. Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=1,
    per_device_train_batch_size=2,
    save_steps=10_000,
    save_total_limit=2,
    logging_dir="./logs",
    logging_steps=500,
    learning_rate=5e-5,
    weight_decay=0.01,
)

# 4. Create Trainer instance
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

# 5. Train the model
print("\nStarting fine-tuning...")
trainer.train()
print("Fine-tuning complete!")

# 6. Save the fine-tuned model
model.save_pretrained("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")
print("Model and tokenizer saved to ./fine_tuned_model")

# Example of how to use the fine-tuned model for generation
print("\nGenerating text with the fine-tuned model:")
input_text = "Artificial intelligence is"
input_ids = tokenizer.encode(input_text, return_tensors="pt")

output = model.generate(input_ids, max_length=50, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)
