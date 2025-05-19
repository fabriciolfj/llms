import os
import subprocess
import json
import requests
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
)
from datasets import load_dataset
import torch

# Configurações
BASE_MODEL = "meta-llama/Llama-2-7b-hf"  # Ou outro modelo compatível com Ollama
DATASET_PATH = "./data/jogos.txt"
OUTPUT_DIR = "./models/fine-tuned-model"
OLLAMA_MODEL_NAME = "jogos-model"
OLLAMA_API_BASE = "http://localhost:11434/api"


# 1. PARTE DE FINE-TUNING - Usando Hugging Face Transformers

def fine_tune_model():
    print("Iniciando processo de fine-tuning...")

    # Carregar dataset
    dataset = load_dataset("text", data_files=DATASET_PATH)
    print(f"Dataset carregado: {dataset}")

    # Filtrar amostras vazias
    dataset = dataset.filter(lambda example: len(example["text"]) > 10)

    # Carregar tokenizer e modelo
    print(f"Carregando modelo base: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float16)

    # Garantir que o tokenizer tenha um token de padding
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Função para tokenizar o dataset
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=512
        )

    # Tokenizar o dataset
    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    # Configurar data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # Não usamos masking para modelos causais como LLaMA
    )

    # Configurar parâmetros de treinamento
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,
        save_steps=500,
        save_total_limit=2,
        learning_rate=5e-5,
        weight_decay=0.01,
        fp16=True,  # Usar precisão mista para economizar memória
        logging_dir="./logs",
        logging_steps=100,
        report_to="none"
    )

    # Inicializar o Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        data_collator=data_collator,
    )

    # Executar o fine-tuning
    print("Iniciando treinamento...")
    trainer.train()

    # Salvar o modelo fine-tuned
    print(f"Salvando modelo fine-tuned em {OUTPUT_DIR}")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("Fine-tuning concluído com sucesso!")
    return OUTPUT_DIR


# 2. PARTE DE INTEGRAÇÃO COM OLLAMA

def convert_and_register_with_ollama(model_dir):
    print("Preparando modelo para Ollama...")

    # Criar Modelfile para o Ollama
    modelfile = f"""
FROM {BASE_MODEL.split('/')[-1]}

# Este é um modelo fine-tuned especializado em jogos
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
"""

    # Salvar o Modelfile
    with open("Modelfile", "w") as f:
        f.write(modelfile)

    # Registrar modelo no Ollama
    print(f"Registrando modelo '{OLLAMA_MODEL_NAME}' no Ollama...")

    # Para modelos grandes precisamos usar quantização
    # Nota: Esta parte pode variar dependendo do formato exato que o Ollama aceita
    # Pode ser necessário converter o modelo para GGUF primeiro

    try:
        # Tentativa de registrar usando o CLI do Ollama
        result = subprocess.run(
            ["ollama", "create", OLLAMA_MODEL_NAME, "-f", "Modelfile", "--from", model_dir],
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            print(f"Erro ao registrar modelo: {result.stderr}")
            return False

        print(f"Modelo '{OLLAMA_MODEL_NAME}' registrado com sucesso no Ollama!")
        return True

    except Exception as e:
        print(f"Erro ao registrar modelo: {e}")
        return False


# 3. TESTAR O MODELO VIA API OLLAMA

def test_ollama_model(prompt="Fale sobre os jogos que você conhece"):
    """Testa o modelo fine-tuned via API Ollama"""
    print(f"\nTestando modelo '{OLLAMA_MODEL_NAME}' com o prompt: '{prompt}'")

    api_url = f"{OLLAMA_API_BASE}/generate"

    payload = {
        "model": OLLAMA_MODEL_NAME,
        "prompt": prompt,
        "stream": False
    }

    try:
        response = requests.post(api_url, json=payload)
        if response.status_code == 200:
            result = response.json()
            return result.get("response", "Sem resposta")
        else:
            return f"Erro na API: {response.status_code}\n{response.text}"
    except Exception as e:
        return f"Erro ao chamar a API: {e}"


# Função principal
def main():
    # Passo 1: Fine-tuning
    model_dir = fine_tune_model()

    # Passo 2: Converter e registrar no Ollama
    success = convert_and_register_with_ollama(model_dir)

    if success:
        # Passo 3: Testar o modelo
        response = test_ollama_model()

        print("\nResposta do modelo fine-tuned:")
        print("-" * 50)
        print(response)
        print("-" * 50)

        print("\nPara usar seu modelo fine-tuned via API:")
        print(f"""
import requests

def gerar_resposta(prompt):
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={{
            "model": "{OLLAMA_MODEL_NAME}",
            "prompt": prompt,
            "stream": False
        }}
    )
    return response.json()["response"]

# Exemplo
resposta = gerar_resposta("Sua pergunta sobre jogos aqui")
print(resposta)
        """)
    else:
        print("Não foi possível registrar o modelo no Ollama.")


if __name__ == "__main__":
    main()