import os
import requests
import json
from datasets import load_dataset

# URL base da API do Ollama
OLLAMA_API_BASE = "http://localhost:11434/api"


def create_model_from_huggingface_dataset(base_model, dataset_path, custom_model_name):
    """
    Cria um modelo Ollama personalizado a partir de um dataset de texto Hugging Face.
    """
    print(f"Carregando dataset de {dataset_path}...")

    try:
        # Carregar o dataset
        dataset = load_dataset("text", data_files=dataset_path)

        # Verificar a estrutura do dataset
        print(f"Estrutura do dataset: {dataset}")

        # Extrair textos do dataset
        text_content = ""

        if 'train' in dataset:
            for item in dataset['train']:
                if 'text' in item:
                    text_content += item['text'] + "\n\n"

        if not text_content:
            print("Erro: Não foi possível extrair texto do dataset!")
            return False

        print(f"Texto extraído: {len(text_content)} caracteres")

        # Limitar o tamanho
        max_chars = 8000
        if len(text_content) > max_chars:
            print(f"Aviso: Texto truncado de {len(text_content)} para {max_chars} caracteres")
            text_content = text_content[:max_chars]

        # Escapar aspas
        text_content = text_content.replace('"', '\\"')

        # Criar o Modelfile
        modelfile = f"""FROM {base_model}
SYSTEM "Você é um assistente especializado em jogos com o seguinte conhecimento: {text_content}"
PARAMETER temperature 0.7
"""

        # Salvar o Modelfile
        with open("Modelfile", "w", encoding="utf-8") as f:
            f.write(modelfile)

        # Criar o modelo usando a API do Ollama
        print(f"Criando modelo '{custom_model_name}' baseado em '{base_model}'...")

        # O Ollama não tem uma API direta para criar modelos com Modelfile
        # Vamos usar o CLI para isso como última opção
        import subprocess
        result = subprocess.run(["ollama", "create", custom_model_name, "-f", "Modelfile"],
                                capture_output=True, text=True)

        if result.returncode != 0:
            print(f"Erro ao criar modelo: {result.stderr}")
            return False

        print(f"Modelo '{custom_model_name}' criado com sucesso!")
        return True

    except Exception as e:
        print(f"Erro ao processar o dataset: {e}")
        return False


def call_ollama_api(model_name, prompt):
    """Chama a API do Ollama para gerar uma resposta"""

    api_url = f"{OLLAMA_API_BASE}/generate"

    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False
    }

    try:
        print(f"Chamando API Ollama em: {api_url}")
        print(f"Payload: {json.dumps(payload, indent=2)}")

        response = requests.post(api_url, json=payload)
        if response.status_code == 200:
            result = response.json()
            return result.get("response", "Sem resposta")
        else:
            return f"Erro na API: {response.status_code}\n{response.text}"
    except Exception as e:
        return f"Erro ao chamar a API: {e}"


def check_available_models():
    """Verifica quais modelos estão disponíveis no Ollama"""

    api_url = f"{OLLAMA_API_BASE}/tags"

    try:
        response = requests.get(api_url)
        if response.status_code == 200:
            result = response.json()
            models = result.get("models", [])
            return [model.get("name") for model in models]
        else:
            return f"Erro ao obter modelos: {response.status_code}\n{response.text}"
    except Exception as e:
        return f"Erro ao verificar modelos: {e}"


# Uso principal
def main():
    import sys

    # Valores padrão
    base_model = "llama2"
    dataset_path = "./data/jogos.txt"
    model_name = "modelo-jogos"

    # Usar argumentos da linha de comando, se disponíveis
    if len(sys.argv) > 1:
        dataset_path = sys.argv[1]
    if len(sys.argv) > 2:
        model_name = sys.argv[2]
    if len(sys.argv) > 3:
        base_model = sys.argv[3]

    # Verificar modelos disponíveis
    print("Modelos disponíveis no Ollama:")
    models = check_available_models()
    print(models)

    # Verificar se o modelo base existe
    if isinstance(models, list) and base_model not in models:
        print(f"Aviso: O modelo base '{base_model}' não parece estar disponível!")
        print(f"Modelos disponíveis: {', '.join(models)}")
        response = input("Deseja continuar mesmo assim? (s/n): ")
        if response.lower() != 's':
            print("Operação cancelada!")
            return

    # Criar o modelo
    success = create_model_from_huggingface_dataset(base_model, dataset_path, model_name)

    if success:
        # Testar o modelo usando a API
        test_prompt = "Fale sobre os jogos que você conhece."
        print(f"\nTestando modelo '{model_name}' com o prompt: '{test_prompt}'")

        response = call_ollama_api(model_name, test_prompt)

        print("\nResposta do modelo:")
        print("-" * 50)
        print(response)
        print("-" * 50)

        # Exemplo de uso
        print("\nPara usar a API do Ollama em Python:")
        print(f"""
import requests

def gerar_resposta(prompt, modelo="{model_name}"):
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={{
            "model": modelo,
            "prompt": prompt,
            "stream": False
        }}
    )
    return response.json()["response"]

# Exemplo de uso
resposta = gerar_resposta("Seu prompt aqui")
print(resposta)
        """)


if __name__ == "__main__":
    main()