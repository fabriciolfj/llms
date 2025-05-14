import os
import subprocess
from datasets import load_dataset


def create_model_from_huggingface_dataset(base_model, dataset_path, custom_model_name):
    """
    Cria um modelo Ollama personalizado a partir de um dataset de texto Hugging Face.

    Args:
        base_model (str): Modelo base do Ollama (ex: llama2, mistral)
        dataset_path (str): Caminho para o arquivo de texto a ser carregado com load_dataset
        custom_model_name (str): Nome para o modelo personalizado
    """
    print(f"Carregando dataset de {dataset_path}...")

    try:
        # Carregar o dataset usando a função load_dataset
        dataset = load_dataset("text", data_files=dataset_path)

        # O dataset carregado geralmente tem uma estrutura como {'train': Dataset}
        # Vamos extrair o conteúdo de texto

        # Verificar a estrutura do dataset
        print(f"Estrutura do dataset: {dataset}")

        # Extrair textos do dataset
        text_content = ""

        # Verificar se o dataset tem a divisão 'train'
        if 'train' in dataset:
            # Concatenar todos os textos do dataset
            for item in dataset['train']:
                if 'text' in item:
                    text_content += item['text'] + "\n\n"

        # Verificar se conseguimos extrair algum texto
        if not text_content:
            print("Erro: Não foi possível extrair texto do dataset!")
            return False

        print(f"Texto extraído: {len(text_content)} caracteres")

        # Limitar o tamanho para evitar problemas (SYSTEM tem um limite de tamanho)
        max_chars = 8000  # Valor seguro, ajuste conforme necessário
        if len(text_content) > max_chars:
            print(f"Aviso: Texto truncado de {len(text_content)} para {max_chars} caracteres")
            text_content = text_content[:max_chars]

        # Escapar aspas para evitar problemas no SYSTEM
        text_content = text_content.replace('"', '\\"')

        # Criar o Modelfile
        modelfile = f"""FROM {base_model}
SYSTEM "Você é um assistente especializado em jogos com o seguinte conhecimento: {text_content}"
PARAMETER temperature 0.7
"""

        # Salvar o Modelfile
        with open("Modelfile", "w", encoding="utf-8") as f:
            f.write(modelfile)

        # Criar o modelo no Ollama
        print(f"Criando modelo '{custom_model_name}' baseado em '{base_model}'...")
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


def model(model_name, prompt):
    """Testa o modelo com um prompt via linha de comando"""
    print(f"\nTestando modelo '{model_name}' com o prompt: '{prompt}'")

    try:
        result = subprocess.run(
            ["ollama", "run", model_name, prompt],
            capture_output=True,
            text=True
        )

        print("\nResposta do modelo:")
        print("-" * 50)
        print(result.stdout)
        print("-" * 50)

    except Exception as e:
        print(f"Erro ao testar modelo: {e}")


# Uso principal
if __name__ == "__main__":
    import sys

    # Valores padrão
    base_model = "llama3"
    dataset_path = "./data/jogos.txt"
    model_name = "modelo-jogos"

    # Usar argumentos da linha de comando, se disponíveis
    if len(sys.argv) > 1:
        dataset_path = sys.argv[1]
    if len(sys.argv) > 2:
        model_name = sys.argv[2]
    if len(sys.argv) > 3:
        base_model = sys.argv[3]

    # Criar o modelo
    success = create_model_from_huggingface_dataset(base_model, dataset_path, model_name)

    if success:
        # Testar o modelo
        test_prompt = "Fale sobre os jogos que você conhece."
        model(model_name, test_prompt)

        print("\nPara usar seu modelo via linha de comando:")
        print(f"ollama run {model_name}")

        print("\nPara usar seu modelo via API Python:")
