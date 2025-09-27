import requests
import csv
import time
import os
import re
import json
import logging
import random
from datetime import datetime
from requests.exceptions import RequestException, ConnectTimeout, ReadTimeout, ConnectionError

# Configuração
BASE_URL = "https://api.github.com/repos"
KEYS_FILE = ".keys"  # Caminho para o arquivo de chaves
INPUT_FILE = "candidates_api.csv"  # CSV de entrada com id, repo_name, stars
DEPS_DIR = "dependencies"  # Diretório para arquivos de dependência
DEP_FILES = ["pyproject.toml", "requirements.txt"]  # Arquivos de dependência
MAX_RETRIES = 5  # Número máximo de tentativas para timeout de conexão
RETRY_DELAY = 10  # Segundos para esperar entre tentativas
CHECKPOINT_FILE = "checkpoint.json"  # Arquivo para salvar o progresso
LOG_FILE = "dependency_fetcher.log"  # Arquivo de log

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Carregar o token do GitHub do arquivo de chaves
def load_token():
    try:
        with open(KEYS_FILE, 'r', encoding='utf-8') as f:
            content = f.read()
            match = re.search(r'GITHUB_TOKEN\s*=\s*"(.+)"', content)
            if match:
                return match.group(1)
            raise ValueError("Token não encontrado no formato esperado em .keys")
    except FileNotFoundError:
        logger.critical(f"Arquivo de chaves {KEYS_FILE} não encontrado!")
        raise

# Criar diretório se não existir
def ensure_directories():
    for dir_path in [DEPS_DIR, os.path.dirname(LOG_FILE)]:
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)

# Carregar checkpoint para retomar o processamento de onde parou
def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        try:
            with open(CHECKPOINT_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError:
            logger.warning(f"Arquivo de checkpoint corrompido. Iniciando do zero.")
            return {"last_processed_idx": 0, "processed_projects": []}
    return {"last_processed_idx": 0, "processed_projects": []}

# Salvar checkpoint
def save_checkpoint(checkpoint_data):
    try:
        with open(CHECKPOINT_FILE, 'w', encoding='utf-8') as f:
            json.dump(checkpoint_data, f)
    except Exception as e:
        logger.error(f"Erro ao salvar checkpoint: {str(e)}")

# Função para exponential backoff
def exponential_backoff(attempt):
    # Adiciona alguma aleatoriedade (jitter) para evitar sincronização
    return RETRY_DELAY * (2 ** attempt) + random.uniform(0, 1)

# Verificar se um arquivo existe no repositório e retornar a URL de download com lógica de retry
def check_file_exists(owner, repo, filename, headers, retries=0):
    url = f"{BASE_URL}/{owner}/{repo}/contents/{filename}"
    try:
        response = requests.get(url, headers=headers, timeout=30)
        
        # Tratar rate limit
        if response.status_code == 403 and "rate limit" in response.text.lower():
            reset_time = int(response.headers.get("x-ratelimit-reset", time.time() + 300))
            wait_time = max(300, reset_time - int(time.time()) + 60)
            logger.warning(f"Rate limit excedido. Aguardando {wait_time // 60} minutos...")
            time.sleep(wait_time)
            return check_file_exists(owner, repo, filename, headers)  # Tentar novamente após esperar
        
        # Arquivo encontrado
        if response.status_code == 200:
            data = response.json()
            return data.get("download_url")
        # Arquivo não encontrado
        elif response.status_code == 404:
            return None
        # Outros erros
        else:
            logger.error(f"Erro ao verificar {filename} em {owner}/{repo}: {response.status_code} - {response.text}")
            if retries < MAX_RETRIES:
                wait_time = exponential_backoff(retries)
                logger.info(f"Tentando novamente em {wait_time:.2f}s... ({retries + 1}/{MAX_RETRIES})")
                time.sleep(wait_time)
                return check_file_exists(owner, repo, filename, headers, retries + 1)
            return None
            
    except (ConnectTimeout, ReadTimeout, ConnectionError) as e:
        if retries < MAX_RETRIES:
            wait_time = exponential_backoff(retries)
            logger.warning(f"Timeout ao conectar-se a {owner}/{repo}/{filename}. Tentando novamente ({retries + 1}/{MAX_RETRIES}) após {wait_time:.2f}s...")
            time.sleep(wait_time)
            return check_file_exists(owner, repo, filename, headers, retries + 1)
        else:
            logger.error(f"Máximo de tentativas excedido para {owner}/{repo}/{filename}. Pulando...")
            return None
    except RequestException as e:
        logger.error(f"Erro de requisição para {owner}/{repo}/{filename}: {str(e)}")
        if retries < MAX_RETRIES:
            wait_time = exponential_backoff(retries)
            logger.info(f"Tentando novamente em {wait_time:.2f}s... ({retries + 1}/{MAX_RETRIES})")
            time.sleep(wait_time)
            return check_file_exists(owner, repo, filename, headers, retries + 1)
        return None
    except Exception as e:
        logger.error(f"Erro inesperado ao verificar {owner}/{repo}/{filename}: {str(e)}")
        if retries < MAX_RETRIES:
            wait_time = exponential_backoff(retries)
            logger.info(f"Tentando novamente em {wait_time:.2f}s... ({retries + 1}/{MAX_RETRIES})")
            time.sleep(wait_time)
            return check_file_exists(owner, repo, filename, headers, retries + 1)
        return None

# Baixar conteúdo do arquivo de uma URL com lógica de retry
def download_file(url, output_path, retries=0):
    try:
        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            with open(output_path, 'wb') as f:
                f.write(response.content)
            logger.info(f"Download concluído: {output_path}")
            return True
        else:
            logger.error(f"Falha ao baixar {url}: {response.status_code}")
            if retries < MAX_RETRIES:
                wait_time = exponential_backoff(retries)
                logger.info(f"Tentando novamente em {wait_time:.2f}s... ({retries + 1}/{MAX_RETRIES})")
                time.sleep(wait_time)
                return download_file(url, output_path, retries + 1)
            return False
    except (ConnectTimeout, ReadTimeout, ConnectionError) as e:
        if retries < MAX_RETRIES:
            wait_time = exponential_backoff(retries)
            logger.warning(f"Timeout ao baixar {url}. Tentando novamente ({retries + 1}/{MAX_RETRIES}) após {wait_time:.2f}s...")
            time.sleep(wait_time)
            return download_file(url, output_path, retries + 1)
        else:
            logger.error(f"Máximo de tentativas excedido ao baixar {url}. Pulando...")
            return False
    except RequestException as e:
        logger.error(f"Erro de requisição ao baixar {url}: {str(e)}")
        if retries < MAX_RETRIES:
            wait_time = exponential_backoff(retries)
            logger.info(f"Tentando novamente em {wait_time:.2f}s... ({retries + 1}/{MAX_RETRIES})")
            time.sleep(wait_time)
            return download_file(url, output_path, retries + 1)
        return False
    except Exception as e:
        logger.error(f"Erro inesperado ao baixar {url}: {str(e)}")
        if retries < MAX_RETRIES:
            wait_time = exponential_backoff(retries)
            logger.info(f"Tentando novamente em {wait_time:.2f}s... ({retries + 1}/{MAX_RETRIES})")
            time.sleep(wait_time)
            return download_file(url, output_path, retries + 1)
        return False

def main():
    start_time = datetime.now()
    logger.info(f"Iniciando processo de coleta de dependências em {start_time}")

    # Garantir que os diretórios necessários existam
    ensure_directories()
    
    # Carregar token GitHub
    try:
        TOKEN = load_token()
        HEADERS = {"Authorization": f"token {TOKEN}", "Accept": "application/vnd.github.v3+json"}
    except Exception as e:
        logger.critical(f"Erro ao carregar o token: {str(e)}")
        return
    
    # Carregar checkpoint
    checkpoint = load_checkpoint()
    last_processed_idx = checkpoint["last_processed_idx"]
    processed_projects = set(checkpoint["processed_projects"])
    
    # Ler o CSV de candidatos
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)  # Pular cabeçalho
            projects = [(int(row[0]), row[1], int(row[2])) for row in reader]
    except FileNotFoundError:
        logger.critical(f"Arquivo de entrada {INPUT_FILE} não encontrado!")
        return
    except Exception as e:
        logger.critical(f"Erro ao ler o arquivo de entrada {INPUT_FILE}: {str(e)}")
        return

    total_projects = len(projects)
    logger.info(f"Carregados {total_projects} projetos de {INPUT_FILE}")
    logger.info(f"Retomando a partir do índice {last_processed_idx} ({len(processed_projects)} projetos já processados)")

    try:
        for idx, (project_id, repo_name, stars) in enumerate(projects, 1):
            # Pular projetos já processados
            if idx <= last_processed_idx or str(project_id) in processed_projects:
                continue
                
            try:
                owner, repo = repo_name.split('/', 1)
                logger.info(f"Processando {idx}/{total_projects}: {repo_name} (ID: {project_id})")

                # Verificar arquivos de dependência
                dep_url = None
                dep_filename = None
                for filename in DEP_FILES:
                    dep_url = check_file_exists(owner, repo, filename, HEADERS)
                    if dep_url:
                        dep_filename = filename
                        break

                # Se um arquivo de dependência existir, baixá-lo
                if dep_url:
                    logger.info(f"OK: Encontrado {dep_filename} para {repo_name}")
                    # Sanitizar repo_name para nome de arquivo
                    safe_repo_name = repo_name.replace('/', '-')
                    # Baixar arquivo de dependência
                    dep_path = os.path.join(DEPS_DIR, f"{project_id}_{safe_repo_name}_{dep_filename}")
                    success = download_file(dep_url, dep_path)
                    if success:
                        processed_projects.add(str(project_id))
                else:
                    logger.info(f"Pulando {repo_name}: Nenhum arquivo de dependência encontrado")
                    processed_projects.add(str(project_id))

                # Atualizar checkpoint e salvar
                checkpoint["last_processed_idx"] = idx
                checkpoint["processed_projects"] = list(processed_projects)
                save_checkpoint(checkpoint)

                # Respeitar rate limit (5.000/hora ≈ 1,2 solicitações/segundo)
                sleep_time = 2 + random.uniform(0, 1)  # Adicionar jitter
                time.sleep(sleep_time)
                
            except Exception as e:
                logger.error(f"Erro ao processar projeto {project_id} ({repo_name}): {str(e)}")
                # Continuar com o próximo projeto em vez de parar
                continue

        end_time = datetime.now()
        duration = end_time - start_time
        logger.info(f"Concluído o processamento de {total_projects} projetos em {duration}")
        
    except KeyboardInterrupt:
        logger.warning("Processo interrompido pelo usuário.")
        # Salvar checkpoint antes de sair
        checkpoint["last_processed_idx"] = idx - 1
        checkpoint["processed_projects"] = list(processed_projects)
        save_checkpoint(checkpoint)
        
    except Exception as e:
        logger.critical(f"Erro crítico no programa principal: {str(e)}")
        # Salvar checkpoint antes de sair
        checkpoint["last_processed_idx"] = idx - 1
        checkpoint["processed_projects"] = list(processed_projects)
        save_checkpoint(checkpoint)

if __name__ == "__main__":
    main()