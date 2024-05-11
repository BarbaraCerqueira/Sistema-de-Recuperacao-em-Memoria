# Sistema de Recuperação de Informação Baseado no Modelo Vetorial

**!! Atualizado (11/05/2024)**: Incluído processo de avaliação do sistema de recuperação de informação baseado no modelo vetorial, com e sem o uso de Porter Stemmer.

## Descrição
Este projeto implementa um sistema de recuperação de informação em memória usando o modelo vetorial. Ele é estruturado em módulos que processam consultas, geram listas invertidas, indexam dados e realizam buscas no índice criado. Também está incluso um avaliador que irá calcular diversar métricas e gráficos para avaliar a qualidade do sistema em duas situações: com e sem o uso de Porter Stemmer.

## Requisitos
- Python 3.x
- Biblioteca NLTK
- Outras bibliotecas Python listadas em `requirements.txt`

## Configuração Inicial
Para configurar o ambiente necessário para executar este sistema, siga os passos abaixo:

1. Clone o repositório:
   ```bash
   git clone https://github.com/BarbaraCerqueira/Sistema-de-Recuperacao-em-Memoria.git
   cd '.\Sistema-de-Recuperacao-em-Memoria\src'
   ```

2. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```

## Estrutura do Projeto
- `src/`: Contém o código fonte principal do sistema.
- `src/modules/`: Contém todos os módulos do sistema, incluindo processador de consultas, gerador de lista invertida, indexador e buscador.
- `src/config/`: Diretório com arquivos de configuração para cada módulo.
- `src/data/`: Armazena os arquivos de dados usados pelo sistema.
- `src/results/`: Diretório para armazenar os resultados das buscas e outros outputs gerados pelo sistema de recuperação de informação.
- `src/avalia/`: Diretório para armazenar os resultados da avaliação do sistema, como métricas e gráficos.
- `src/modules/utils`: Contém alguns utilitários, dentre eles arquivos de configuração de ambiente e uma função de preprocessamento de texto.

## Como Executar
Para rodar o sistema completo, execute o seguinte comando na pasta 'src' do projeto:

```bash
python main.py
```

## Arquivos de Configuração
Cada módulo do sistema utiliza um arquivo de configuração específico, localizado no diretório `src/config/`. Estes arquivos já estão ajustados com os caminhos dos arquivos de entrada e saída para cada módulo.

## Módulos
### Processador de Consultas
Este módulo implementa um processador de consultas, responsável por ler um arquivo XML contendo consultas e seus resultados esperados, normalizar os textos das consultas e salvar os resultados em arquivos CSV. Configurações em `pc.cfg`.

### Gerador de Lista Invertida
Este módulo implementa um gerador de listas invertidas simples, responsável por processar um conjunto de documentos em formato XML e construir uma lista invertida simples, que será escrita em um arquivo CSV. Configurações em `gli.cfg`.

### Indexador
Este módulo implementa um indexador, responsável por criar um modelo vetorial a partir de uma lista invertida simples usando a métrica tf-idf (Term Frequency-Inverse Document Frequency). O modelo vetorial é gerado na forma de um arquivo CSV. Configurações em `index.cfg`.

### Buscador
Este módulo implementa um buscador, responsável por obter os resultados de um conjunto de consultas a partir de um modelo vetorial, procurando os documentos mais relevantes a cada consulta. Configurações em `busca.cfg`.

### Avaliação
Esse módulo implementa um avaliador, responsável por avaliar o sistema de recuperação em memória baseado no modelo vetorial. Ela irá ler os arquivos de resultados gerados e analisá-los, gerando uma série de medidas e gráficos para medir a qualidade do modelo, os quais estão listados a seguir:
   1. Gráfico de 11 pontos de precisão e recall
   2. F1
   3. Precision@5
   4. Precision@10
   5. Histograma de R-Precision (comparativo)
   6. MAP
   7. MRR
   8. Discounted Cumulative Gain (médio)
   9. Normalized Discounted Cumulative Gain

## Utilitários

### Setup_nltk
Prepara as dependências do NLTK necessárias para o projeto. Nesse caso, faz o download do pacote de stopwords.

### Setup_logging
Configura a biblioteca logging.

### Preprocessing
Contém uma função de preprocessamento de texto que recebe um texto e remove acentos, stopwords, caracteres não-alfabéticos, aplica stemming nas palavras e transforma tudo para maiúsculo.

## Saída de Logs
O sistema gera logs detalhados para cada módulo, facilitando a depuração e o monitoramento de processos. Os logs incluem informações sobre início e fim de operações, erros, e tempos de processamento.
