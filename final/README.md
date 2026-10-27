# EnergIA — Monitoramento e desagregação de cargas

Projeto de monitoramento não invasivo de consumo de energia elétrica (NILM), desenvolvido como solução acadêmica para identificar o uso individual de geladeira, máquina de lavar roupas e televisão a partir da potência elétrica agregada.

## Solução final

Durante o desenvolvimento, estava prevista a recepção das leituras por meio de um broker MQTT conectado ao sensor SCT-013-30. Como não foi possível concluir a integração com o broker, foi utilizada uma alternativa baseada em arquivos CSV.

O fluxo final é:

```text
SCT-013-30 ou arquivo simulado
              ↓
       Time, Unix, Aggregate
              ↓
     Características temporais
              ↓
       Modelos Random Forest
              ↓
Estado, consumo e custo por aparelho
              ↓
          Interface Streamlit
```

O arquivo de entrada do sensor contém somente:

- `Time`: data e horário da leitura;
- `Unix`: timestamp Unix;
- `Aggregate`: potência elétrica agregada em watts.

As cargas individuais são estimadas pelo modelo e não fazem parte do arquivo do sensor.

## Equipamentos identificados

| Coluna da base | Equipamento |
|---|---|
| `Appliance1` | Geladeira |
| `Appliance5` | Máquina de lavar roupas |
| `Appliance8` | Televisão |

## Estrutura do projeto

```text
PI4/
├── random_forest_nilm.py
├── model/
│   ├── gerar_dados_sct013_30.py
│   └── simulador_streamlit_sct013_cpu.py
├── model/CLEAN_House1.csv       # base de treinamento, não obrigatória no GitHub
├── resultados_cpu/               # modelos e métricas gerados localmente
├── requirements.txt
└── README.md
```

## Requisitos

- Python 3.10 ou superior;
- 16 GB de RAM recomendados;
- sistema operacional Windows, Linux ou macOS;
- acesso ao arquivo `CLEAN_House1.csv` para treinamento.

Instale as dependências:

```bash
pip install -r requirements.txt
```

Caso o arquivo `requirements.txt` ainda não exista, instale diretamente:

```bash
pip install numpy pandas scikit-learn joblib streamlit
```

## Base de dados

O treinamento utiliza a base `CLEAN_House1.csv`, pertencente ao conjunto REFIT. Essa base representa uma residência do Reino Unido e contém o consumo agregado e as cargas individuais dos equipamentos.

O arquivo possui grande volume de dados e não deve ser enviado ao GitHub convencional. Recomenda-se utilizar Git LFS, armazenamento externo ou disponibilizar instruções para obtenção da base.

## 1. Gerar leituras simuladas do SCT-013-30

O gerador cria 43.200 leituras, uma a cada dois segundos durante 24 horas.

```bash
python model/gerar_dados_sct013_30.py \
  --output model/sct013_30_simulado_1dia.csv \
  --start "2026-10-27 00:00:00"
```

O arquivo gerado contém apenas:

```text
Time,Unix,Aggregate
```

## 2. Treinar o Random Forest

O treinamento utiliza as colunas individuais da base `CLEAN_House1.csv` como rótulos. O modelo aprende a classificar se cada equipamento está ligado ou desligado.

Para um computador com 16 GB de RAM:

```bash
python random_forest_nilm.py \
  --input model/CLEAN_House1.csv \
  --output-dir resultados_cpu \
  --max-rows 1500000 \
  --max-train-samples 100000 \
  --max-eval-samples 100000 \
  --trees 100 \
  --n-jobs 4
```

Para utilizar toda a base:

```bash
python random_forest_nilm.py \
  --input model/CLEAN_House1.csv \
  --output-dir resultados_cpu \
  --max-rows 0 \
  --max-train-samples 100000 \
  --max-eval-samples 200000 \
  --trees 100 \
  --n-jobs 4
```

O treinamento gera:

```text
resultados_cpu/random_forest_models.joblib
resultados_cpu/metricas_random_forest.csv
resultados_cpu/metricas_random_forest.json
```

## 3. Executar a interface Streamlit

Depois de gerar o CSV simulado e treinar os modelos:

```bash
streamlit run model/simulador_streamlit_sct013_cpu.py
```

A aplicação carrega automaticamente:

```text
model/sct013_30_simulado_1dia.csv
resultados_cpu/random_forest_models.joblib
resultados_cpu/metricas_random_forest.json
```

Na interface é possível:

- reproduzir as leituras a cada dois segundos;
- visualizar a potência agregada;
- acompanhar o estado estimado dos equipamentos;
- consultar a confiança das previsões;
- visualizar o consumo individual acumulado em kWh;
- consultar o valor consumido por aparelho;
- visualizar o custo total acumulado;
- consultar a projeção mensal da fatura;
- alterar a tarifa de energia.

## Cálculo do consumo e custo

Para cada leitura, o consumo individual é estimado por:

```text
Consumo (kWh) = Potência (W) × intervalo (s) / 3.600.000
```

O custo é calculado por:

```text
Custo (R$) = Consumo (kWh) × Tarifa (R$/kWh)
```

Como o Random Forest realiza a classificação do estado do equipamento, a aplicação utiliza potências nominais estimadas para calcular o consumo:

- Geladeira: 85 W;
- Máquina de lavar roupas: 1.350 W;
- TV: 65 W.

## Limitações e adaptação para o Brasil

Os dados `CLEAN_House1.csv` representam o comportamento de uma residência do Reino Unido. Portanto, eles não representam necessariamente o padrão de consumo brasileiro.

Para utilizar a solução em residências brasileiras, recomenda-se:

1. coletar dados reais com o SCT-013-30 em residências brasileiras;
2. considerar tensão nominal e frequência de 60 Hz;
3. ajustar as potências nominais dos equipamentos;
4. utilizar tarifas e bandeiras tarifárias da distribuidora local;
5. considerar hábitos de uso e horários de consumo brasileiros;
6. retreinar o modelo com dados locais;
7. validar o modelo em residências diferentes das utilizadas no treinamento.

Os resultados atuais devem ser interpretados como estimativas experimentais. A aplicação não substitui um medidor homologado de energia elétrica.

## MQTT e evolução futura

A solução atual utiliza CSV como substituição temporária ao broker MQTT. Em uma evolução futura, o módulo que lê o arquivo poderá ser substituído por um consumidor MQTT, mantendo o restante da arquitetura:

```text
Mensagem MQTT → pré-processamento → Random Forest → Streamlit
```

Essa substituição permitirá receber dados diretamente do ESP32 e processar as leituras em tempo real.

## Arquivos experimentais

O repositório pode conter versões exploratórias de modelos GRU, TCN e outras abordagens de deep learning. Esses arquivos são úteis para comparação acadêmica, mas a solução final para execução em CPU utiliza o Random Forest:

```text
random_forest_nilm.py
model/gerar_dados_sct013_30.py
model/simulador_streamlit_sct013_cpu.py
```
