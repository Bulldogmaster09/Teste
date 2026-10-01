# InvestorMe — Hidden Factor Model Prototype

Protótipo local do núcleo quantitativo do InvestorMe.

## O que já existe

- Feature engineering com dados point-in-time.
- Previsão de retorno para múltiplos horizontes.
- Similaridade entre estados históricos de empresas.
- Descoberta de fatores não lineares via Random Forest.
- Incerteza baseada em casos históricos semelhantes.
- Métricas de validação.
- Estrutura para uma futura camada de notícias/LM.

## O que ainda não está implementado

Este protótipo não é ainda o modelo de produção. Ainda faltam:

1. ingestão real de notícias e documentos regulatórios;
2. LM financeiro para extração de eventos;
3. embeddings e RAG;
4. banco temporal de eventos;
5. dados reais de mercado/fundamentos;
6. walk-forward completo com custos de transação;
7. calibração probabilística;
8. ensemble especializado por horizonte/setor/regime;
9. monitoramento de drift;
10. API e integração com a interface do InvestorMe.

## Executar

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python demo.py
```

No Windows:

```powershell
.venv\Scripts\activate
pip install -r requirements.txt
python demo.py
```

## Princípio de segurança estatística

Nenhuma informação posterior ao instante da previsão deve entrar nas features.
O treinamento de produção deve usar validação walk-forward e testes
out-of-sample. Fatores encontrados pelo sistema são hipóteses estatísticas,
não provas de causalidade.

## Próxima etapa

Conectar o `EventEngine` a um pipeline de notícias/LM e criar um dataset
point-in-time real. Só depois disso o predictor deve ser comparado contra
baselines simples e modelos de mercado tradicionais.
