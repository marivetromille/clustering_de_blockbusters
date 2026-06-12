#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 31 16:13:23 2026
@author: marianaalcantaravetromille
TCC: Clustering não supervisionado de blockbusters com análise de Sazonalidade
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import zscore
from sklearn.cluster import KMeans
import pingouin as pg
import plotly.express as px
import prince
from scipy.stats import chi2_contingency
from sklearn.metrics import silhouette_score

# =============================================================================
# 1. IMPORTAÇÃO E TRATAMENTO DO BANCO DE DADOS
# =============================================================================

# Carregando o arquivo Excel conforme estrutura descrita
filmes = pd.read_excel('dados_dos_blockbusters.xlsx')

# Exibindo informações iniciais
print("--- Estrutura Inicial do Banco de Dados ---")
filmes.info()

# Remoção de valores nulos/faltantes (Listwise Deletion conforme metodologia)
filmes.dropna(inplace=True)

# Engenharia de Recursos: Operacionalização da Sazonalidade (Einav, 2007)
# Meses de alta densidade de mercado: Maio, Junho, Julho e Dezembro
meses_alta = ['Maio', 'Junho', 'Julho', 'Dezembro']
filmes['sazonalidade'] = np.where(filmes['mes_de_lancamento'].isin(meses_alta), 1, 0)

# =============================================================================
# 2. CLUSTERIZAÇÃO NAS VARIÁVEIS QUANTITATIVAS
# =============================================================================

# Separando as variáveis numéricas contínuas centrais
df_quanti = filmes[['orcamento_de_producao', 'roi', 'faturamento_bruto_mundial']].copy()

print("\n--- Estatísticas Descritivas Básicas ---")
print(df_quanti.describe())

# Padronização por meio do Z-Score com Correção de Bessel (ddof=1)
df_quanti_pad = df_quanti.apply(zscore, ddof=1)

# Identificação da quantidade de clusters (Método Elbow)
elbow = []
K = range(1, 11) 
for k in K:
    kmeanElbow = KMeans(n_clusters=k, init='random', random_state=100, n_init=10).fit(df_quanti_pad)
    elbow.append(kmeanElbow.inertia_)
    
plt.figure(figsize=(12, 6))
plt.plot(K, elbow, marker='o', color='blue')
plt.xlabel('Nº Clusters', fontsize=12)
plt.xticks(range(1, 11))
plt.ylabel('WCSS (Inércia)', fontsize=12)
plt.title('Método de Elbow - Estruturas Financeiras de Blockbusters', fontsize=14)
plt.grid(True, linestyle='--')
plt.show()

# Identificação da quantidade de clusters (Método da Silhueta)
silhueta = []
I = range(2, 11) 
for i in I:    
    kmeansSil = KMeans(n_clusters=i, init='random', random_state=100, n_init=10).fit(df_quanti_pad)
    silhueta.append(silhouette_score(df_quanti_pad, kmeansSil.labels_))

plt.figure(figsize=(12, 6))
plt.plot(range(2, 11), silhueta, color='purple', marker='o')
plt.xlabel('Nº Clusters', fontsize=12)
plt.ylabel('Silhueta Média', fontsize=12)
plt.title('Método da Silhueta - Validação dos Grupos de Filmes', fontsize=14)
plt.grid(True, linestyle='--')
plt.show()

# Definição e treinamento do modelo final (K=3 conforme retorno do terminal)
kmeans_final = KMeans(n_clusters=3, init='random', random_state=100, n_init=10).fit(df_quanti_pad)
kmeans_clusters = kmeans_final.labels_

# Atribuindo os rótulos de clusters de volta aos DataFrames
filmes['Cluster'] = kmeans_clusters
df_quanti_pad['Cluster'] = kmeans_clusters

# Convertendo para categórica para análises estatísticas
filmes['Cluster'] = filmes['Cluster'].astype('category')
df_quanti_pad['Cluster'] = df_quanti_pad['Cluster'].astype('category')

# =============================================================================
# 3. VALIDAÇÃO DE SIGNIFICÂNCIA POR ANOVA (Fávero & Belfiore, 2017)
# =============================================================================
# Correção: O parâmetro data deve apontar para o DataFrame 'filmes' para avaliar as escalas reais
print("\n--- ANOVA: Orçamento de Produção por Cluster ---")
print(pg.anova(dv='orcamento_de_producao', between='Cluster', data=filmes, detailed=True).T)

print("\n--- ANOVA: ROI por Cluster ---")
print(pg.anova(dv='roi', between='Cluster', data=filmes, detailed=True).T)

print("\n--- ANOVA: Faturamento de Bilheteria Mundial por Cluster ---")
print(pg.anova(dv='faturamento_bruto_mundial', between='Cluster', data=filmes, detailed=True).T)

# Análise Descritiva do "DNA" dos Clusters
print("\n--- Perfil Financeiro Médio dos Clusters ---")
print(filmes.groupby(by=['Cluster'], observed=False)[['orcamento_de_producao', 'roi', 'faturamento_bruto_mundial']].mean())

print("\n--- Distribuição de Estrelas, Extensão de Marca e Sazonalidade por Cluster ---")
print(filmes.groupby(by=['Cluster'], observed=False)[['presenca_de_estrelas', 'extensao_de_marca', 'sazonalidade', 'proporcao_de_novatos_no_elenco_principal']].mean())

# =============================================================================
# 4. ANÁLISE DE CORRESPONDÊNCIA MÚLTIPLA (ACM / MCA)
# =============================================================================

# Construção do DataFrame Qualitativo com rótulos textuais legíveis
df_quali = pd.DataFrame()
df_quali['Cluster'] = 'Cluster_' + filmes['Cluster'].astype(str)
df_quali['Star_Power'] = np.where(filmes['presenca_de_estrelas'] == 1, 'Com_Estrela', 'Sem_Estrela')
df_quali['Franquia'] = np.where(filmes['extensao_de_marca'] == 1, 'Franquia_Sim', 'Franquia_Nao')
df_quali['Sazonalidade'] = np.where(filmes['sazonalidade'] == 1, 'Alta_Temporada', 'Temporada_Regular')

# Mapeamento dos gêneros principais (conforme grafia exata identificada no Pandas)
df_quali['Acao'] = np.where(filmes['genero_acao'] == 1, 'Ação_Sim', 'Ação_Nao')
df_quali['Aventura'] = np.where(filmes['genero_aventura'] == 1, 'Aventura_Sim', 'Aventura_Nao')
df_quali['SciFi'] = np.where(filmes['genero_sci_fi'] == 1, 'SciFi_Sim', 'SciFi_Nao')

# Testes Qui-Quadrado de Associação (Referência de validação para a Banca)
print("\n--- Testes de Associação Qui-Quadrado (P-Valor) ---")
for col in ['Star_Power', 'Franquia', 'Sazonalidade', 'Acao', 'Aventura', 'SciFi']:    
    tabela = chi2_contingency(pd.crosstab(df_quali["Cluster"], df_quali[col]))    
    print(f"Associação Cluster vs {col} -> p-valor: {round(tabela[1], 4)}")

# Modelagem da Correspondência Múltipla (ACM) - Extraindo 3 dimensões latentes
mca = prince.MCA(n_components=3, random_state=100).fit(df_quali)

# Extração e preparação das coordenadas dos componentes (Coordenadas Principais)
coord_padrao = mca.column_coordinates(df_quali)

# Organização da estrutura para plotagem visual
chart = coord_padrao.reset_index()
var_chart = pd.Series(chart['index'].str.split('_', expand=True).iloc[:, 0])

chart_df_mca = pd.DataFrame({    
    'categoria': chart['index'],    
    'Dimensao_1': chart[0],    
    'Dimensao_2': chart[1],    
    'Dimensao_3': chart[2],    
    'Variavel_Origem': var_chart
})

# Construindo o gráfico tridimensional iterativo com Plotly
fig = px.scatter_3d(    
    chart_df_mca,     
    x='Dimensao_1',     
    y='Dimensao_2',     
    z='Dimensao_3',     
    color='Variavel_Origem',     
    text='categoria',     
    title='Mapa Perceptual Interativo: Segmentação de Blockbusters (K-means + ACM)'
)

# Customizando tamanho dos marcadores textuais e salvando em arquivo local
fig.update_traces(textposition='top center', marker=dict(size=5))
fig.write_html('mapa_perceptual_blockbusters.html')

print("\n[Sucesso] Mapa perceptual tridimensional exportado com sucesso para o arquivo 'mapa_perceptual_blockbusters.html'.")
print("Abra-o no seu navegador para explorar os agrupamentos, incluindo os vetores de Sazonalidade!")
# =============================================================================
# Fim do Script!
# =============================================================================