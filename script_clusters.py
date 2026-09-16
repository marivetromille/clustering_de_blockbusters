#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 31 16:13:23 2026
@author: marianaalcantaravetromille
Clustering não supervisionado de blockbusters: fórmulas de sucesso comercial no cinema
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import zscore, chi2_contingency, kruskal, shapiro
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import pingouin as pg
import plotly.express as px
import prince

# Todos os 19 gêneros cinematográficos coletados na base (Tabela 1), mapeados para o
# rótulo utilizado nas colunas qualitativas e nos testes Qui-Quadrado/ACM
generos_avaliados = {
    'genero_acao': 'Acao',
    'genero_aventura': 'Aventura',
    'genero_sci_fi': 'SciFi',
    'genero_fantasia': 'Fantasia',
    'genero_animacao': 'Animacao',
    'genero_comedia': 'Comedia',
    'genero_drama': 'Drama',
    'genero_romance': 'Romance',
    'genero_crime': 'Crime',
    'genero_misterio': 'Misterio',
    'genero_familia': 'Familia',
    'genero_musical': 'Musical',
    'genero_suspense': 'Suspense',
    'genero_biografia': 'Biografia',
    'genero_historia': 'Historia',
    'genero_guerra': 'Guerra',
    'genero_horror': 'Horror',
    'genero_esporte': 'Esporte',
    'genero_faroeste': 'Faroeste',
}

# Variáveis qualitativas nominais submetidas aos testes Qui-Quadrado e à ACM
variaveis_qualitativas = ['Star_Power', 'Franquia', 'Sazonalidade'] + list(generos_avaliados.values())

# =============================================================================
# 1. IMPORTAÇÃO E TRATAMENTO DO BANCO DE DADOS
# =============================================================================

# Carregando o arquivo Excel conforme estrutura descrita
try:
    filmes = pd.read_excel('dados_dos_blockbusters.xlsx')
except FileNotFoundError:
    raise FileNotFoundError(
        "Arquivo 'dados_dos_blockbusters.xlsx' não encontrado no diretório atual. "
        "Verifique se o script está sendo executado na mesma pasta do arquivo de dados."
    )

# Exibindo informações iniciais
print("--- Estrutura Inicial do Banco de Dados ---")
filmes.info()

# Colunas efetivamente utilizadas ao longo da análise
colunas_analise = [
    'orcamento_de_producao', 'roi', 'faturamento_bruto_mundial', 'mes_de_lancamento',
    'presenca_de_estrelas', 'extensao_de_marca', 'proporcao_de_novatos_no_elenco_principal',
] + list(generos_avaliados.keys())

colunas_faltantes = [c for c in colunas_analise if c not in filmes.columns]
if colunas_faltantes:
    raise KeyError(f"Colunas esperadas ausentes no banco de dados: {colunas_faltantes}")

# Remoção de valores nulos/faltantes (Listwise Deletion conforme metodologia)
# Restrita às colunas de análise, para não descartar linhas por nulos em colunas não utilizadas
n_antes = len(filmes)
filmes.dropna(subset=colunas_analise, inplace=True)
# Reindexação após o descarte de linhas: evita desalinhamento por índice ao
# atribuir Series construídas separadamente (ex.: rótulos de cluster do sklearn,
# que sempre retornam com índice 0..N-1) de volta ao DataFrame 'filmes'.
filmes.reset_index(drop=True, inplace=True)
n_depois = len(filmes)
print(f"\n[Tratamento] {n_antes - n_depois} linha(s) removida(s) por valores ausentes "
      f"nas colunas de análise ({n_depois} de {n_antes} linhas mantidas).")

# Normalização de grafia (espaços/caixa) antes de comparar com a lista de meses
filmes['mes_de_lancamento'] = filmes['mes_de_lancamento'].astype(str).str.strip().str.capitalize()

# Engenharia de Recursos: Operacionalização da Sazonalidade (Einav, 2007)
# Meses de alta densidade de mercado: Maio, Junho, Julho e Dezembro
meses_alta = ['Maio', 'Junho', 'Julho', 'Dezembro']
filmes['sazonalidade'] = np.where(filmes['mes_de_lancamento'].isin(meses_alta), 1, 0)

# =============================================================================
# 2. CLUSTERIZAÇÃO NAS VARIÁVEIS QUANTITATIVAS
# =============================================================================

# Separando as variáveis numéricas contínuas centrais
df_quanti = filmes[['orcamento_de_producao', 'faturamento_bruto_mundial']].copy()

print("\n--- Estatísticas Descritivas Básicas ---")
print(df_quanti.describe())

# Variância zero impediria a padronização (divisão por zero -> NaN/Inf no KMeans)
colunas_constantes = df_quanti.std(ddof=0)
colunas_constantes = colunas_constantes[colunas_constantes == 0].index.tolist()
if colunas_constantes:
    raise ValueError(f"Variável(is) com variância zero, impossível padronizar via Z-Score: {colunas_constantes}")

# Padronização por meio do Z-Score com divisor populacional (ddof=0)
df_quanti_pad = df_quanti.apply(zscore, ddof=0)

# Diagnóstico de outliers (|z| > 3): o k-means é sensível a distâncias euclidianas,
# de modo que valores extremos merecem verificação, ainda que não sejam removidos aqui
# por representarem casos legítimos de blockbusters de bilheteria/orçamento excepcionais
for coluna in df_quanti_pad.columns:
    n_outliers = (df_quanti_pad[coluna].abs() > 3).sum()
    print(f"[Diagnóstico] {coluna}: {n_outliers} outlier(s) com |z| > 3 "
          f"(assimetria/skewness = {df_quanti[coluna].skew():.3f})")


# Formatação de gráficos conforme a Tabela 8 do Manual de TCC (sem grade, sem borda,
# sem título interno, eixos principais em linha sólida preta de 1,5 pt)
def formatar_grafico_conforme_manual(ax):
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for lado in ['left', 'bottom']:
        ax.spines[lado].set_color('black')
        ax.spines[lado].set_linewidth(1.5)
    ax.tick_params(colors='black')
    for eixo in [ax.xaxis, ax.yaxis]:
        eixo.label.set_fontname('Arial')
        eixo.label.set_fontsize(11)
        eixo.label.set_color('black')

# Identificação da quantidade de clusters (Método Elbow)
elbow = []
k_range = range(1, 11)
for k in k_range:
    kmeanElbow = KMeans(n_clusters=k, init='k-means++', random_state=100, n_init=10).fit(df_quanti_pad)
    elbow.append(kmeanElbow.inertia_)

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(k_range, elbow, marker='o', color='blue')
ax.set_xlabel('Nº Clusters')
ax.set_xticks(range(1, 11))
ax.set_ylabel('WCSS (Inércia)')
formatar_grafico_conforme_manual(ax)
plt.savefig('elbow_blockbusters.png', dpi=300, bbox_inches='tight')
plt.show()

# Identificação da quantidade de clusters (Método da Silhueta)
silhueta = []
i_range = range(2, 11)
for i in i_range:
    kmeansSil = KMeans(n_clusters=i, init='k-means++', random_state=100, n_init=10).fit(df_quanti_pad)
    silhueta.append(silhouette_score(df_quanti_pad, kmeansSil.labels_))

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(i_range, silhueta, color='purple', marker='o')
ax.set_xlabel('Nº Clusters')
ax.set_ylabel('Silhueta Média')
formatar_grafico_conforme_manual(ax)
plt.savefig('silhueta_blockbusters.png', dpi=300, bbox_inches='tight')
plt.show()

# =============================================================================
# VALIDAÇÃO COMPLEMENTAR: JUSTIFICATIVA DA ESCOLHA DE K=3
# =============================================================================

for k_candidato in [2, 3, 4]:
    kmeans_candidato = KMeans(n_clusters=k_candidato, init='k-means++', random_state=100, n_init=10).fit(df_quanti_pad)
    silhueta_candidato = silhueta[list(i_range).index(k_candidato)]
    tamanhos_candidato = pd.Series(kmeans_candidato.labels_).value_counts().sort_index().tolist()
    print(f"[Validação K] K={k_candidato}: silhueta = {silhueta_candidato:.4f}, "
          f"tamanhos dos clusters = {tamanhos_candidato}")

# =============================================================================
# VALIDAÇÃO COMPLEMENTAR: ANOVA E QUI-QUADRADO PARA K=2, K=3 E K=4
# =============================================================================

df_quali_base = pd.DataFrame()
df_quali_base['Star_Power'] = np.where(filmes['presenca_de_estrelas'] == 1, 'Com_Estrela', 'Sem_Estrela')
df_quali_base['Franquia'] = np.where(filmes['extensao_de_marca'] == 1, 'Franquia_Sim', 'Franquia_Nao')
df_quali_base['Sazonalidade'] = np.where(filmes['sazonalidade'] == 1, 'Alta_Temporada', 'Temporada_Regular')
for coluna_original, rotulo in generos_avaliados.items():
    df_quali_base[rotulo] = np.where(filmes[coluna_original] == 1, f'{rotulo}_Sim', f'{rotulo}_Nao')

for k_candidato in [2, 3, 4]:
    kmeans_candidato = KMeans(n_clusters=k_candidato, init='k-means++', random_state=100, n_init=10).fit(df_quanti_pad)
    filmes_temp = filmes.copy()
    filmes_temp['ClusterTeste'] = pd.Series(kmeans_candidato.labels_).astype('category')

    print(f"\n[Validação K={k_candidato}] --- ANOVA ---")
    for variavel in ['orcamento_de_producao', 'roi', 'faturamento_bruto_mundial']:
        resultado = pg.anova(dv=variavel, between='ClusterTeste', data=filmes_temp, detailed=True)
        F = resultado.loc[0, 'F']
        p = resultado.loc[0, 'p_unc']
        np2 = resultado.loc[0, 'np2']
        sig = "significativo" if p < 0.05 else "NÃO significativo"
        print(f"  {variavel}: F={F:.3f}, p={p:.6f}, np2={np2:.4f} -> {sig}")

    print(f"[Validação K={k_candidato}] --- Qui-Quadrado ---")
    df_quali_candidato = df_quali_base.copy()
    df_quali_candidato['ClusterTeste'] = 'Cluster_' + pd.Series(kmeans_candidato.labels_).astype(str)
    n_viola_cochran = 0
    n_viola_severa = 0
    for col in variaveis_qualitativas:
        chi2_val, p_val, dof, expected = chi2_contingency(
            pd.crosstab(df_quali_candidato['ClusterTeste'], df_quali_candidato[col])
        )
        sig = "significativo" if p_val < 0.05 else "NÃO significativo"
        viola = expected.min() < 5
        viola_severa = expected.min() < 1
        n_viola_cochran += viola
        n_viola_severa += viola_severa
        # McHugh (2013): nenhuma célula pode ter frequência esperada < 1, independentemente
        # da proporção de células >= 5 exigida pela regra dos 80% (Cochran)
        if viola_severa:
            flag = " [freq. esperada mínima < 1 - violação SEVERA do critério de McHugh]"
        elif viola:
            flag = " [freq. esperada mínima < 5]"
        else:
            flag = ""
        print(f"  {col}: chi2={chi2_val:.3f}, dof={dof}, p={p_val:.6f} -> {sig}{flag}")
    print(f"[Validação K={k_candidato}] Variáveis com violação do pressuposto de frequência esperada: "
          f"{n_viola_cochran} de {len(variaveis_qualitativas)} (branda, <5); "
          f"{n_viola_severa} de {len(variaveis_qualitativas)} (severa, <1).")

# Definição e treinamento do modelo final

k_escolhido = 3
silhueta_k_escolhido = silhueta[list(i_range).index(k_escolhido)]
print(f"\n[Info] K escolhido para o modelo final: {k_escolhido} "
      f"(silhueta média = {silhueta_k_escolhido:.4f})")

kmeans_final = KMeans(n_clusters=k_escolhido, init='k-means++', random_state=100, n_init=10).fit(df_quanti_pad)
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
# O parâmetro data aponta para o DataFrame 'filmes' para avaliar as escalas reais (não padronizadas)
print("\n--- ANOVA por Cluster ---")
for variavel in ['orcamento_de_producao', 'roi', 'faturamento_bruto_mundial']:
    print(f"\n--- ANOVA: {variavel} por Cluster ---")
    print(pg.anova(dv=variavel, between='Cluster', data=filmes, detailed=True).T)

    # Verificação de robustez: homogeneidade de variâncias (Levene, Fávero & Belfiore, 2017),
    # normalidade por grupo (Shapiro-Wilk, Fávero & Belfiore, 2017) e alternativas ao F-test
    # clássico sob heterocedasticidade/não normalidade (Welch; Kruskal-Wallis, Fávero &
    # Belfiore, 2017), aplicadas sobre as escalas reais (não padronizadas)
    levene = pg.homoscedasticity(data=filmes, dv=variavel, group='Cluster')
    welch = pg.welch_anova(data=filmes, dv=variavel, between='Cluster')
    grupos_kw = [filmes.loc[filmes['Cluster'] == c, variavel].values for c in filmes['Cluster'].cat.categories]
    kw_stat, kw_p = kruskal(*grupos_kw)
    shapiro_ps = [shapiro(grupo)[1] for grupo in grupos_kw]
    n_nao_normal = sum(p_sw < 0.05 for p_sw in shapiro_ps)
    print(f"[Robustez] Levene: p={levene['pval'].values[0]:.6f} (equal_var={levene['equal_var'].values[0]}) | "
          f"Welch ANOVA: F={welch['F'].values[0]:.3f}, p={welch['p_unc'].values[0]:.6g} | "
          f"Kruskal-Wallis: H={kw_stat:.3f}, p={kw_p:.6g}")
    print(f"[Robustez] Shapiro-Wilk (normalidade por cluster): {n_nao_normal} de {len(shapiro_ps)} grupos "
          f"rejeitam normalidade (p<0,05) -> p-valores por cluster: {[f'{p_sw:.4g}' for p_sw in shapiro_ps]}")

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

# Mapeamento de todos os 19 gêneros coletados na base (Tabela 1)
for coluna_original, rotulo in generos_avaliados.items():
    df_quali[rotulo] = np.where(filmes[coluna_original] == 1, f'{rotulo}_Sim', f'{rotulo}_Nao')

# Testes Qui-Quadrado de Associação (Referência de validação para a Banca)
print("\n--- Testes de Associação Qui-Quadrado (P-Valor) ---")
for col in variaveis_qualitativas:
    chi2, p, dof, expected = chi2_contingency(pd.crosstab(df_quali["Cluster"], df_quali[col]))
    # McHugh (2013): frequência esperada < 1 em qualquer célula é uma violação severa,
    # distinta da violação branda (< 5) tolerada pela regra dos 80% de Cochran
    if expected.min() < 1:
        aviso = " [ATENÇÃO: frequência esperada mínima < 1 - violação SEVERA de McHugh (2013), resultado não deve ser interpretado]"
    elif expected.min() < 5:
        aviso = " [ATENÇÃO: frequência esperada mínima < 5, resultado pode não ser confiável]"
    else:
        aviso = ""
    print(f"Associação Cluster vs {col} -> p-valor: {round(p, 4)}{aviso}")

# Modelagem da Correspondência Múltipla (ACM) - Extraindo 3 dimensões latentes
mca = prince.MCA(n_components=3, random_state=100).fit(df_quali)

n_vars_ativas = df_quali.shape[1]
n_categorias = sum(df_quali[c].nunique() for c in df_quali.columns)
max_dim = n_categorias - n_vars_ativas
mca_inercia = prince.MCA(n_components=max_dim, random_state=100, correction='benzecri').fit(df_quali)
inercia_bruta_3d = mca.cumulative_percentage_of_variance_[2]
inercia_ajustada_3d = mca_inercia.cumulative_percentage_of_variance_[2]
print(f"\n[ACM] Inércia explicada pelas 3 dimensões retidas: {inercia_bruta_3d:.2f}% (bruta) "
      f"/ {inercia_ajustada_3d:.2f}% (ajustada por Benzécri)")

# Extração e preparação das coordenadas dos componentes (Coordenadas Principais)
coord_padrao = mca.column_coordinates(df_quali)

chart = coord_padrao.reset_index()
var_chart = pd.Series(chart['index'].str.split('__', expand=True).iloc[:, 0])

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
