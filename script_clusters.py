#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 31 16:13:23 2026
@author: Mariana Alcantara Vetromille

Perfis de sucesso comercial entre blockbusters: uma abordagem de clustering não supervisionado

Etapas:
    0. Configurações, constantes e funções auxiliares
    1. Coleta de dados
    2. Tratamento de dados faltantes e correção pela inflação
    3. Padronização multivariada (Z-score)
    4. Agrupamento iterativo (K-means, Elbow e Silhouette)
    5. Validação macroeconômica (ANOVA) e perfil dos clusters
    6. Cruzamento qualitativo (Qui-Quadrado e ACM)
    7. Comparação com a versão nominal (o que mudou com a correção)

Saídas: elbow_blockbusters.png, silhueta_blockbusters.png,
        mapa_perceptual_blockbusters.html e resultados_completos.xlsx
        (uma aba por tabela, com valores numéricos; a aba 'Indice' lista o conteúdo)
"""

# =============================================================================
# 0. CONFIGURAÇÕES, CONSTANTES E FUNÇÕES AUXILIARES
# =============================================================================

import re
from itertools import combinations, permutations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import zscore, chi2_contingency, kruskal, shapiro, ttest_ind, mannwhitneyu
from scipy.stats.contingency import association
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
import pingouin as pg
import plotly.express as px
import prince

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)

# ----- Parâmetros da análise --------------------------------------------------
ARQUIVO_DADOS = 'dados_dos_blockbusters.xlsx'
ARQUIVO_RESULTADOS = 'resultados_completos.xlsx'
ARQUIVO_ELBOW = 'elbow_blockbusters.png'
ARQUIVO_SILHUETA = 'silhueta_blockbusters.png'
ARQUIVO_MAPA = 'mapa_perceptual_blockbusters.html'
ARQUIVO_INFLACAO = 'inflacao_eua.xlsx'      # CPI-U do BLS (série CUUR0000SA0), tabela mensal
ANO_BASE_PRECOS = 2025                     # ano de referência dos dólares constantes
SEMENTE = 100                    # random_state (K-means e ACM)
N_INIT = 10                      # inicializações independentes do K-means
K_ESCOLHIDO = 3                  # justificado na Etapa 4
K_VALIDADOS = [2, 3, 4]          # partições comparadas com testes estatísticos
ALFA = 0.05                      # nível de significância de todos os testes
LIMITE_OUTLIER = 3               # |z| acima do qual um valor é tratado como outlier
LIMITE_RESIDUO = 1.96            # |resíduo ajustado| que sinaliza célula atípica (5%)
MESES_ALTA_TEMPORADA = ['Maio', 'Junho', 'Julho', 'Dezembro']   # Einav (2007)
ORDEM_MESES = ['Janeiro', 'Fevereiro', 'Março', 'Abril', 'Maio', 'Junho', 'Julho',
               'Agosto', 'Setembro', 'Outubro', 'Novembro', 'Dezembro']

# ----- Variáveis quantitativas ------------------------------------------------
VARIAVEIS_CLUSTERIZACAO = ['orcamento_de_producao', 'faturamento_bruto_mundial']
VARIAVEIS_FINANCEIRAS = ['orcamento_de_producao', 'roi', 'faturamento_bruto_mundial']   # ordem da Tabela 2
ORDEM_TABELAS_3_4 = ['orcamento_de_producao', 'faturamento_bruto_mundial', 'roi']        # ordem das Tabelas 3 e 4
QUANTITATIVAS = {                # coluna do banco -> nome de exibição
    'orcamento_de_producao': 'Orçamento de Produção',
    'faturamento_bruto_mundial': 'Faturamento Mundial Bruto',
    'roi': 'Retorno sobre Investimento (ROI)',
    'proporcao_de_novatos_no_elenco_principal': 'Proporção de Novatos no Elenco',
    'ano': 'Ano de Lançamento',
}

# ----- Variáveis qualitativas binárias ----------------------------------------
GENEROS = {                      # coluna do banco -> (rótulo interno, nome de exibição)
    'genero_acao': ('Acao', 'Ação'),
    'genero_aventura': ('Aventura', 'Aventura'),
    'genero_sci_fi': ('SciFi', 'Ficção Científica'),
    'genero_fantasia': ('Fantasia', 'Fantasia'),
    'genero_animacao': ('Animacao', 'Animação'),
    'genero_comedia': ('Comedia', 'Comédia'),
    'genero_drama': ('Drama', 'Drama'),
    'genero_romance': ('Romance', 'Romance'),
    'genero_crime': ('Crime', 'Crime'),
    'genero_misterio': ('Misterio', 'Mistério'),
    'genero_familia': ('Familia', 'Família'),
    'genero_musical': ('Musical', 'Musical'),
    'genero_suspense': ('Suspense', 'Suspense'),
    'genero_biografia': ('Biografia', 'Biografia'),
    'genero_historia': ('Historia', 'História'),
    'genero_guerra': ('Guerra', 'Guerra'),
    'genero_horror': ('Horror', 'Horror'),
    'genero_esporte': ('Esporte', 'Esporte'),
    'genero_faroeste': ('Faroeste', 'Faroeste'),
}
generos_avaliados = {coluna: rotulo for coluna, (rotulo, _) in GENEROS.items()}

COLUNA_BINARIA = {'Star_Power': 'presenca_de_estrelas', 'Franquia': 'extensao_de_marca',
                  'Sazonalidade': 'sazonalidade',
                  **{rotulo: coluna for coluna, rotulo in generos_avaliados.items()}}
variaveis_fixas = ['Star_Power', 'Franquia', 'Sazonalidade']
variaveis_qualitativas = list(COLUNA_BINARIA)

CATEGORIAS_BINARIAS = {'Star_Power': ('Com_Estrela', 'Sem_Estrela'),
                       'Franquia': ('Franquia_Sim', 'Franquia_Nao'),
                       'Sazonalidade': ('Alta_Temporada', 'Temporada_Regular'),
                       **{rotulo: (f'{rotulo}_Sim', f'{rotulo}_Nao') for rotulo in generos_avaliados.values()}}

# ----- Rótulos de apresentação (tabelas e mapa) -------------------------------
NOMES_CLUSTERS = {0: 'Cluster 0 (Blockbusters Eficientes)',
                  1: 'Cluster 1 (Superproduções Bilionárias)',
                  2: 'Cluster 2 (Blockbusters de Baixo Retorno)'}

NOMES_QUALITATIVAS = {'Star_Power': 'Sinalização de Apelo de Elenco (Star Power)',
                      'Franquia': 'Extensão Corporativa de Marca (Franquia)',
                      'Sazonalidade': 'Sazonalidade de Lançamento (Alta Temporada)',
                      **{rotulo: nome for _, (rotulo, nome) in GENEROS.items()}}
NOMES_EXTRAS = {'Mes': 'Mês de lançamento (categorias originais)',
                'Pais': 'País de origem (categorias originais)'}
NOMES_TODAS = {**NOMES_QUALITATIVAS, **NOMES_EXTRAS}

NOMES_MAPA = {'Cluster': 'Cluster', 'Star_Power': 'Presença de estrelas', 'Franquia': 'Franquia',
              'Sazonalidade': 'Sazonalidade',
              **{rotulo: nome for _, (rotulo, nome) in GENEROS.items()}, 'SciFi': 'Sci-Fi'}
ROTULOS_ESPECIAIS_MAPA = {'Com_Estrela': 'Com estrela', 'Sem_Estrela': 'Sem estrela',
                          'Alta_Temporada': 'Alta temporada', 'Temporada_Regular': 'Temporada regular'}

ORDEM_TABELA_7 = ['SciFi', 'Acao', 'Aventura', 'Comedia', 'Drama']


def rotulo_categoria(variavel, categoria):
    """Rótulo legível de uma categoria do prince (ex.: 'SciFi_Nao' -> 'Sci-Fi: não')."""
    nome = NOMES_MAPA.get(variavel, variavel)
    if categoria in ROTULOS_ESPECIAIS_MAPA:
        return ROTULOS_ESPECIAIS_MAPA[categoria]
    if categoria.endswith('_Sim'):
        return f'{nome}: sim'
    if categoria.endswith('_Nao'):
        return f'{nome}: não'
    return categoria.replace('_', ' ')


def nome_categoria(variavel, categoria):
    return rotulo_categoria(variavel, categoria) if variavel in CATEGORIAS_BINARIAS else str(categoria)


ROTULO_SIM = {v: rotulo_categoria(v, cats[0]) for v, cats in CATEGORIAS_BINARIAS.items()}


# ----- Formatação e registro de tabelas ---------------------------------------
def cabecalho(texto):
    print(f"\n--- {texto} ---")


def formatar_br(valor, casas=2):
    """Formata no padrão brasileiro: ponto no milhar e vírgula no decimal."""
    texto = f"{valor:,.{casas}f}"
    return texto.replace(",", "X").replace(".", ",").replace("X", ".")


_SUPERESCRITOS = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


def formatar_cientifico_br(valor, casas=4):
    mantissa, expoente = f"{valor:.{casas}e}".split("e")
    return f"{mantissa.replace('.', ',')} × 10{str(int(expoente)).translate(_SUPERESCRITOS)}"


def formatar_numero_tabela(valor):
    return formatar_cientifico_br(valor) if abs(valor) >= 1e6 else formatar_br(valor, 4)


def casas(n):
    return lambda v: formatar_br(v, n)


def pct(n=1):
    return lambda v: formatar_br(v, n) + '%'


def fracao_pct(n=2):
    return lambda v: formatar_br(v * 100, n) + '%'


def valor_p(v):
    return '< 0,0001' if v < 0.0001 else formatar_br(v, 4)


def p_cientifico(v):
    return f"{v:.2e}".replace('.', ',')


def com_sinal(n=2):
    return lambda v: f"{v:+.{n}f}".replace('.', ',')


def _formato_padrao(v):
    if isinstance(v, (bool, np.bool_)):
        return 'sim' if v else 'não'
    if isinstance(v, (int, np.integer)):
        return str(v)
    if isinstance(v, (float, np.floating)):
        return formatar_br(v, 2)
    return str(v)


def formatar_df(df, fmt=None):
    fmt = fmt or {}
    saida = pd.DataFrame(index=df.index)
    for coluna in df.columns:
        f = fmt.get(coluna, _formato_padrao)
        saida[coluna] = ['' if pd.isna(v) else f(v) for v in df[coluna]]
    return saida


def mostrar(titulo, df_formatado, colunas=None):
    cabecalho(titulo)
    print((df_formatado if colunas is None else df_formatado[colunas]).to_string(index=False))


tabelas = {}


def registrar(nome, df, titulo, fmt=None, colunas=None, imprimir=True, ajuste=None):
    tabelas[nome] = (titulo, df.reset_index(drop=True))
    if imprimir:
        formatada = formatar_df(df, fmt)
        mostrar(titulo, formatada if ajuste is None else ajuste(formatada, df), colunas)


def ano_sem_milhar(colunas):
    def ajustar(formatada, df):
        linha_ano = df['Variável'] == QUANTITATIVAS['ano']
        for c in colunas:
            formatada.loc[linha_ano, c] = df.loc[linha_ano, c].map(lambda v: f"{v:.2f}".replace('.', ','))
        return formatada
    return ajustar


def exportar_tabelas(caminho):
    indice = pd.DataFrame([{'Aba': nome, 'Descrição': titulo, 'Linhas': len(df)}
                           for nome, (titulo, df) in tabelas.items()])
    try:
        with pd.ExcelWriter(caminho, engine='openpyxl') as escritor:
            indice.to_excel(escritor, sheet_name='Indice', index=False)
            for nome, (_, df) in tabelas.items():
                df.to_excel(escritor, sheet_name=nome[:31], index=False)
            for aba in escritor.sheets.values():
                aba.freeze_panes = 'A2'
                for coluna in aba.columns:
                    largura = max(len(str(c.value)) if c.value is not None else 0 for c in coluna[:200])
                    aba.column_dimensions[coluna[0].column_letter].width = min(largura + 2, 60)
        print(f"\n[Exportação] {len(tabelas)} tabelas salvas em '{caminho}' (aba 'Indice' lista o conteúdo).")
    except PermissionError:
        print(f"\n[Aviso] Não foi possível gravar '{caminho}': feche o arquivo no Excel e execute novamente.")


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


def grafico_por_k(x, y, cor, rotulo_y, arquivo):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(x, y, marker='o', color=cor)
    ax.set_xlabel('Nº Clusters')
    ax.set_xticks(list(x))
    ax.set_ylabel(rotulo_y)
    formatar_grafico_conforme_manual(ax)
    plt.savefig(arquivo, dpi=300, bbox_inches='tight')
    plt.show()


# ----- Funções estatísticas ---------------------------------------------------
def ler_cpi_anual(caminho):
    """Média anual do CPI-U a partir do xlsx do BLS (tabela mensal com cabeçalho 'Year')."""
    bruto = pd.read_excel(caminho, header=None)
    linha_cabecalho = bruto.index[bruto[0].astype(str).str.strip() == 'Year'][0]
    tabela = bruto.iloc[linha_cabecalho + 1:].copy()
    tabela.columns = bruto.iloc[linha_cabecalho].tolist()
    tabela = tabela.apply(pd.to_numeric, errors='coerce')          # células vazias viram NaN
    meses = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    return pd.DataFrame({'Ano': tabela['Year'].astype(int).values,
                         'Meses disponíveis': tabela[meses].notna().sum(axis=1).values,
                         'CPI-U (média anual)': tabela[meses].mean(axis=1).values})


def anova_por_cluster(rotulos, variaveis, base=None):
    dados = (filmes if base is None else base)[variaveis].assign(Cluster=pd.Categorical(np.asarray(rotulos)))
    linhas = []
    for v in variaveis:
        a = pg.anova(dv=v, between='Cluster', data=dados, detailed=True)
        entre, interna = a.iloc[0], a.iloc[1]
        linhas.append({'Código': v, 'Variável': QUANTITATIVAS[v],
                       'SQ Entre': entre['SS'], 'SQ Interna': interna['SS'],
                       'GL Entre': int(entre['DF']), 'GL Interna': int(interna['DF']),
                       'QM Entre': entre['MS'], 'QM Interna': interna['MS'],
                       'F': entre['F'], 'Valor-p': entre['p_unc'], 'η² parcial': entre['np2']})
    return pd.DataFrame(linhas)


def testar_associacoes(rotulos, categoricas, n_familia=None):
    """Qui-Quadrado (cluster x variável) para cada coluna de `categoricas`."""
    rotulos = pd.Series(np.asarray(rotulos), name='Cluster')
    n_familia = len(categoricas.columns) if n_familia is None else n_familia
    resumo, detalhe = [], []
    for var in categoricas.columns:
        obs = pd.crosstab(rotulos, np.asarray(categoricas[var]))
        chi2, p, dof, esp = chi2_contingency(obs)
        p_bonferroni = min(p * n_familia, 1)
        total = obs.values.sum()
        tot_cluster = obs.sum(axis=1).values[:, None]
        tot_categoria = obs.sum(axis=0).values[None, :]
        residuo = (obs.values - esp) / np.sqrt(esp * (1 - tot_cluster / total) * (1 - tot_categoria / total))
        i, j = np.unravel_index(np.argmin(esp), esp.shape)
        resumo.append({'Código': var, 'Variável': NOMES_TODAS.get(var, var), 'GL': int(dof),
                       'χ²': chi2, 'Valor-p': p, 'Valor-p (Bonferroni)': p_bonferroni,
                       'V de Cramér': association(obs, method='cramer'),
                       'Freq. esperada mínima': esp.min(),
                       'Célula da mínima': f"{nome_categoria(var, obs.columns[j])} × Cluster {obs.index[i]}",
                       'Células': esp.size, 'Células < 5': int((esp < 5).sum()), 'Células < 1': int((esp < 1).sum()),
                       '% células ≥ 5': (esp >= 5).mean() * 100,
                       'Marca': '**' if esp.min() < 1 else ('*' if esp.min() < 5 else ''),
                       'Significativo': p < ALFA,
                       'Significativo (Bonferroni)': p_bonferroni < ALFA})
        primeiro = CATEGORIAS_BINARIAS[var][0] if var in CATEGORIAS_BINARIAS else None
        for categoria in sorted(obs.columns, key=lambda c: c != primeiro):     # "sim" antes de "não"
            b = obs.columns.get_loc(categoria)
            for a, cluster in enumerate(obs.index):
                detalhe.append({'Código': var, 'Variável': NOMES_TODAS.get(var, var),
                                'Categoria': nome_categoria(var, categoria), 'Cluster': cluster,
                                'Observado': int(obs.iat[a, b]), 'Esperado': esp[a, b],
                                'Resíduo ajustado': residuo[a, b]})
    return pd.DataFrame(resumo), pd.DataFrame(detalhe)


def composicao_categorias(detalhe, n_por_cluster):
    """Nº e % de filmes de cada categoria na base e em cada cluster."""
    obs = detalhe.pivot(index=['Variável', 'Categoria'], columns='Cluster', values='Observado')
    obs = obs.reindex(pd.MultiIndex.from_frame(detalhe[['Variável', 'Categoria']].drop_duplicates()))
    n_base = obs.sum(axis=1)
    resultado = pd.DataFrame({'N base': n_base, '% base': n_base / n_por_cluster.sum() * 100})
    for k in obs.columns:
        resultado[f'N Cluster {k}'] = obs[k]
        resultado[f'% Cluster {k}'] = obs[k] / n_por_cluster[k] * 100
    return resultado.reset_index()



def main():
    """Executa o pipeline completo: coleta, tratamento, padronização, clusterização,
    validação estatística, cruzamento qualitativo e exportação dos resultados."""


    global filmes

    # =============================================================================
    # 1. COLETA DE DADOS
    # =============================================================================

    try:
        filmes = pd.read_excel(ARQUIVO_DADOS)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Arquivo '{ARQUIVO_DADOS}' não encontrado no diretório atual. "
            "Verifique se o script está sendo executado na mesma pasta do arquivo de dados."
        )

    print("--- Estrutura Inicial do Banco de Dados ---")
    filmes.info()

    # Colunas efetivamente utilizadas na análise
    colunas_analise = [
        'orcamento_de_producao', 'roi', 'faturamento_bruto_mundial', 'mes_de_lancamento',
        'presenca_de_estrelas', 'extensao_de_marca', 'proporcao_de_novatos_no_elenco_principal',
    ] + list(generos_avaliados.keys())

    colunas_faltantes = [c for c in colunas_analise if c not in filmes.columns]
    if colunas_faltantes:
        raise KeyError(f"Colunas esperadas ausentes no banco de dados: {colunas_faltantes}")

    # Definição do ROI: a coluna 'roi' deve coincidir com (Faturamento - Orçamento) / Orçamento
    diff_multiplicador = (filmes['roi'] - filmes['faturamento_bruto_mundial'] / filmes['orcamento_de_producao']).abs().max()
    diff_percentual = (filmes['roi'] - (filmes['faturamento_bruto_mundial'] - filmes['orcamento_de_producao']) / filmes['orcamento_de_producao']).abs().max()
    print(f"\n[Diagnóstico] ROI: maior diferença vs. Faturamento/Orçamento = {diff_multiplicador:.4f}; "
          f"vs. (Faturamento-Orçamento)/Orçamento = {diff_percentual:.4f}")

    # =============================================================================
    # 2. TRATAMENTO DE DADOS
    # =============================================================================

    # Registro do que será removido (antes do descarte)
    ausentes = filmes[colunas_analise].isna()
    nulos_por_coluna = ausentes.sum()
    nulos_por_coluna = nulos_por_coluna[nulos_por_coluna > 0]
    removidos = filmes.loc[ausentes.any(axis=1), ['rank', 'titulo_ingles', 'ano']].copy()
    if not removidos.empty:
        removidos['Colunas ausentes'] = ausentes[ausentes.any(axis=1)].apply(lambda l: ', '.join(l.index[l]), axis=1)
    removidos.columns = ['Rank', 'Título', 'Ano'] + (['Colunas ausentes'] if not removidos.empty else [])

    # Exclusão em lista (listwise)
    n_antes = len(filmes)
    filmes = filmes.dropna(subset=colunas_analise).reset_index(drop=True)
    n_depois = len(filmes)
    print(f"\n[Tratamento] {n_antes - n_depois} linha(s) removida(s) por valores ausentes "
          f"nas colunas de análise ({n_depois} de {n_antes} linhas mantidas).")
    if not nulos_por_coluna.empty:
        registrar('Nulos_por_coluna',
                  pd.DataFrame({'Coluna': nulos_por_coluna.index, 'Linhas ausentes': nulos_por_coluna.values,
                                '% da base original': nulos_por_coluna.values / n_antes * 100}),
                  'Valores ausentes por coluna de análise (base original)', fmt={'% da base original': pct(2)})
        registrar('Filmes_removidos', removidos, 'Filmes removidos por valores ausentes')

    # ----- Correção pela inflação: dólares constantes de ANO_BASE_PRECOS ----------
    try:
        cpi = ler_cpi_anual(ARQUIVO_INFLACAO)
    except FileNotFoundError:
        raise FileNotFoundError(f"Arquivo '{ARQUIVO_INFLACAO}' não encontrado no diretório atual. "
                                "Ele é necessário para converter orçamento e faturamento em dólares constantes.")

    anos_necessarios = sorted(set(filmes['ano'].astype(int)) | {ANO_BASE_PRECOS})
    ausentes_cpi = [a for a in anos_necessarios if a not in set(cpi['Ano'])]
    if ausentes_cpi:
        raise ValueError(f"CPI ausente para os anos: {ausentes_cpi}")
    cpi = cpi[cpi['Ano'].isin(anos_necessarios)].reset_index(drop=True)
    indice_cpi = cpi.set_index('Ano')['CPI-U (média anual)']
    cpi['Fator para dólares constantes'] = indice_cpi[ANO_BASE_PRECOS] / cpi['CPI-U (média anual)']
    registrar('CPI_anual', cpi, f'CPI-U (BLS, CUUR0000SA0) | média anual e fator para dólares de {ANO_BASE_PRECOS}',
              fmt={'CPI-U (média anual)': casas(3), 'Fator para dólares constantes': casas(4)})
    incompletos = cpi.loc[cpi['Meses disponíveis'] < 12, 'Ano'].tolist()
    if incompletos:
        print(f"Atenção: média anual do CPI calculada com menos de 12 meses em {incompletos} (mês sem publicação no arquivo).")

    filmes['orcamento_nominal'] = filmes['orcamento_de_producao']
    filmes['faturamento_nominal'] = filmes['faturamento_bruto_mundial']
    filmes['fator_inflacao'] = filmes['ano'].astype(int).map(cpi.set_index('Ano')['Fator para dólares constantes'])
    filmes['orcamento_de_producao'] = filmes['orcamento_nominal'] * filmes['fator_inflacao']
    filmes['faturamento_bruto_mundial'] = filmes['faturamento_nominal'] * filmes['fator_inflacao']

    roi_constante = ((filmes['faturamento_bruto_mundial'] - filmes['orcamento_de_producao'])
                     / filmes['orcamento_de_producao'])
    roi_inalterado = bool(np.allclose(roi_constante, filmes['roi']))
    print(f"\n[Correção pela inflação] valores em dólares constantes de {ANO_BASE_PRECOS}; "
          f"ROI com valores constantes idêntico ao original: {roi_inalterado}")
    if not roi_inalterado:
        raise ValueError("O ROI recalculado com valores constantes difere do ROI original: verifique a base.")

    registrar('Correcao_resumo',
              pd.DataFrame([{'Variável': nome, 'Média nominal (US$)': filmes[nom].mean(),
                             f'Média constante (US$ de {ANO_BASE_PRECOS})': filmes[col].mean(),
                             'Razão de médias': filmes[col].mean() / filmes[nom].mean(),
                             'Fator mínimo': filmes['fator_inflacao'].min(), 'Fator máximo': filmes['fator_inflacao'].max()}
                            for nome, nom, col in [('Orçamento de Produção', 'orcamento_nominal', 'orcamento_de_producao'),
                                                   ('Faturamento Mundial Bruto', 'faturamento_nominal', 'faturamento_bruto_mundial')]]),
              f'Correção pela inflação | valores nominais x dólares constantes de {ANO_BASE_PRECOS}',
              fmt={'Razão de médias': casas(3), 'Fator mínimo': casas(4), 'Fator máximo': casas(4)})

    # ----- Engenharia de recursos: sazonalidade de lançamento ---------------------
    mes_registrado = filmes['mes_de_lancamento'].astype(str).str.strip()
    filmes['mes_de_lancamento'] = mes_registrado.where(mes_registrado.str.contains(r'\(', regex=True), mes_registrado.str.capitalize())
    filmes['sazonalidade'] = np.where(filmes['mes_de_lancamento'].isin(MESES_ALTA_TEMPORADA), 1, 0)


    def mes_do_ano_de_lancamento(mes_registrado, ano):
        """Em registros com duas datas (relançamento/remake), devolve o mês do ano de lançamento do filme."""
        datas = {int(ano_do_mes): mes.capitalize()
                for mes, ano_do_mes in re.findall(r'([^\W\d_]+)\s*\((\d{4})\)', mes_registrado)}
        return datas.get(int(ano), mes_registrado)


    filmes['mes_do_ano_de_lancamento'] = [mes_do_ano_de_lancamento(m, a)
                                          for m, a in zip(filmes['mes_de_lancamento'], filmes['ano'])]
    filmes['sazonalidade_alternativa'] = np.where(filmes['mes_do_ano_de_lancamento'].isin(MESES_ALTA_TEMPORADA), 1, 0)

    meses_fora_padrao = filmes[~filmes['mes_de_lancamento'].isin(ORDEM_MESES)]
    if meses_fora_padrao.empty:
        print("\n[Verificação] Todos os meses de lançamento estão no padrão (um mês por filme).")
    else:
        texto_temporada = {1: 'Alta temporada', 0: 'Temporada regular'}
        registrar('Meses_fora_do_padrao',
                  pd.DataFrame({'Rank': meses_fora_padrao['rank'], 'Título': meses_fora_padrao['titulo_ingles'],
                                'Ano': meses_fora_padrao['ano'], 'Mês registrado': meses_fora_padrao['mes_de_lancamento'],
                                'Mês do ano de lançamento': meses_fora_padrao['mes_do_ano_de_lancamento'],
                                'Sazonalidade usada': meses_fora_padrao['sazonalidade'].map(texto_temporada),
                                'Sazonalidade pelo mês do ano': meses_fora_padrao['sazonalidade_alternativa'].map(texto_temporada)}),
                  'Meses de lançamento com duas datas (classificados como temporada regular por não coincidirem com um mês)')

    # ----- Variáveis qualitativas (base para o Qui-Quadrado e para a ACM) ---------
    df_quali_base = pd.DataFrame()
    for variavel, (categoria_sim, categoria_nao) in CATEGORIAS_BINARIAS.items():
        df_quali_base[variavel] = np.where(filmes[COLUNA_BINARIA[variavel]] == 1, categoria_sim, categoria_nao)

    # =============================================================================
    # 3. PADRONIZAÇÃO MULTIVARIADA (Z-SCORE)
    # =============================================================================

    desvios = filmes[VARIAVEIS_CLUSTERIZACAO].std(ddof=0)
    if (desvios == 0).any():
        raise ValueError(f"Variável(is) com variância zero, impossível padronizar via Z-Score: "
                         f"{desvios[desvios == 0].index.tolist()}")

    # Divisor populacional (ddof=0)
    z_financeiras = filmes[VARIAVEIS_FINANCEIRAS].apply(zscore, ddof=0)
    df_quanti_pad = z_financeiras[VARIAVEIS_CLUSTERIZACAO].copy()


    def descrever(serie):
        z = zscore(serie, ddof=0)
        return {'N': int(serie.count()), 'Média': serie.mean(), 'Mediana': serie.median(),
                'Desvio-padrão': serie.std(), 'Mínimo': serie.min(), 'Q1': serie.quantile(0.25),
                'Q3': serie.quantile(0.75), 'Máximo': serie.max(), 'Assimetria': serie.skew(),
                'Curtose': serie.kurt(), 'Outliers (|z| > 3)': int((np.abs(z) > LIMITE_OUTLIER).sum())}


    descritivas_base_df = pd.DataFrame([{'Variável': nome,
                                         'Papel na análise': 'Clusterização' if col in VARIAVEIS_CLUSTERIZACAO else 'Descritiva',
                                         **descrever(filmes[col])} for col, nome in QUANTITATIVAS.items()])

    # ----- Comparativo da proporção entre as médias gerais de Orçamento e Faturamento ---------------
    media_orcamento_geral = filmes['orcamento_de_producao'].mean()
    media_faturamento_geral = filmes['faturamento_bruto_mundial'].mean()
    comparativo_orcamento_faturamento = pd.DataFrame([
        {'Variável': 'Faturamento Mundial Bruto em relação a Orçamento de Produção', 'Papel na análise': 'Comparação',
         'Proporção da Média Geral (%)': media_faturamento_geral / media_orcamento_geral * 100},
        {'Variável': 'Orçamento de Produção em relação a Faturamento Mundial Bruto', 'Papel na análise': 'Comparação',
         'Proporção da Média Geral (%)': media_orcamento_geral / media_faturamento_geral * 100},
    ])
    descritivas_base_df = pd.concat([descritivas_base_df, comparativo_orcamento_faturamento], ignore_index=True)

    registrar('Descritivas_base', descritivas_base_df,
              'Estatísticas descritivas das variáveis quantitativas (base tratada)',
              fmt={'Assimetria': casas(3), 'Curtose': casas(3), 'Proporção da Média Geral (%)': pct(2)},
              colunas=['Variável', 'N', 'Média', 'Mediana', 'Desvio-padrão', 'Mínimo', 'Q1', 'Q3',
                       'Máximo', 'Assimetria', 'Curtose', 'Outliers (|z| > 3)', 'Proporção da Média Geral (%)'],
              ajuste=ano_sem_milhar(['Média', 'Mediana', 'Desvio-padrão', 'Mínimo', 'Q1', 'Q3', 'Máximo']))

    # =============================================================================
    # 4. AGRUPAMENTO ITERATIVO (K-MEANS, ELBOW E SILHOUETTE)
    # =============================================================================

    # Cache: cada K é treinado uma única vez e reaproveitado nas etapas seguintes
    modelos_kmeans = {}


    def obter_kmeans(k):
        if k not in modelos_kmeans:
            modelos_kmeans[k] = KMeans(n_clusters=k, init='k-means++', random_state=SEMENTE,
                                       n_init=N_INIT).fit(df_quanti_pad)
        return modelos_kmeans[k]


    # ----- Método de Elbow (Figura 3) e Coeficiente de Silhueta (Figura 2) ----
    k_elbow = range(1, 11)
    k_silhueta = range(2, 11)
    wcss = {k: obter_kmeans(k).inertia_ for k in k_elbow}
    silhueta = {k: silhouette_score(df_quanti_pad, obter_kmeans(k).labels_) for k in k_silhueta}

    grafico_por_k(k_elbow, list(wcss.values()), 'blue', 'WCSS (Inércia)', ARQUIVO_ELBOW)
    grafico_por_k(k_silhueta, list(silhueta.values()), 'purple', 'Silhueta Média', ARQUIVO_SILHUETA)

    # ----- Silhueta, inércia e tamanho dos clusters por K ---------------------
    linhas_k = []
    for k in k_elbow:
        tamanhos = pd.Series(obter_kmeans(k).labels_).value_counts().sort_index().tolist()
        linhas_k.append({'K': k, 'WCSS (Inércia)': wcss[k], 'Silhueta média': silhueta.get(k, np.nan),
                         'Tamanhos dos clusters': str(tamanhos), 'Menor cluster (n)': min(tamanhos),
                         'Menor cluster (% da base)': min(tamanhos) / len(filmes) * 100,
                         'K escolhido': 'sim' if k == K_ESCOLHIDO else ''})
    registrar('Validacao_K', pd.DataFrame(linhas_k), 'Validação de K | inércia, silhueta e tamanho dos clusters',
              fmt={'Silhueta média': casas(4), 'Menor cluster (% da base)': pct(2)})

    # ----- Reforço estatístico: ANOVA e Qui-Quadrado para K = 2, 3 e 4 -------
    anova_por_k = pd.concat([anova_por_cluster(obter_kmeans(k).labels_, VARIAVEIS_FINANCEIRAS).assign(K=k)
                             for k in K_VALIDADOS], ignore_index=True)
    anova_por_k['Significativo'] = anova_por_k['Valor-p'] < ALFA
    registrar('Validacao_K_ANOVA', anova_por_k[['K', 'Variável', 'F', 'Valor-p', 'η² parcial', 'Significativo']],
              'Validação de K | ANOVA das variáveis financeiras',
              fmt={'F': casas(2), 'Valor-p': valor_p, 'η² parcial': casas(4)})

    qui2_por_k = pd.concat([testar_associacoes(obter_kmeans(k).labels_, df_quali_base)[0].assign(K=k)
                            for k in K_VALIDADOS], ignore_index=True)
    tabelas['Validacao_K_Qui2'] = ('Validação de K | Qui-Quadrado das 22 variáveis qualitativas (formato longo, '
                                   'inclui valor-p nominal e com correção de Bonferroni)', qui2_por_k)

    # Valor-p nominal e com correção de Bonferroni
    p_por_k = (qui2_por_k.assign(texto=qui2_por_k['Valor-p'].map(casas(4)) + qui2_por_k['Marca'])
               .pivot(index='Variável', columns='K', values='texto')
               .reindex([NOMES_QUALITATIVAS[c] for c in variaveis_qualitativas]))
    p_bonferroni_por_k = (qui2_por_k.assign(texto=qui2_por_k['Valor-p (Bonferroni)'].map(casas(4)))
                          .pivot(index='Variável', columns='K', values='texto')
                          .reindex([NOMES_QUALITATIVAS[c] for c in variaveis_qualitativas]))
    tabela_p_por_k = pd.DataFrame(index=p_por_k.index)
    for k in K_VALIDADOS:
        tabela_p_por_k[f'K = {k} (nominal)'] = p_por_k[k]
        tabela_p_por_k[f'K = {k} (Bonferroni)'] = p_bonferroni_por_k[k]
    resumo_por_k = pd.DataFrame(
        {f'K = {k}': [str(int(g['Significativo'].sum())), str(int(g['Significativo (Bonferroni)'].sum())),
                      str(int((g['Células < 5'] > 0).sum())), str(int((g['Células < 1'] > 0).sum()))]
         for k, g in qui2_por_k.groupby('K')},
        index=['Associações significativas (p < 0,05)', 'Associações significativas após Bonferroni (p < 0,05)',
               'Variáveis com célula esperada < 5', 'Variáveis com célula esperada < 1'])

    colunas_resumo = tabela_p_por_k.rename_axis('Variável').reset_index().columns.tolist()
    linha_em_branco = pd.DataFrame([{c: '' for c in colunas_resumo}])
    indicador_nominal, indicador_bonferroni = resumo_por_k.index[0], resumo_por_k.index[1]
    linhas_contagem = []
    for nome_indicador in resumo_por_k.index:
        linha = {'Variável': nome_indicador}
        for k in K_VALIDADOS:
            valor = resumo_por_k.loc[nome_indicador, f'K = {k}']
            if nome_indicador == indicador_nominal:
                linha[f'K = {k} (nominal)'], linha[f'K = {k} (Bonferroni)'] = valor, ''
            elif nome_indicador == indicador_bonferroni:
                linha[f'K = {k} (nominal)'], linha[f'K = {k} (Bonferroni)'] = '', valor
            else:
                linha[f'K = {k} (nominal)'], linha[f'K = {k} (Bonferroni)'] = valor, valor
        linhas_contagem.append(linha)
    tabela_resumo_unida = pd.concat([tabela_p_por_k.rename_axis('Variável').reset_index(), linha_em_branco,
                                     pd.DataFrame(linhas_contagem)[colunas_resumo]], ignore_index=True)
    tabelas['Validacao_K_Qui2_resumo'] = (
        'Validação de K | Qui-Quadrado por variável (valor-p nominal e com correção de Bonferroni) e '
        'contagens-resumo por K, família de 22 testes qualitativos', tabela_resumo_unida)

    mostrar('Validação de K | Qui-Quadrado: valor-p nominal e com correção de Bonferroni por variável',
            tabela_p_por_k.rename_axis('Variável').reset_index())
    mostrar('Validação de K | Qui-Quadrado: contagens-resumo por K', resumo_por_k.rename_axis('Indicador').reset_index())
    print("* frequência esperada mínima entre 1 e 5 | ** frequência esperada mínima < 1")

    # ----- Modelo final -------------------------------------------------------
    z_nominal = filmes[['orcamento_nominal', 'faturamento_nominal']].apply(zscore, ddof=0)
    rotulos_nominais = KMeans(n_clusters=K_ESCOLHIDO, init='k-means++', random_state=SEMENTE,
                              n_init=N_INIT).fit(z_nominal).labels_


    def alinhar_rotulos(rotulos, referencia):
        contingencia = pd.crosstab(pd.Series(referencia, name='ref'), pd.Series(rotulos, name='novo'))
        linhas, colunas = linear_sum_assignment(-contingencia.values)
        correspondencia = {contingencia.columns[c]: contingencia.index[l] for l, c in zip(linhas, colunas)}
        return np.array([correspondencia[r] for r in rotulos])


    rotulos_finais = alinhar_rotulos(obter_kmeans(K_ESCOLHIDO).labels_, rotulos_nominais)
    filmes['Cluster'] = pd.Categorical(rotulos_finais)
    clusters = list(filmes['Cluster'].cat.categories)
    n_por_cluster = filmes['Cluster'].value_counts().sort_index()

    # =============================================================================
    # 5. VALIDAÇÃO MACROECONÔMICA (ANOVA) E PERFIL DOS CLUSTERS
    # =============================================================================

    # ----- Tabela 2: ANOVA por cluster --------------------------------------------
    anova_final = anova_por_cluster(rotulos_finais, list(QUANTITATIVAS))
    tabelas['ANOVA_K3'] = ('ANOVA por cluster (K = 3) | todas as variáveis quantitativas', anova_final)

    linhas2 = []
    for _, r in anova_final.set_index('Código').loc[VARIAVEIS_FINANCEIRAS].iterrows():
        linhas2.append([r['Variável'], 'Entre Clusters', formatar_numero_tabela(r['SQ Entre']), r['GL Entre'],
                        formatar_numero_tabela(r['QM Entre']), formatar_br(r['F'], 2), valor_p(r['Valor-p']),
                        formatar_br(r['η² parcial'], 4)])
        linhas2.append(['', 'Interna', formatar_numero_tabela(r['SQ Interna']), r['GL Interna'],
                        formatar_numero_tabela(r['QM Interna']), '', '', ''])
    registrar('Tabela_2', pd.DataFrame(linhas2, columns=['Variável Macroeconômica', 'Tipo de Variabilidade',
                                                            'Soma dos Quadrados', 'Graus de Liberdade',
                                                            'Quadrado Médio', 'Estatística F', 'Valor-p',
                                                            'η² Parcial']),
              'Tabela 2 | ANOVA por cluster')

    mostrar('ANOVA por cluster | demais variáveis quantitativas (exploratória)',
            formatar_df(anova_final[~anova_final['Código'].isin(VARIAVEIS_FINANCEIRAS)],
                        {'F': casas(2), 'Valor-p': valor_p, 'η² parcial': casas(4)}),
            ['Variável', 'F', 'Valor-p', 'η² parcial'])

    # ----- Robustez da ANOVA: Levene, Welch, Kruskal-Wallis e Shapiro-Wilk --------
    linhas_robustez = []
    for v in QUANTITATIVAS:
        grupos = [filmes.loc[filmes['Cluster'] == c, v].values for c in clusters]
        levene = pg.homoscedasticity(data=filmes, dv=v, group='Cluster')
        welch = pg.welch_anova(data=filmes, dv=v, between='Cluster')
        h, p_kw = kruskal(*grupos)
        p_shapiro = [shapiro(g)[1] for g in grupos]
        linhas_robustez.append({'Código': v, 'Variável': QUANTITATIVAS[v],
                                'Levene W': levene['W'].values[0], 'Levene p': levene['pval'].values[0],
                                'Welch F': welch['F'].values[0], 'Welch p': welch['p_unc'].values[0],
                                'Kruskal-Wallis H': h, 'Kruskal-Wallis p': p_kw,
                                **{f'Shapiro p Cluster {c}': p for c, p in zip(clusters, p_shapiro)},
                                'Grupos não normais': sum(p < ALFA for p in p_shapiro)})
    registrar('Robustez_ANOVA', pd.DataFrame(linhas_robustez),
              'Robustez da ANOVA | Levene, Welch, Kruskal-Wallis e Shapiro-Wilk (grupos não normais, de 3)',
              fmt={'Levene p': p_cientifico, 'Welch p': p_cientifico, 'Kruskal-Wallis p': p_cientifico,
                   **{f'Shapiro p Cluster {c}': p_cientifico for c in clusters}},
              colunas=['Variável', 'Levene W', 'Levene p', 'Welch F', 'Welch p', 'Kruskal-Wallis H',
                       'Kruskal-Wallis p', 'Grupos não normais'])

    # ----- Tabela 3: testes de robustez da comparação de médias entre os clusters -
    robustez_fin = pd.DataFrame(linhas_robustez).set_index('Código').loc[ORDEM_TABELAS_3_4]
    linhas3 = []
    for v in ORDEM_TABELAS_3_4:
        r = robustez_fin.loc[v]
        linhas3.append([QUANTITATIVAS[v], formatar_br(r['Levene W'], 2), valor_p(r['Levene p']),
                        formatar_br(r['Welch F'], 2), valor_p(r['Welch p']),
                        formatar_br(r['Kruskal-Wallis H'], 2), valor_p(r['Kruskal-Wallis p'])])
    registrar('Tabela_3', pd.DataFrame(linhas3, columns=['Variável', 'Levene W', 'Levene p', 'Welch F', 'Welch p',
                                                             'Kruskal-Wallis H', 'Kruskal-Wallis p']),
              'Tabela 3 | Testes de robustez da comparação de médias entre os clusters')

    # ----- Comparações par a par entre clusters ----------
    pares = list(combinations(clusters, 2))
    linhas_pares = []
    for v in QUANTITATIVAS:
        for a, b in pares:
            x = filmes.loc[filmes['Cluster'] == a, v].values
            y = filmes.loc[filmes['Cluster'] == b, v].values
            p_welch = ttest_ind(x, y, equal_var=False)[1]
            u, p_mw = mannwhitneyu(x, y, alternative='two-sided')
            linhas_pares.append({'Variável': QUANTITATIVAS[v], 'Comparação': f'Cluster {a} vs Cluster {b}',
                                 'Diferença de médias': x.mean() - y.mean(),
                                 'p Welch (Bonferroni)': min(p_welch * len(pares), 1),
                                 'p Mann-Whitney (Bonferroni)': min(p_mw * len(pares), 1),
                                 'Delta de Cliff': 2 * u / (len(x) * len(y)) - 1})
    registrar('Comparacoes_pareadas', pd.DataFrame(linhas_pares),
              'Comparações par a par entre clusters | Welch e Mann-Whitney com correção de Bonferroni',
              fmt={'Diferença de médias': casas(2), 'p Welch (Bonferroni)': p_cientifico,
                   'p Mann-Whitney (Bonferroni)': p_cientifico, 'Delta de Cliff': casas(3)})

    # ----- Tabela 4: comparações pareadas entre clusters --------------------------
    nomes_financeiras_3_4 = [QUANTITATIVAS[v] for v in ORDEM_TABELAS_3_4]
    pareadas_fin = (pd.DataFrame(linhas_pares)
                    .set_index('Variável').loc[nomes_financeiras_3_4].reset_index())
    linhas4 = []
    for _, r in pareadas_fin.iterrows():
        linhas4.append([r['Variável'], r['Comparação'].replace(' vs ', ' × '),
                        formatar_br(r['Diferença de médias'], 2), valor_p(r['p Welch (Bonferroni)']),
                        valor_p(r['p Mann-Whitney (Bonferroni)']), formatar_br(r['Delta de Cliff'], 3)])
    registrar('Tabela_4',
              pd.DataFrame(linhas4, columns=['Variável', 'Comparação',
                                             f'Diferença de médias (US$ de {ANO_BASE_PRECOS})*',
                                             'p de Welch(Bonferroni)', 'p de Mann-Whitney(Bonferroni)', 'Delta de Cliff']),
              'Tabela 4 | Comparações pareadas entre clusters, com correção de Bonferroni')

    # ----- Tabela 5: perfil dos clusters ------------------------------------------
    medias = filmes.groupby('Cluster', observed=False)[
        ['orcamento_de_producao', 'faturamento_bruto_mundial', 'roi', 'presenca_de_estrelas',
         'proporcao_de_novatos_no_elenco_principal', 'extensao_de_marca', 'sazonalidade']].mean()
    indicadores = [
        (f'Orçamento Médio (US$ de {ANO_BASE_PRECOS})', 'orcamento_de_producao', casas(2)),
        (f'Faturamento Médio (US$ de {ANO_BASE_PRECOS})', 'faturamento_bruto_mundial', casas(2)),
        ('ROI Médio', 'roi', casas(2)),
        ('Presença de Estrelas (%)', 'presenca_de_estrelas', fracao_pct()),
        ('Proporção de Novatos no Elenco (%)', 'proporcao_de_novatos_no_elenco_principal', fracao_pct()),
        ('Extensão de Marca (%)', 'extensao_de_marca', fracao_pct()),
        ('Alta Temporada (%)', 'sazonalidade', fracao_pct()),
    ]
    tabela5 = pd.DataFrame({NOMES_CLUSTERS[c]: [fmt(medias.loc[c, col]) for _, col, fmt in indicadores] for c in clusters})
    tabela5.insert(0, 'Indicador de Atração e Performance de Mercado', [nome for nome, _, _ in indicadores])
    registrar('Tabela_5', tabela5, 'Tabela 5 | Perfil dos clusters')

    # ----- Proporções pareadas entre clusters (orçamento, faturamento e ROI) ------
    linhas_proporcoes = []
    for v in ORDEM_TABELAS_3_4:
        for cluster, referencia in permutations(clusters, 2):
            linhas_proporcoes.append({'Variável': QUANTITATIVAS[v], 'Cluster': f'Cluster {cluster}',
                                      'Em relação a': f'Cluster {referencia}',
                                      'Média do Cluster': medias.loc[cluster, v],
                                      'Média do Cluster de Referência': medias.loc[referencia, v],
                                      'Proporção (%)': medias.loc[cluster, v] / medias.loc[referencia, v] * 100})
    registrar('Proporcoes_pareadas', pd.DataFrame(linhas_proporcoes),
              'Proporções pareadas entre clusters | proporção das médias de Orçamento, Faturamento e ROI de cada '
              'cluster em relação a cada um dos outros dois',
              fmt={'Média do Cluster': casas(2), 'Média do Cluster de Referência': casas(2), 'Proporção (%)': pct(2)})

    verificacao_nomes = {
        'Cluster 0 (Eficientes): menor orçamento médio e maior ROI médio':
            medias['orcamento_de_producao'].idxmin() == 0 and medias['roi'].idxmax() == 0,
        'Cluster 1 (Superproduções): maior orçamento médio e maior faturamento médio':
            medias['orcamento_de_producao'].idxmax() == 1 and medias['faturamento_bruto_mundial'].idxmax() == 1,
        'Cluster 2 (Baixo Retorno): menor ROI médio': medias['roi'].idxmin() == 2}
    cabecalho('Verificação dos nomes dos clusters')
    for descricao, confere in verificacao_nomes.items():
        print(f"{'OK      ' if confere else 'REVISAR '} {descricao}")

    # ----- Perfil completo dos clusters (todas as variáveis quantitativas) --------
    def linha_perfil(nome, grupo, serie):
        return {'Variável': nome, 'Grupo': 'Base' if grupo == 'Base' else f'Cluster {grupo}',
                'N': int(serie.count()), 'Média': serie.mean(), 'Mediana': serie.median(),
                'Desvio-padrão': serie.std(), 'Mínimo': serie.min(), 'Máximo': serie.max()}


    sem_outlier_roi = filmes[z_financeiras['roi'].abs() <= LIMITE_OUTLIER]      # ROI sem os filmes com |z| > 3
    linhas_perfil = []
    for v, nome in QUANTITATIVAS.items():
        for grupo in [*clusters, 'Base']:
            linhas_perfil.append(linha_perfil(nome, grupo, filmes[v] if grupo == 'Base'
                                              else filmes.loc[filmes['Cluster'] == grupo, v]))
        if v == 'roi':       # sensibilidade da média do ROI aos filmes de ROI extremo
            for grupo in [*clusters, 'Base']:
                linhas_perfil.append(linha_perfil('ROI sem outliers de ROI (|z| > 3)', grupo, sem_outlier_roi['roi'] if grupo == 'Base'
                                                  else sem_outlier_roi.loc[sem_outlier_roi['Cluster'] == grupo, 'roi']))
    registrar('Perfil_quantitativas', pd.DataFrame(linhas_perfil),
              'Perfil dos clusters | estatísticas descritivas de todas as variáveis quantitativas',
              ajuste=ano_sem_milhar(['Média', 'Mediana', 'Desvio-padrão', 'Mínimo', 'Máximo']))

    # ----- Tamanho dos clusters e localização dos outliers ------------------------
    registrar('Composicao_clusters',
              pd.DataFrame({'Cluster': [NOMES_CLUSTERS[k] for k in clusters], 'Filmes': n_por_cluster.values,
                            '% da base': n_por_cluster.values / len(filmes) * 100}),
              'Composição dos clusters', fmt={'% da base': pct(1)})

    NOMES_CURTOS = {'orcamento_de_producao': 'Orçamento', 'faturamento_bruto_mundial': 'Faturamento', 'roi': 'ROI'}
    criterios = pd.DataFrame({NOMES_CURTOS[v]: z_financeiras[v].abs() > LIMITE_OUTLIER for v in VARIAVEIS_FINANCEIRAS})

    # Contagem por variável e por filme: um mesmo filme pode ser outlier em mais de uma variável
    contagens = {QUANTITATIVAS[v]: criterios[NOMES_CURTOS[v]] for v in VARIAVEIS_FINANCEIRAS}
    contagens.update({'Filmes distintos (orçamento ou faturamento)': criterios[['Orçamento', 'Faturamento']].any(axis=1),
                      'Filmes em ambos (orçamento e faturamento)': criterios[['Orçamento', 'Faturamento']].all(axis=1),
                      'Filmes distintos (qualquer das três variáveis)': criterios.any(axis=1)})
    outliers_por_cluster = pd.DataFrame({nome: serie.groupby(filmes['Cluster'], observed=False).sum()
                                         for nome, serie in contagens.items()}).T
    outliers_por_cluster.columns = [f'Cluster {c}' for c in outliers_por_cluster.columns]
    outliers_por_cluster['Total'] = outliers_por_cluster.sum(axis=1)
    registrar('Outliers_por_cluster', outliers_por_cluster.rename_axis('Contagem').reset_index(),
              'Outliers (|z| > 3) por cluster | por variável e por filme')

    lista_outliers = filmes.loc[criterios.any(axis=1), ['Cluster', 'rank', 'titulo_ingles', 'ano', *VARIAVEIS_FINANCEIRAS]].copy()
    for v in VARIAVEIS_FINANCEIRAS:
        lista_outliers[f'z {QUANTITATIVAS[v]}'] = z_financeiras.loc[lista_outliers.index, v]
    lista_outliers['Critério(s)'] = criterios.loc[lista_outliers.index].apply(lambda l: ', '.join(l.index[l]), axis=1)
    lista_outliers = (lista_outliers.rename(columns={'rank': 'Rank', 'titulo_ingles': 'Título', 'ano': 'Ano',
                                                     **{v: QUANTITATIVAS[v] for v in VARIAVEIS_FINANCEIRAS}})
                      .sort_values(['Cluster', 'Faturamento Mundial Bruto'], ascending=[True, False]))
    registrar('Outliers_lista', lista_outliers, 'Filmes com |z| > 3 em orçamento, faturamento ou ROI',
              fmt={'Orçamento de Produção': casas(0), 'Faturamento Mundial Bruto': casas(0),
                   'Retorno sobre Investimento (ROI)': casas(2), 'z Orçamento de Produção': casas(2),
                   'z Faturamento Mundial Bruto': casas(2), 'z Retorno sobre Investimento (ROI)': casas(2)},
              colunas=['Cluster', 'Rank', 'Título', 'Ano', 'Orçamento de Produção', 'Faturamento Mundial Bruto',
                       'z Orçamento de Produção', 'z Faturamento Mundial Bruto', 'Critério(s)'])

    # =============================================================================
    # 6. CRUZAMENTO QUALITATIVO (QUI-QUADRADO E ACM)
    # =============================================================================

    df_quali = df_quali_base.copy()
    df_quali.insert(0, 'Cluster', 'Cluster_' + filmes['Cluster'].astype(str))

    # ----- Qui-Quadrado de independência (Cluster x variável qualitativa) ---------
    resumo_qui, detalhe_qui = testar_associacoes(rotulos_finais, df_quali_base)
    extras = pd.DataFrame({'Mes': filmes['mes_de_lancamento'], 'Pais': filmes['pais_de_origem']})

    resumo_extras, detalhe_extras = testar_associacoes(rotulos_finais, extras, n_familia=1)

    qui = resumo_qui.set_index('Código')
    generos_ordenados = qui.loc[[c for c in qui.index if c not in variaveis_fixas]].sort_values('χ²', ascending=False).index.tolist()
    ordem_tabela_6 = variaveis_fixas + generos_ordenados

    # ----- Tabela 6: resumo dos testes Qui-Quadrado -------------------------------
    ordem_tabela_6_tcc = qui.sort_values('χ²', ascending=False).index.tolist()
    t6 = qui.loc[ordem_tabela_6_tcc]
    diagnostico_unico = np.select(
        [t6['Significativo'] & t6['Significativo (Bonferroni)'], t6['Significativo']],
        ['Significativo (nominal e Bonferroni).', 'Significativo apenas no nível nominal.'],
        default='Não significativo.')
    registrar('Tabela_6',
              pd.DataFrame({'Atributo Qualitativo Nominal': t6['Variável'], 'GL': t6['GL'],
                            'χ² calculado': t6['χ²'].map(casas(3)),
                            'Valor-p': t6['Valor-p'].map(casas(4)) + t6['Marca'],
                            'Valor-p (Bonferroni)': t6['Valor-p (Bonferroni)'].map(casas(4)),
                            'Diagnóstico': diagnostico_unico}),
              'Tabela 6 | Resumo dos testes de associação Qui-Quadrado de independência (χ²) entre os perfis e atributos')
    print("* frequência esperada mínima entre 1 e 5 | ** frequência esperada mínima < 1")

    # ----- Diagnóstico completo do Qui-Quadrado (inclui mês e país, fora do modelo) -
    diagnostico = pd.concat([resumo_qui.assign(**{'Incluída na ACM': 'sim'}).set_index('Código').loc[ordem_tabela_6].reset_index(),
                             resumo_extras.assign(**{'Incluída na ACM': 'não'})], ignore_index=True)
    registrar('Qui2_diagnostico', diagnostico,
              'Diagnóstico do Qui-Quadrado | tamanho de efeito e frequências esperadas (todas as variáveis)',
              fmt={'V de Cramér': casas(3), 'Freq. esperada mínima': casas(2), '% células ≥ 5': pct(1)},
              colunas=['Variável', 'V de Cramér', 'Freq. esperada mínima', 'Célula da mínima', 'Células < 5',
                       'Células < 1', '% células ≥ 5', 'Incluída na ACM'])

    # ----- Sensibilidade da sazonalidade a meses com duas datas -------------------
    if not meses_fora_padrao.empty:
        linhas_sensibilidade = []
        for descricao, coluna in [('Como registrada no banco (usada nas análises)', 'sazonalidade'),
                                  ('Com o mês do ano de lançamento', 'sazonalidade_alternativa')]:
            flag = filmes[coluna]
            r, _ = testar_associacoes(rotulos_finais, pd.DataFrame({'Sazonalidade': np.where(flag == 1, 'Alta_Temporada', 'Temporada_Regular')}))
            por_cluster = flag.groupby(filmes['Cluster'], observed=False).mean() * 100
            linhas_sensibilidade.append({'Classificação': descricao, 'Filmes em alta temporada': int(flag.sum()),
                                         **{f'% alta temporada Cluster {c}': por_cluster[c] for c in clusters},
                                         'χ²': r.loc[0, 'χ²'], 'Valor-p': r.loc[0, 'Valor-p']})
        registrar('Sazonalidade_sensibilidade', pd.DataFrame(linhas_sensibilidade),
                  'Sensibilidade da sazonalidade aos meses com duas datas',
                  fmt={**{f'% alta temporada Cluster {c}': pct(2) for c in clusters}, 'χ²': casas(3), 'Valor-p': casas(4)})

    # ----- Composição das categorias por cluster ----------------------------------
    categorias_cluster = composicao_categorias(detalhe_qui, n_por_cluster)
    formatos_composicao = {'% base': pct(1), **{f'% Cluster {c}': pct(1) for c in clusters}}

    # Tabela 7: gêneros com associação significativa (p < 0,05), na ordem do TCC
    significativos = [c for c in generos_ordenados if qui.loc[c, 'Significativo']]
    significativos.sort(key=lambda c: ORDEM_TABELA_7.index(c) if c in ORDEM_TABELA_7 else len(ORDEM_TABELA_7))
    linhas7 = []
    for c in significativos:
        r = categorias_cluster[(categorias_cluster['Variável'] == NOMES_QUALITATIVAS[c])
                               & (categorias_cluster['Categoria'] == ROTULO_SIM[c])].iloc[0]
        linhas7.append([NOMES_QUALITATIVAS[c]] + [pct(1)(r[f'% Cluster {k}']) for k in clusters] + [pct(1)(r['% base'])])
    registrar('Tabela_7', pd.DataFrame(linhas7, columns=['Gênero'] + [f'Cluster {k}' for k in clusters] + ['Base']),
              'Tabela 7 | % de filmes por gênero e cluster (gêneros com p < 0,05)')

    categorias_extras = composicao_categorias(detalhe_extras, n_por_cluster)
    categorias_extras['ordem'] = [ORDEM_MESES.index(c) if v == NOMES_EXTRAS['Mes'] and c in ORDEM_MESES
                                  else (99 if v == NOMES_EXTRAS['Mes'] else -n)
                                  for v, c, n in zip(categorias_extras['Variável'], categorias_extras['Categoria'],
                                                     categorias_extras['N base'])]
    categorias_extras = (categorias_extras.sort_values(['Variável', 'ordem']).drop(columns='ordem'))

    categorias_todas = pd.concat([categorias_cluster.assign(**{'Incluída na ACM': 'sim'}),
                                  categorias_extras.assign(**{'Incluída na ACM': 'não'})], ignore_index=True)
    registrar('Categorias_por_cluster', categorias_todas,
              'Composição de todas as categorias por cluster | nº e % de filmes (inclui mês e país, fora do modelo)',
              fmt=formatos_composicao)

    # ----- Direção das associações: resíduos padronizados ajustados ---------------
    residuos = (detalhe_qui[detalhe_qui['Categoria'] == detalhe_qui['Código'].map(ROTULO_SIM)]
                .pivot(index='Código', columns='Cluster', values='Resíduo ajustado').loc[ordem_tabela_6])
    residuos.columns = [f'Cluster {c}' for c in residuos.columns]
    tabelas['Residuos_ajustados'] = ('Resíduos padronizados ajustados da categoria "sim" (valores numéricos)',
                                     residuos.rename(index=NOMES_QUALITATIVAS).rename_axis('Variável').reset_index())
    mostrar('Direção das associações | resíduo ajustado da categoria "sim" (* = |resíduo| > 1,96)',
            residuos.apply(lambda col: col.map(lambda r: com_sinal(2)(r) + ('*' if abs(r) > LIMITE_RESIDUO else '')))
            .rename(index=NOMES_QUALITATIVAS).rename_axis('Variável').reset_index())
    print(f"Resíduos sem correção para comparações múltiplas ({residuos.size} células): "
          f"cerca de {residuos.size * ALFA:.0f} marcações ao acaso são esperadas.")

    # ----- ACM: inércia explicada por dimensão ------------------------------------
    mca = prince.MCA(n_components=3, random_state=SEMENTE).fit(df_quali)

    n_categorias = sum(df_quali[c].nunique() for c in df_quali.columns)
    max_dim = n_categorias - df_quali.shape[1]
    mca_bruta = prince.MCA(n_components=max_dim, random_state=SEMENTE).fit(df_quali)
    mca_ajustada = prince.MCA(n_components=max_dim, random_state=SEMENTE, correction='benzecri').fit(df_quali)


    def inercia_por_dimensao(acumulada, n):
        acumulada = np.asarray(acumulada, dtype=float)[:n]
        por_dimensao = np.diff(np.r_[0, acumulada])
        falta = n - len(acumulada)
        return np.r_[por_dimensao, [np.nan] * falta], np.r_[acumulada, [np.nan] * falta]


    n_dim = min(len(np.asarray(mca_bruta.cumulative_percentage_of_variance_)), max_dim)
    bruta, bruta_acum = inercia_por_dimensao(mca_bruta.cumulative_percentage_of_variance_, n_dim)
    ajustada, ajustada_acum = inercia_por_dimensao(mca_ajustada.cumulative_percentage_of_variance_, n_dim)
    registrar('ACM_inercia',
              pd.DataFrame({'Dimensão': range(1, n_dim + 1), '% bruta': bruta, '% bruta acumulada': bruta_acum,
                            '% ajustada (Benzécri)': ajustada, '% ajustada acumulada': ajustada_acum,
                            'No mapa': np.where(np.arange(1, n_dim + 1) <= 3, 'sim', '')}),
              'ACM | inércia explicada por dimensão (as três primeiras compõem o mapa)',
              fmt={c: casas(2) for c in ['% bruta', '% bruta acumulada', '% ajustada (Benzécri)', '% ajustada acumulada']})

    # ----- ACM: coordenadas, distâncias e rótulos do mapa -------------------------
    coordenadas = mca.column_coordinates(df_quali).reset_index()

    partes = coordenadas['index'].str.split('__', n=1, expand=True)
    chart_df_mca = pd.DataFrame({
        'categoria': [rotulo_categoria(v, c) for v, c in zip(partes[0], partes[1])],
        'Dimensao_1': coordenadas[0],
        'Dimensao_2': coordenadas[1],
        'Dimensao_3': coordenadas[2],
        'Variavel_Origem': partes[0].map(NOMES_MAPA),
    })

    dimensoes = ['Dimensao_1', 'Dimensao_2', 'Dimensao_3']
    coords = chart_df_mca.set_index('categoria')[dimensoes]
    clusters_mapa = [c for c in coords.index if c.startswith('Cluster')]
    dist_clusters = pd.DataFrame({f'Dist. {c}': np.linalg.norm(coords.values - coords.loc[c].values, axis=1)
                                  for c in clusters_mapa}, index=coords.index)
    n_categoria = {**dict(zip(categorias_cluster['Categoria'], categorias_cluster['N base'])),
                   **{f'Cluster {k}': q for k, q in n_por_cluster.items()}}
    mapa = pd.DataFrame({'Tipo': ['Cluster' if c in clusters_mapa else 'Categoria' for c in coords.index],
                         'Variável': chart_df_mca['Variavel_Origem'].values, 'Categoria': coords.index,
                         'N': [n_categoria[c] for c in coords.index]})
    mapa['% da base'] = mapa['N'] / len(filmes) * 100
    for i, nome_dimensao in enumerate(['Dimensão 1', 'Dimensão 2', 'Dimensão 3']):
        mapa[nome_dimensao] = coords.values[:, i]
    mapa['Distância à origem'] = np.linalg.norm(coords.values, axis=1)
    mapa = pd.concat([mapa, dist_clusters.reset_index(drop=True)], axis=1)
    mapa['Cluster mais próximo'] = np.where(mapa['Tipo'] == 'Categoria', dist_clusters.idxmin(axis=1).str.replace('Dist. ', '', regex=False).values, '')
    mapa['Dist. ao mais próximo'] = np.where(mapa['Tipo'] == 'Categoria', dist_clusters.min(axis=1).values, np.nan)
    mapa = mapa.sort_values('Distância à origem', ascending=False)
    tabelas['ACM_mapa_categorias'] = ('ACM | coordenadas e distâncias das categorias no mapa (3 dimensões)', mapa.reset_index(drop=True))
    mostrar('ACM | posição das categorias no mapa (distância à origem e cluster mais próximo)',
            formatar_df(mapa[mapa['Tipo'] == 'Categoria'], {'% da base': pct(1), 'Dist. ao mais próximo': casas(2),
                                                            'Distância à origem': casas(2)}),
            ['Categoria', 'N', '% da base', 'Distância à origem', 'Cluster mais próximo', 'Dist. ao mais próximo'])

    # ----- Filmes por cluster (registro completo da base) -------------------------
    generos_do_filme = filmes[list(generos_avaliados)].apply(
        lambda l: ', '.join(nome for coluna, (_, nome) in GENEROS.items() if l[coluna] == 1), axis=1)
    registrar('Filmes_por_cluster',
              pd.DataFrame({'Cluster': filmes['Cluster'], 'Rank': filmes['rank'], 'Título': filmes['titulo_ingles'],
                            'Ano': filmes['ano'], 'Mês': filmes['mes_de_lancamento'], 'País': filmes['pais_de_origem'],
                            f'Orçamento (US$ de {ANO_BASE_PRECOS})': filmes['orcamento_de_producao'],
                            f'Faturamento (US$ de {ANO_BASE_PRECOS})': filmes['faturamento_bruto_mundial'],
                            'Orçamento nominal (US$)': filmes['orcamento_nominal'],
                            'Faturamento nominal (US$)': filmes['faturamento_nominal'],
                            'Fator de correção': filmes['fator_inflacao'], 'ROI': filmes['roi'],
                            'Presença de estrelas': filmes['presenca_de_estrelas'].astype(int),
                            'Extensão de marca': filmes['extensao_de_marca'].astype(int),
                            'Alta temporada': filmes['sazonalidade'],
                            'Proporção de novatos': filmes['proporcao_de_novatos_no_elenco_principal'],
                            'Gêneros': generos_do_filme,
                            'Outlier (|z| > 3)': criterios.any(axis=1).map({True: 'sim', False: ''})
                            }).sort_values(['Cluster', f'Faturamento (US$ de {ANO_BASE_PRECOS})'], ascending=[True, False]),
              'Todos os filmes com o cluster atribuído', imprimir=False)

    # ----- Mapa perceptual interativo (Figura 4) ----------------------------------
    fig_mapa = px.scatter_3d(
        chart_df_mca,
        x='Dimensao_1',
        y='Dimensao_2',
        z='Dimensao_3',
        color='Variavel_Origem',
        text='categoria',
        labels={
            'Dimensao_1': 'Dimensão 1',
            'Dimensao_2': 'Dimensão 2',
            'Dimensao_3': 'Dimensão 3',
            'Variavel_Origem': 'Variável',
            'categoria': 'Categoria',
        }
    )
    fig_mapa.update_traces(textposition='top center', marker=dict(size=5))
    fig_mapa.update_layout(
        paper_bgcolor='white',
        plot_bgcolor='white',
        scene=dict(
            xaxis=dict(showbackground=False, showgrid=False, zeroline=False),
            yaxis=dict(showbackground=False, showgrid=False, zeroline=False),
            zaxis=dict(showbackground=False, showgrid=False, zeroline=False),
        )
    )
    fig_mapa.write_html(ARQUIVO_MAPA)

    # =============================================================================
    # 7. COMPARAÇÃO COM A VERSÃO NOMINAL (O QUE MUDOU COM A CORREÇÃO)
    # =================================================================f============
    # Os resultados principais acima usam dólares constantes. Aqui o mesmo pipeline (mesmos parâmetros e
    # semente) é aplicado aos valores nominais apenas para comparação.

    # ----- Escolha de K: valores nominais x constantes ----------------------------
    modelos_nominais = {k: KMeans(n_clusters=k, init='k-means++', random_state=SEMENTE, n_init=N_INIT).fit(z_nominal)
                        for k in k_elbow}
    silhueta_nominal = {k: silhouette_score(z_nominal, modelos_nominais[k].labels_) for k in k_silhueta}
    linhas_cmp = []
    for k in k_elbow:
        tam_nominal = sorted(np.bincount(modelos_nominais[k].labels_).tolist(), reverse=True)
        tam_constante = sorted(np.bincount(obter_kmeans(k).labels_).tolist(), reverse=True)
        linhas_cmp.append({'K': k,
                           'WCSS (Inércia) nominal': modelos_nominais[k].inertia_, 'WCSS (Inércia) constante': wcss[k],
                           'Silhueta nominal': silhueta_nominal.get(k, np.nan), 'Silhueta constante': silhueta.get(k, np.nan),
                           'Tamanhos nominal': str(tam_nominal), 'Tamanhos constante': str(tam_constante)})
    registrar('Comparacao_K', pd.DataFrame(linhas_cmp), 'Escolha de K | versão nominal x dólares constantes',
              fmt={'Silhueta nominal': casas(4), 'Silhueta constante': casas(4),
                   'WCSS (Inércia) nominal': casas(2), 'WCSS (Inércia) constante': casas(2)})
    print(f"Melhor K pela silhueta (K >= 2): nominal = {max(silhueta_nominal, key=silhueta_nominal.get)} | "
          f"constante = {max(silhueta, key=silhueta.get)}")

    # ----- Partição (K escolhido): quem mudou de cluster --------------------------
    cruzamento = pd.crosstab(pd.Series(rotulos_nominais, name='n'), pd.Series(rotulos_finais, name='c')).reindex(
        index=clusters, columns=clusters, fill_value=0)
    trocaram = int(len(filmes) - np.trace(cruzamento.values))
    ari = adjusted_rand_score(rotulos_nominais, rotulos_finais)
    print(f"Índice de Rand ajustado = {formatar_br(ari, 3)} | "
          f"filmes que mudaram de cluster: {trocaram} de {len(filmes)} ({formatar_br(trocaram / len(filmes) * 100, 1)}%)")

    cruzamento_tabela = cruzamento.rename(index=lambda k: f'Nominal: Cluster {k}',
                                          columns=lambda k: f'Constante: Cluster {k}')
    cruzamento_tabela['Total'] = cruzamento_tabela.sum(axis=1)
    cruzamento_tabela.loc['Total'] = cruzamento_tabela.sum(axis=0)
    cruzamento_tabela = cruzamento_tabela.rename_axis('Partição nominal').reset_index()

    colunas_particao = cruzamento_tabela.columns.tolist()
    linha_em_branco_particao = pd.DataFrame([{c: '' for c in colunas_particao}])
    resumo_particao = pd.DataFrame([
        {'Partição nominal': 'Índice de Rand Ajustado (ARI)', 'Constante: Cluster 0': formatar_br(ari, 3)},
        {'Partição nominal': 'Filmes que mudaram de cluster', 'Constante: Cluster 0': trocaram},
        {'Partição nominal': 'Total de filmes', 'Constante: Cluster 0': len(filmes)},
        {'Partição nominal': '% que mudaram de cluster', 'Constante: Cluster 0': pct(1)(trocaram / len(filmes) * 100)},
    ]).reindex(columns=colunas_particao, fill_value='')
    tabela_particao_unida = pd.concat([cruzamento_tabela, linha_em_branco_particao, resumo_particao], ignore_index=True)
    registrar('Comparacao_particao', tabela_particao_unida,
              'Partição nominal x partição em dólares constantes (rótulos alinhados), com totais por cluster, '
              'Índice de Rand Ajustado e filmes que mudaram de cluster')

    mudaram = filmes[rotulos_nominais != rotulos_finais]
    registrar('Filmes_que_mudaram',
              pd.DataFrame({'Rank': mudaram['rank'], 'Título': mudaram['titulo_ingles'], 'Ano': mudaram['ano'].astype(int),
                            'Cluster nominal': rotulos_nominais[mudaram.index], 'Cluster constante': rotulos_finais[mudaram.index],
                            'Orçamento nominal (US$)': mudaram['orcamento_nominal'],
                            f'Orçamento (US$ de {ANO_BASE_PRECOS})': mudaram['orcamento_de_producao'],
                            'Faturamento nominal (US$)': mudaram['faturamento_nominal'],
                            f'Faturamento (US$ de {ANO_BASE_PRECOS})': mudaram['faturamento_bruto_mundial']}
                           ).sort_values(['Cluster nominal', 'Cluster constante', 'Ano']),
              'Filmes que trocaram de cluster com a correção pela inflação', imprimir=False)

    # ----- ANOVA e Qui-Quadrado: valores nominais x constantes --------------------
    base_nominal = pd.DataFrame({'orcamento_de_producao': filmes['orcamento_nominal'],
                                 'faturamento_bruto_mundial': filmes['faturamento_nominal'],
                                 'roi': filmes['roi'], 'ano': filmes['ano']})
    variaveis_cmp = ['orcamento_de_producao', 'faturamento_bruto_mundial', 'roi', 'ano']
    anova_nominal = anova_por_cluster(rotulos_nominais, variaveis_cmp, base=base_nominal).set_index('Código')
    anova_constante = anova_por_cluster(rotulos_finais, variaveis_cmp).set_index('Código')
    registrar('Comparacao_ANOVA',
              pd.DataFrame({'Variável': anova_nominal['Variável'],
                            'F nominal': anova_nominal['F'], 'F constante': anova_constante['F'],
                            'p nominal': anova_nominal['Valor-p'], 'p constante': anova_constante['Valor-p'],
                            'η² parcial nominal': anova_nominal['η² parcial'],
                            'η² parcial constante': anova_constante['η² parcial']}).reset_index(drop=True),
              'ANOVA (K escolhido) | partição nominal com valores nominais x partição constante com valores constantes',
              fmt={'F nominal': casas(2), 'F constante': casas(2), 'p nominal': p_cientifico, 'p constante': p_cientifico,
                   'η² parcial nominal': casas(4), 'η² parcial constante': casas(4)})

    resumo_nominal, _ = testar_associacoes(rotulos_nominais, df_quali_base)
    resumo_nominal = resumo_nominal.set_index('Código').loc[ordem_tabela_6]
    resumo_constante = resumo_qui.set_index('Código').loc[ordem_tabela_6]
    cmp_qui = pd.DataFrame({'Variável': resumo_nominal['Variável'],
                            'χ² nominal': resumo_nominal['χ²'], 'p nominal': resumo_nominal['Valor-p'],
                            'χ² constante': resumo_constante['χ²'], 'p constante': resumo_constante['Valor-p'],
                            'Significativo nominal': resumo_nominal['Significativo'],
                            'Significativo constante': resumo_constante['Significativo']}).reset_index(drop=True)
    cmp_qui['Mudou de conclusão'] = np.where(cmp_qui['Significativo nominal'] != cmp_qui['Significativo constante'], 'sim', '')
    registrar('Comparacao_Qui2', cmp_qui, 'Qui-Quadrado | versão nominal x dólares constantes',
              fmt={'χ² nominal': casas(3), 'χ² constante': casas(3), 'p nominal': casas(4), 'p constante': casas(4)})

    # ----- ACM: inércia das três dimensões do mapa, nominal x constante -----------
    df_quali_nominal = df_quali_base.copy()
    df_quali_nominal.insert(0, 'Cluster', 'Cluster_' + pd.Series(rotulos_nominais).astype(str))
    mca_bruta_nominal = prince.MCA(n_components=max_dim, random_state=SEMENTE).fit(df_quali_nominal)
    mca_ajustada_nominal = prince.MCA(n_components=max_dim, random_state=SEMENTE, correction='benzecri').fit(df_quali_nominal)
    registrar('Comparacao_ACM',
              pd.DataFrame({'Dimensão': [1, 2, 3],
                            '% bruta nominal': inercia_por_dimensao(mca_bruta_nominal.cumulative_percentage_of_variance_, 3)[0],
                            '% bruta constante': bruta[:3],
                            '% ajustada nominal': inercia_por_dimensao(mca_ajustada_nominal.cumulative_percentage_of_variance_, 3)[0],
                            '% ajustada constante': ajustada[:3]}),
              'ACM | inércia das três dimensões do mapa, nominal x constante',
              fmt={c: casas(2) for c in ['% bruta nominal', '% bruta constante', '% ajustada nominal', '% ajustada constante']})

    exportar_tabelas(ARQUIVO_RESULTADOS)

    print(f"\n[Sucesso] Mapa perceptual tridimensional exportado com sucesso para o arquivo '{ARQUIVO_MAPA}'.")
    print("Abra-o no seu navegador para explorar os agrupamentos!")

    # =============================================================================
    # Fim do Script!
    # =============================================================================


if __name__ == "__main__":
    main()
