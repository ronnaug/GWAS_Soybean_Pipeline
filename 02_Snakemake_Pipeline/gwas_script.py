import sys
import pandas as pd
import numpy as np
import allel
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import statsmodels.api as sm

# ------------------
# 1. Параметры и Загрузка Данных
# -----------------

if len(sys.argv) < 4:
    print("Использование: python gwas_script.py <VCF_FILE> <PHENOTYPE_FILE> <OUTPUT_PREFIX>")
    sys.exit(1)

VCF_FILE = sys.argv[1]    
PHENOTYPE_FILE = sys.argv[2]
OUTPUT_PREFIX = sys.argv[3] 

# Чтение VCF
try:
    callset = allel.read_vcf(VCF_FILE)
except Exception as e:
    print(f"Ошибка чтения VCF {VCF_FILE}: {e}")
    sys.exit(1)
    
sample_ids = callset['samples']
genotypes = allel.GenotypeArray(callset['calldata/GT'])
G = genotypes.to_n_alt().T

# Создание исходного DataFrame для SNP-позиций
snp_positions = pd.DataFrame({
    'CHR': callset['variants/CHROM'].astype(str),
    'BP': callset['variants/POS']
})

# Заполнение пропущенных данных (импутация средним)
missing_mask = np.isnan(G)
col_means = np.nanmean(G, axis=0)
G_imputed = G.copy()
G_imputed[missing_mask] = np.take(col_means, np.where(missing_mask)[1])
G_imputed = np.nan_to_num(G_imputed, nan=0)

# Загрузка фенотипов
pheno_df = pd.read_csv(PHENOTYPE_FILE, sep='\t', header=None, names=['IID', 'Yield'])
pheno_df['Yield'] = pd.to_numeric(pheno_df['Yield'], errors='coerce')

# -----------------
# 2. PCA и Синхронизация
# -----------------

# Запуск PCA
pca = PCA(n_components=10)
X_pca = pca.fit_transform(G_imputed)
pca_df = pd.DataFrame(X_pca, index=sample_ids, columns=[f'PC{i}' for i in range(1, 11)])

# Объединение и удаление NaN
merged_df = pca_df.merge(pheno_df, left_index=True, right_on='IID')
merged_df.dropna(subset=['Yield'], inplace=True)

# Синхронизация генотипов
valid_ids = merged_df['IID'].tolist()
sample_ids_list = sample_ids.tolist()
valid_indices = [sample_ids_list.index(iid) for iid in valid_ids if iid in sample_ids_list]
G_final = G_imputed[valid_indices, :]

# Определение финальных массивов
Y = merged_df['Yield'].values
P = sm.add_constant(merged_df[['PC1', 'PC2', 'PC3', 'PC4', 'PC5']].values)
num_snp = G_final.shape[1]

print(f" Синхронизация завершена. Для GWAS используется {merged_df.shape[0]} образцов и {num_snp} SNP.")

# -----------------
# 3. GWAS (OLS с PCA)
# -----------------

p_values = np.empty(num_snp)

for i in range(num_snp):
    X_snp = G_final[:, i].reshape(-1, 1)
    X = np.hstack((P, X_snp))
    
    model = sm.OLS(Y, X, missing='drop')
    results = model.fit()
    
    p_values[i] = results.pvalues[-1]

# Создание итогового DataFrame
gwas_res = snp_positions.copy()
gwas_res['P'] = p_values
gwas_res['SNP'] = [f'SNP_{i}' for i in range(num_snp)]

# Сохранение результатов GWAS
gwas_res.to_csv(f'{OUTPUT_PREFIX}.gwas.csv', index=False)
print(f" Результаты GWAS сохранены в {OUTPUT_PREFIX}.gwas.csv")

# -----------------
# 4. Визуализация (Сохранение файлов PNG)
# -----------------

# --- 4.1. Manhattan Plot ---
def manhattan_plot(df, title, filename):
    df['log_P'] = -np.log10(df['P'])
    df['CHR'] = pd.to_numeric(df['CHR'], errors='coerce')
    df.dropna(subset=['CHR'], inplace=True)
    df.sort_values(['CHR', 'BP'], inplace=True)
    df['ind'] = range(len(df))
    
    bonf_threshold = -np.log10(0.05 / len(df))
    
    fig, ax = plt.subplots(figsize=(16, 6))
    
    for i, (name, group) in enumerate(df.groupby('CHR')):
        group.plot(kind='scatter', x='ind', y='log_P', color=['#588d92', '#86b0b4'][i % 2], ax=ax, s=50, alpha=0.8)
    
    ax.set_xticks(df.groupby('CHR')['ind'].mean())
    ax.set_xticklabels(df.CHR.unique(), rotation=45)
    ax.set_xlabel('Chromosome')
    ax.axhline(y=bonf_threshold, color='red', linestyle='--', linewidth=1.5, label=f'Bonferroni ({bonf_threshold:.2f})')
    
    ax.set_title(title)
    ax.set_ylim(0, df['log_P'].max() * 1.1)
    ax.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

manhattan_plot(gwas_res.copy(), 'Manhattan Plot for Yield (GWAS-OLS with PCA)', f'{OUTPUT_PREFIX}.manhattan.png')
print(f" Manhattan Plot сохранен в {OUTPUT_PREFIX}.manhattan.png")


# --- 4.2. QQ Plot ---
def qq_plot(pvalues, title, filename):
    pvalues = pvalues[~np.isnan(pvalues)]
    pvalues = np.sort(pvalues)
    
    observed = -np.log10(pvalues)
    
    n = len(pvalues)
    # Используем формулу для ожидаемого распределения p-значений
    expected = -np.log10(np.arange(1, n + 1) / n)
    
    plt.figure(figsize=(8, 8))
    plt.scatter(expected, observed, s=20)
    
    max_val = max(observed.max(), expected.max())
    plt.plot([0, max_val], [0, max_val], color='red', linestyle='--')
    
    plt.xlabel('Expected -log10(P-value)')
    plt.ylabel('Observed -log10(P-value)')
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(filename)
    plt.close()

qq_plot(gwas_res['P'].values, 'QQ Plot for Yield (GWAS-OLS with PCA)', f'{OUTPUT_PREFIX}.qq.png')
print(f" QQ Plot сохранен в {OUTPUT_PREFIX}.qq.png")


# --- 5. Сохранение PC-координат ---
merged_df[['IID'] + [f'PC{i}' for i in range(1, 6)]].to_csv(f'{OUTPUT_PREFIX}.pca.csv', index=False)