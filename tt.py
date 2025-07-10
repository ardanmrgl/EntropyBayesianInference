import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
from lifelines import KaplanMeierFitter
import seaborn as sns

from scipy.stats import spearmanr

from lifelines.statistics import multivariate_logrank_test
from statsmodels.formula.api import mixedlm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import powerlaw

# 1. VERİ YÜKLEME VE ÖN İŞLEME
def load_and_preprocess(file_path):
    """Veriyi yükleyip ön işlemleri uygular"""
    data = pd.read_csv(file_path, encoding='utf-8', sep=';', low_memory=False)
    
    # Tarih dönüşümleri
    data['Cig Tartim Tarihi'] = pd.to_datetime(data['Cig Tartim Tarihi'], format='%Y-%m-%d-%H.%M.%S', errors='coerce')
    data['Ym Tartim Tarihi'] = pd.to_datetime(data['Ym Tartim Tarihi'], format='%Y-%m-%d-%H.%M.%S', errors='coerce')
    
    # Eksik tarihleri temizle
    data = data.dropna(subset=['Cig Tartim Tarihi', 'Ym Tartim Tarihi'])
    
    # Süre hesaplama (dakika cinsinden)
    data['Gecis_Suresi_dk'] = (data['Ym Tartim Tarihi'] - data['Cig Tartim Tarihi']).dt.total_seconds() / 60
    
    # Vardiya belirleme
    data['Cig_Saat'] = data['Cig Tartim Tarihi'].dt.hour
    data['Vardiya'] = pd.cut(data['Cig_Saat'],
                            bins=[0, 8, 16, 24],
                            labels=['00:00-08:00', '08:00-16:00', '16:00-00:00'],
                            right=False)
    
    # Ürün kimliği oluşturma
    data['Urun_Kimligi'] = data['Cig Stok Kodu'].astype(str) + '_' + data['Ym Stok Kodu'].astype(str)
    
    # Ürün grubu oluşturma
    data['Urun_Grubu'] = pd.cut(data['Cig Stok Kodu'],
                               bins=[0, 1000, 2000, 3000, 4000, 5000, np.inf],
                               labels=['Grup1', 'Grup2', 'Grup3', 'Grup4', 'Grup5', 'Grup6'])
    
    return data

# 2. ÜRÜN BAZLI TAHMİN ARALIKLARI
def urun_bazli_tahmin_araliklari(data):
    """Ürün gruplarına göre tahmin aralıkları oluşturur"""
    results = []
    
    # Global vardiya ortalamaları
    global_vardiya = data.groupby('Vardiya', observed=True)['Gecis_Suresi_dk'].agg(['mean', 'std'])
    
    for (cig_kodu, vardiya), group in data.groupby(['Cig Stok Kodu', 'Vardiya'], observed=True):
        if len(group) >= 3:
            mean = group['Gecis_Suresi_dk'].mean()
            std = group['Gecis_Suresi_dk'].std()
            n = len(group)
            t_value = stats.t.ppf(0.95, df=n-1)  # %90 güven aralığı
            margin = t_value * (std / np.sqrt(n))
            metod = 'Ürün-Vardiya Özel'
        else:
            mean = global_vardiya.loc[vardiya, 'mean']
            margin = 1.5 * 60  # 1.5 saatlik sabit güven aralığı
            metod = 'Global Vardiya Ort.'
        
        results.append({
            'Cig Stok Kodu': cig_kodu,
            'Vardiya': vardiya,
            'Ortalama_Sure_saat': mean / 60,
            'Tahmin_Alt_saat': (mean - margin) / 60,
            'Tahmin_Ust_saat': (mean + margin) / 60,
            'Metod': metod,
            'Ornek_Sayisi': len(group)
        })
    
    return pd.DataFrame(results)

# 3. TEMEL ANALİZLER
def basic_analyses(data):
    """Temel istatistiksel analizleri gerçekleştirir"""
    # Vardiya bazlı istatistikler
    vardiya_stats = data.groupby('Vardiya', observed=True).agg(
        Ortalama=('Gecis_Suresi_dk', 'mean'),
        Std=('Gecis_Suresi_dk', 'std'),
        Sayi=('Gecis_Suresi_dk', 'count')
    )
    vardiya_stats['CI_95'] = 1.96 * vardiya_stats['Std'] / np.sqrt(vardiya_stats['Sayi'])
    
    # ANOVA testi
    anova_result = stats.f_oneway(
        *[group['Gecis_Suresi_dk'] for _, group in data.groupby('Vardiya', observed=True)]
    )
    
    # Tukey HSD testi
    tukey_result = pairwise_tukeyhsd(
        endog=data['Gecis_Suresi_dk'],
        groups=data['Vardiya'],
        alpha=0.05
    )
    
    return vardiya_stats, anova_result, tukey_result

#SURVIVAL ANALİZİ
def perform_survival_analysis(data):
    
    plt.figure(figsize=(12, 7))
    ax = plt.subplot(111)
    
    palette = {"00:00-08:00": "#1f77b4", "08:00-16:00": "#ff7f0e", "16:00-00:00": "#2ca02c"}
    
    for vardiya in data['Vardiya'].cat.categories:
        kmf = KaplanMeierFitter()
        kmf.fit(data[data['Vardiya'] == vardiya]['Gecis_Suresi_dk'], 
                label=vardiya)
        kmf.plot_survival_function(ax=ax, ci_show=True, color=palette[vardiya])
    
    plt.title('Fermentasyon Tamamlanma Olasılığı', fontsize=14)
    plt.xlabel('Zaman (Saat)')
    plt.ylabel('Tamamlanmamış Oran')
    plt.grid(True, alpha=0.3)
    
    logrank = multivariate_logrank_test(data['Gecis_Suresi_dk'], data['Vardiya'])
    return logrank.p_value, plt.gcf()

#KARIŞIK ETKİLER MODELİ
def run_mixed_effects_model(data):
    """Karışık etkiler modelini çalıştırır"""
    # Eksik verileri temizle
    data_clean = data.dropna(subset=['Gecis_Suresi_dk', 'Vardiya', 'Cig Stok Kodu'])
    
    try:
        model = mixedlm("Gecis_Suresi_dk ~ C(Vardiya)", 
                       data=data_clean,
                       groups=data_clean["Cig Stok Kodu"],
                       re_formula="~C(Vardiya)")
        return model.fit(method='powell')
    except Exception as e:
        print(f"Model hatası: {str(e)}")
        return None

# 6. GÖRSELLEŞTİRME
def create_main_plot(data, desktop_path):
    """Ana karşılaştırma grafiğini oluşturur"""
    plt.figure(figsize=(12, 7))
    palette = {"00:00-08:00": "#1f77b4", "08:00-16:00": "#ff7f0e", "16:00-00:00": "#2ca02c"}

    ax = sns.barplot(
        x='Vardiya', 
        y='Gecis_Suresi_dk', 
        hue='Vardiya', 
        data=data,
        estimator=np.mean, 
        errorbar=None, 
        palette=palette, 
        legend=False
)
                    
    
    plt.title('Vardiyalara Göre Ortalama Süreler', fontsize=14)
    plt.ylabel('Ortalama Süre (Saat)')
    plt.xticks(rotation=45)
    
    # İstatistiksel işaretler
    max_y = data['Gecis_Suresi_dk'].max()
    for i, j in [(0,1), (1,2)]:
        ax.plot([i, i, j, j], [max_y*0.9, max_y*0.95, max_y*0.95, max_y*0.9], lw=1, c='k')
        ax.text((i+j)/2, max_y, '***', ha='center')
    
    plt.tight_layout()
    plot_path = os.path.join(desktop_path, 'ana_grafik.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    return plot_path

# 7. RAPOR OLUŞTURMA
def generate_scientific_report(results, desktop_path):
    """Analiz raporu oluşturur"""
    stats, anova, tukey, logrank_p, model = results
    
    # Tukey sonuçlarını DataFrame'e çevir
    tukey_df = pd.DataFrame(tukey._results_table.data[1:], 
                           columns=tukey._results_table.data[0])
    
    report = f"""
    FERMENTASYON ANALİZ RAPORU
    =========================
    1. TEMEL İSTATİSTİKLER
    {stats.to_string()}
    
    2. ANOVA SONUCU
    p-değeri: {anova.pvalue:.4f}
    
    3. TUKEY TESTİ
    {tukey_df.to_string(index=False)}
    
    4. SURVIVAL ANALİZ
    Log-rank p: {logrank_p:.4f}
    
    5. KARIŞIK MODEL
    {model.summary().as_text() if model else 'Model çalıştırılamadı'}
    """
    
    report_path = os.path.join(desktop_path, 'rapor.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    return report_path

# 8. ÜRÜN-VARDİYA İSTATİSTİKLERİ (GÜNCELLENMİŞ)
def generate_product_shift_stats(data):
    """Her ürün için vardiyalara göre ortalama süreler ve sapmaları hesaplar"""
    # Vardiya genel ortalamalarını hesapla (saat cinsinden)
    vardiya_ort = data.groupby('Vardiya', observed=True)['Gecis_Suresi_dk'].mean() / 60
    
    # Ürün-vardiya bazlı istatistikleri hesapla
    stats = data.groupby(['Cig Stok Kodu', 'Vardiya'], observed=True).agg(
        Ortalama=('Gecis_Suresi_dk', 'mean'),
        Std=('Gecis_Suresi_dk', 'std'),
        Ornek=('Gecis_Suresi_dk', 'count')
    ).reset_index()
    
    # Saate çevir ve sapmaları hesapla
    stats['Ortalama_saat'] = stats['Ortalama'] / 60
    
    # Vardiya ortalamalarını eşleştirerek sapmaları hesapla
    vardiya_dict = vardiya_ort.to_dict()
    stats['Sapma_saat'] = stats.apply(
        lambda x: x['Ortalama_saat'] - vardiya_dict[x['Vardiya']], 
        axis=1
    )
    stats['Sapma_yuzde'] = stats.apply(
        lambda x: x['Sapma_saat'] / vardiya_dict[x['Vardiya']] * 100, 
        axis=1
    )
    
    # Pivot table oluştur
    pivot = stats.pivot_table(
        index='Cig Stok Kodu',
        columns='Vardiya',
        values=['Ortalama_saat', 'Sapma_saat', 'Sapma_yuzde', 'Ornek'],
        aggfunc='first'
    )
    
    # Sütun isimlerini düzenle
    pivot.columns = ['_'.join(col).strip() for col in pivot.columns.values]
    
    # Sütun sıralamasını düzenle
    columns_ordered = []
    for vardiya in ['00:00-08:00', '08:00-16:00', '16:00-00:00']:
        for metric in ['Ortalama_saat', 'Sapma_saat', 'Sapma_yuzde', 'Ornek']:
            col_name = f"{metric}_{vardiya}"
            if col_name in pivot.columns:
                columns_ordered.append(col_name)
    
    return pivot[columns_ordered], vardiya_ort


def calculate_entropy(series, bins=10):
    """Shannon entropisi hesaplar"""
    counts, _ = np.histogram(series, bins=bins, density=True)
    probabilities = counts / np.sum(counts)
    probabilities = probabilities[probabilities > 0]
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return entropy
'''
def entropic_prior_model(data):
    """
    Entropiye Dayalı Bayes Önceliği uygular:
    Her ürün-vardiya çifti için entropi ve varyansa dayalı prior tahmini üretir.
    Uç değerler IQR yöntemiyle temizlenir.
    """
    results = []
    grouped = data.groupby(['Cig Stok Kodu', 'Vardiya'])

    for (cig_kodu, vardiya), group in grouped:
        sureler = group['Gecis_Suresi_dk'].dropna()
        if len(sureler) >= 5:
            # Uç değer temizliği (IQR yöntemi)
            Q1 = sureler.quantile(0.25)
            Q3 = sureler.quantile(0.75)
            IQR = Q3 - Q1
            filtered_sureler = sureler[(sureler >= Q1 - 1.5 * IQR) & (sureler <= Q3 + 1.5 * IQR)]

            if len(filtered_sureler) >= 5:
                entropy = calculate_entropy(filtered_sureler)
                mean_duration = filtered_sureler.mean() / 60
                prior_variance = 0.5 * entropy + 0.1

                # Entropi sınıfı belirle
                if entropy < 1:
                    entropy_class = 'Stabil'
                elif entropy < 2:
                    entropy_class = 'Dalgalı'
                else:
                    entropy_class = 'Kritik'

                results.append({
                    'Cig Stok Kodu': cig_kodu,
                    'Vardiya': vardiya,
                    'Entropi': entropy,
                    'Tahmini_Sure': mean_duration,
                    'Prior_Varyans': prior_variance,
                    'Entropi_Sinifi': entropy_class
                })

    return pd.DataFrame(results)
    '''
def bayesian_posterior(mean_likelihood, var_likelihood, mean_prior, var_prior):
    var_post = 1 / (1/var_likelihood + 1/var_prior)
    mean_post = var_post * (mean_likelihood/var_likelihood + mean_prior/var_prior)
    return mean_post, var_post

def bayes_model_with_entropy(data):
    results = []
    grouped = data.groupby(['Cig Stok Kodu', 'Vardiya'])

    for (cig_kodu, vardiya), group in grouped:
        sureler = group['Gecis_Suresi_dk'].dropna()
        if len(sureler) >= 5:
            Q1 = sureler.quantile(0.25)
            Q3 = sureler.quantile(0.75)
            IQR = Q3 - Q1
            filtered_sureler = sureler[(sureler >= Q1 - 1.5 * IQR) & (sureler <= Q3 + 1.5 * IQR)]

            if len(filtered_sureler) >= 5:
                entropy = calculate_entropy(filtered_sureler)
                mean_likelihood = filtered_sureler.mean() / 60
                var_likelihood = np.var(filtered_sureler / 60)

                mean_prior = mean_likelihood
                var_prior = 0.3 * entropy + 0.05

                mean_post, var_post = bayesian_posterior(mean_likelihood, var_likelihood, mean_prior, var_prior)

                results.append({
                    'Cig Stok Kodu': cig_kodu,
                    'Vardiya': vardiya,
                    'Entropi': entropy,
                    'Prior_Varyans': var_prior,
                    'Likelihood_Mean': mean_likelihood,
                    'Likelihood_Varyans': var_likelihood,
                    'Posterior_Tahmin': mean_post,
                    'Posterior_Varyans': var_post
                })

    df = pd.DataFrame(results)

    if not df.empty:
        # Posterior varyansa göre güven sınıfı belirleme (Hibrit Z-score yöntemi)
        mean_var = df['Posterior_Varyans'].mean()
        std_var = df['Posterior_Varyans'].std()

        if std_var < 0.1:
            kesin_eşik = 0.75
            şüpheli_eşik = 0.95
        else:
            kesin_eşik = mean_var - std_var
            şüpheli_eşik = mean_var + std_var

        def classify(var):
            if var < kesin_eşik:
                return "Kesin"
            elif var > şüpheli_eşik:
                return "Şüpheli"
            else:
                return "Orta"

        df['Guven_Skoru'] = df['Posterior_Varyans'].apply(classify)

    return df

def generate_klasik_df(product_stats, bayes_model_df):
    """Klasik ortalamaları ürün-vardiya uzun formatına getirir ve Bayes model ile eşleşenleri döner"""
    klasik_df = product_stats.reset_index().melt(
        id_vars=["Cig Stok Kodu"], 
        value_vars=[col for col in product_stats.columns if "Ortalama_saat" in col],
        var_name="Vardiya", 
        value_name="Klasik_Ortalama"
    )
    klasik_df["Vardiya"] = klasik_df["Vardiya"].str.replace("Ortalama_saat_", "", regex=False)

    # Bayes modeli ile eşleşenleri al
    merged_df = pd.merge(
        bayes_model_df[["Cig Stok Kodu", "Vardiya"]],
        klasik_df,
        on=["Cig Stok Kodu", "Vardiya"],
        how="left"
    )
    return merged_df

# ANA İŞLEM AKIŞI (TAM VERSİYON)
def main():
    # 1. Veri yükleme
    file_path = r"C:/Users/ARDAN/Desktop/ucaylikcigmamul22.csv"
    data = load_and_preprocess(file_path)
    
    # 2. Analizleri çalıştırma
    urun_tahminleri = urun_bazli_tahmin_araliklari(data)
    vardiya_stats, anova, tukey = basic_analyses(data)
    logrank_p, survival_plot = perform_survival_analysis(data)
    mixed_model = run_mixed_effects_model(data)
    product_stats, vardiya_ort = generate_product_shift_stats(data) 
    bayes_model_df = bayes_model_with_entropy(data)
    klasik_df = generate_klasik_df(product_stats, bayes_model_df)  # GÜNCELLENDİ
        # Bayes Öncelik Modeli
    bayes_model_df = bayes_model_with_entropy(data)
    
    # 3. Çıktıları oluşturma
    desktop_path = os.path.join(os.path.expanduser('~'), 'Desktop')
    
    # Excel çıktısı
    output_path = os.path.join(desktop_path, 'fermentasyon_analiz_sonuclari3.xlsx')
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # Ana veri
        final_data = data.merge(urun_tahminleri, on=['Cig Stok Kodu', 'Vardiya'], how='left')
        final_data.to_excel(writer, sheet_name='Ham_Veri', index=False)
        bayes_model_df.to_excel(writer, sheet_name='Bayes_Model', index=False)
        # Ürün-vardiya istatistikleri
        product_stats.to_excel(writer, sheet_name='Urun_Vardiya_Stats', index=True)
        # Entropi ile posterior varyans arasındaki ilişki
        posterior_corr = bayes_model_df[['Entropi', 'Posterior_Varyans']].corr().iloc[0, 1]
        print(f"[POSTERIOR VAR KORELASYON] Entropi vs Posterior Varyans Korelasyonu: {posterior_corr:.3f}")
        # Grup bilgileri
        gruplar = pd.DataFrame({
            'Grup': ['Grup1', 'Grup2', 'Grup3', 'Grup4', 'Grup5', 'Grup6'],
            'Aralik': ['0-1000', '1001-2000', '2001-3000', '3001-4000', '4001-5000', '5000+']
        })
        gruplar.to_excel(writer, sheet_name='Gruplar', index=False)
        
        # Vardiya ortalamaları
        pd.DataFrame({
            'Vardiya': vardiya_ort.index,
            'Ortalama_Sure_saat': vardiya_ort.values
        }).to_excel(writer, sheet_name='Vardiya_Ortalamalari', index=False)
    
    # Grafikler
    plot_path = create_main_plot(data, desktop_path)
    survival_plot.savefig(os.path.join(desktop_path, 'survival_analysis.png'), dpi=300)
    
    # Rapor
    results = (vardiya_stats, anova, tukey, logrank_p, mixed_model)
    report_path = generate_scientific_report(results, desktop_path)
    
    print("ANALİZ BAŞARIYLA TAMAMLANDI!")
    print(f"Excel çıktısı: {output_path}")
    print(f"Rapor: {report_path}")
    print(f"Grafikler: {plot_path} ve survival_analysis.png")


    # --- Entropi vs Posterior Varyans Grafiği ---
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x="Entropi", y="Posterior_Varyans", data=bayes_model_df)
    plt.title("Entropi vs Posterior Varyans")
    plt.xlabel("Entropi")
    plt.ylabel("Posterior Varyans")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(desktop_path, "entropi_vs_posterior_var.png"), dpi=300)
    plt.close()
    # --- BAYES VS KLASİK ORTALAMA KARŞILAŞTIRMA ---

    # 1. Uzun formata getir
    classic_means = product_stats.reset_index()[['Cig Stok Kodu'] + [col for col in product_stats.columns if 'Ortalama_saat' in col]]
    classic_long = classic_means.melt(id_vars='Cig Stok Kodu', var_name='Vardiya', value_name='Klasik_Ortalama')
    classic_long['Vardiya'] = classic_long['Vardiya'].str.extract(r'Ortalama_saat_(.*)')

    # 2. Bayes çıktısıyla birleştir
    bayes_long = bayes_model_df[['Cig Stok Kodu', 'Vardiya', 'Posterior_Tahmin', 'Entropi']]
    compare_df = pd.merge(bayes_long, classic_long, on=['Cig Stok Kodu', 'Vardiya'], how='inner')

    # 3. Fark hesapla
    compare_df['Fark (Bayes - Klasik)'] = compare_df['Posterior_Tahmin'] - compare_df['Klasik_Ortalama']

    # 4. Korelasyon analizi
    correlation = compare_df[['Entropi', 'Fark (Bayes - Klasik)']].corr().iloc[0,1]

    # 5. Grafik
    plt.figure(figsize=(10, 6))
    plt.scatter(compare_df['Entropi'], compare_df['Fark (Bayes - Klasik)'], alpha=0.7)
    plt.axhline(0, color='gray', linestyle='--')
    plt.title(f'Entropi vs Bayes-Klasik Farkı (Korelasyon: {correlation:.2f})')
    plt.xlabel('Entropi')
    plt.ylabel('Bayes - Klasik Ortalama (saat)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(desktop_path, 'entropi_vs_fark.png'), dpi=300)
    plt.close()

    # 6. Excel'e ek olarak kaydet
    with pd.ExcelWriter(output_path, mode='a', engine='openpyxl') as writer:
        compare_df.to_excel(writer, sheet_name='Bayes_vs_Klasik', index=False)

    # 1. Entropi ile sapma farkı arasındaki korelasyonu hesapla
    bayes_model_df["Klasik_Ortalama"] = klasik_df["Klasik_Ortalama"].values
    bayes_model_df["Fark"] = bayes_model_df["Posterior_Tahmin"] - bayes_model_df["Klasik_Ortalama"]

    corr = bayes_model_df["Entropi"].corr(bayes_model_df["Fark"])

    # Spearman korelasyonu alternatif olarak (sıralama bazlı):
    # spearman_corr, _ = spearmanr(bayes_model_df["Entropi"], bayes_model_df["Fark"])

    # 2. Grafik çiz
    plt.figure(figsize=(12, 6))
    sns.scatterplot(x="Entropi", y="Fark (Bayes - Klasik)", data=compare_df)  # DÜZELTTİM
    plt.axhline(0, linestyle='--', color='gray')
    plt.title(f"Entropi vs (Bayes - Klasik Tahmin)  — Korelasyon: {corr:.2f}")
    plt.xlabel("Entropi")
    plt.ylabel("Bayes - Klasik Ortalama (saat)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(desktop_path, "entropi_vs_bayes_klasik_fark2.png"), dpi=300)
    plt.show()

    return compare_df

if __name__ == "__main__":
    compare_df = main()

from scipy.stats import linregress

# Bayes vs Klasik farkı için regresyon
slope, intercept, r_value, p_value, std_err = linregress(compare_df['Entropi'], compare_df['Fark (Bayes - Klasik)'])

print(f"[REGRESYON] Eğim: {slope:.3f}")
print(f"[REGRESYON] p-değeri: {p_value:.4f}")
print(f"[REGRESYON] R-kare: {r_value**2:.3f}")