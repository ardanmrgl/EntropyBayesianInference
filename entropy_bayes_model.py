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

# DATA LOADING AND PREPROCESSING
def load_and_preprocess(file_path):

    data = pd.read_csv(file_path, encoding='utf-8', sep=';', low_memory=False)
    
    # Date conversions
    data['Raw Product Weighing Date'] = pd.to_datetime(data['Raw Product Weighing Date'], format='%Y-%m-%d-%H.%M.%S', errors='coerce')
    data['Semi-finished Product Weighing Date'] = pd.to_datetime(data['Semi-finished Product Weighing Date'], format='%Y-%m-%d-%H.%M.%S', errors='coerce')
    
    # Remove missing dates
    data = data.dropna(subset=['Raw Product Weighing Date', 'Semi-finished Product Weighing Date'])
    
    # Calculate transition time (in minutes)
    data['Transition_Time_Min'] = (data['Semi-finished Product Weighing Date'] - data['Raw Product Weighing Date']).dt.total_seconds() / 60
    
    # Assign shift
    data['Raw_Product_Time'] = data['Raw Product Weighing Date'].dt.hour
    data['Shift'] = pd.cut(data['Raw_Product_Time'],
                            bins=[0, 8, 16, 24],
                            labels=['00:00-08:00', '08:00-16:00', '16:00-00:00'],
                            right=False)
    
    # Generate product ID
    data['Product_ID'] = data['Raw Product Stock Code'].astype(str) + '_' + data['Semi-finished Product Stock Code'].astype(str)
    
    # Assign product group
    data['Product_Group'] = pd.cut(data['Raw Product Stock Code'],
                               bins=[0, 1000, 2000, 3000, 4000, 5000, np.inf],
                               labels=['Group1', 'Group2', 'Group3', 'Group4', 'Group5', 'Group6'])
    
    return data

# PRODUCT-BASED FORECAST RANGE
def urun_bazli_tahmin_araliklari(data):
    """Creates prediction intervals by product groups."""
    results = []
    
    # Global shift averages
    global_vardiya = data.groupby('Shift', observed=True)['Transition_Time_Min'].agg(['mean', 'std'])
    
    for (cig_kodu, vardiya), group in data.groupby(['Raw Product Stock Code', 'Shift'], observed=True):
        if len(group) >= 3:
            mean = group['Transition_Time_Min'].mean()
            std = group['Transition_Time_Min'].std()
            n = len(group)
            t_value = stats.t.ppf(0.95, df=n-1)  # 90% confidence interval
            margin = t_value * (std / np.sqrt(n))
            metod = 'Product–Shift Specific'
        else:
            mean = global_vardiya.loc[vardiya, 'mean']
            margin = 1.5 * 60  # 1.5 hour fixed confidence interval
            metod = 'Global Shift Average.'
        
        results.append({
            'Raw Product Stock Code': cig_kodu,
            'Shift': vardiya,
            'Average_Time_Hour': mean / 60,
            'CI_Lower_Bound_Hour': (mean - margin) / 60,
            'CI_Upper_Bound_Hour': (mean + margin) / 60,
            'Method': metod,
            'Sample_Size': len(group)
        })
    
    return pd.DataFrame(results)

# BASİC ANALYSİS
def basic_analyses(data):
    """Performs basic statistical analyses."""
    # Shift Basic Stats
    vardiya_stats = data.groupby('Shift', observed=True).agg(
        Average=('Transition_Time_Min', 'mean'),
        Std=('Transition_Time_Min', 'std'),
        Count=('Transition_Time_Min', 'count')
    )
    
    vardiya_stats['CI_95'] = 1.96 * vardiya_stats['Std'] / np.sqrt(vardiya_stats['Count'])
    
    # ANOVA test
    anova_result = stats.f_oneway(
        *[group['Transition_Time_Min'] for _, group in data.groupby('Shift', observed=True)]
    )
    
    # Tukey HSD test
    tukey_result = pairwise_tukeyhsd(
        endog=data['Transition_Time_Min'],
        groups=data['Shift'],
        alpha=0.05
    )
    
    return vardiya_stats, anova_result, tukey_result

# SURVIVAL ANALYSİS
def perform_survival_analysis(data):
    
    plt.figure(figsize=(12, 7))
    ax = plt.subplot(111)
    
    palette = {"00:00-08:00": "#1f77b4", "08:00-16:00": "#ff7f0e", "16:00-00:00": "#2ca02c"}
    
    for vardiya in data['Shift'].cat.categories:
        kmf = KaplanMeierFitter()
        kmf.fit(data[data['Shift'] == vardiya]['Transition_Time_Min'], 
                label=vardiya)
        kmf.plot_survival_function(ax=ax, ci_show=True, color=palette[vardiya])
    
    plt.title('Probability of Fermentation Completion by Shift', fontsize=14)
    plt.xlabel('Time (hours)')
    plt.ylabel('Survival Probability')
    plt.grid(True, alpha=0.3)
    
    logrank = multivariate_logrank_test(data['Transition_Time_Min'], data['Shift'])
    return logrank.p_value, plt.gcf()

# MIXED EFFECTS MODEL
def run_mixed_effects_model(data):
    
    # Clean missing values
    data_clean = data.dropna(subset=['Transition_Time_Min', 'Shift', 'Raw Product Stock Code'])
    
    try:
        model = mixedlm("Transition_Time_Min ~ C(Shift)", 
                       data=data_clean,
                       groups=data_clean["Raw Product Stock Code"],
                       re_formula="~C(Shift)")
        return model.fit(method='powell')
    except Exception as e:
        print(f"Model hatası: {str(e)}")
        return None

# 6. VİSUALİZATİON
def create_main_plot(data, desktop_path):
    """Creates the main comparison plot."""
    plt.figure(figsize=(12, 7))
    palette = {"00:00-08:00": "#1f77b4", "08:00-16:00": "#ff7f0e", "16:00-00:00": "#2ca02c"}

    ax = sns.barplot(
        x='Shift', 
        y='Transition_Time_Min', 
        hue='Shift', 
        data=data,
        estimator=np.mean, 
        errorbar=None, 
        palette=palette, 
        legend=False
)
                    
    
    plt.title('Average Transition Times by Shift', fontsize=14)
    plt.ylabel('Average Transition Time (hours)')
    plt.xticks(rotation=45)
    
    # Statistical signs
    max_y = data['Transition_Time_Min'].max()
    for i, j in [(0,1), (1,2)]:
        ax.plot([i, i, j, j], [max_y*0.9, max_y*0.95, max_y*0.95, max_y*0.9], lw=1, c='k')
        ax.text((i+j)/2, max_y, '***', ha='center')
    
    plt.tight_layout()
    plot_path = os.path.join(desktop_path, 'Averiage Transition Times by Shift.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    return plot_path

# 7. REPORT
def generate_scientific_report(results, desktop_path):
    """Generates the analysis report."""
    stats, anova, tukey, logrank_p, model = results
    
    # Convert Tukey results to DataFrame
    tukey_df = pd.DataFrame(tukey._results_table.data[1:], 
                           columns=tukey._results_table.data[0])
    
    report = f"""
    FERMENTATION ANALYSIS REPORT
    ============================
    1. DESCRIPTIVE STATISTICS
    {stats.to_string()}

    2. ANOVA RESULT
    p-value: {anova.pvalue:.4f}

    3. TUKEY TEST
    {tukey_df.to_string(index=False)}

    4. SURVIVAL ANALYSIS
    Log-rank p: {logrank_p:.4f}

    5. MIXED-EFFECTS MODEL
    {model.summary().as_text() if model else 'Model could not be fitted'}
    """
        
    report_path = os.path.join(desktop_path, 'report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    return report_path

# 8. PRODUCT-SHIFT STATISTICS
def generate_product_shift_stats(data):
    """Calculates average transition times and deviations by product and shift."""
    # # Calculate shift averages (in hours)
    vardiya_ort = data.groupby('Shift', observed=True)['Transition_Time_Min'].mean() / 60
    
    # Product-Shift-based statistics hesapla
    stats = data.groupby(['Raw Product Stock Code', 'Shift'], observed=True).agg(
        Average=('Transition_Time_Min', 'mean'),
        Std=('Transition_Time_Min', 'std'),
        Sample=('Transition_Time_Min', 'count')
    ).reset_index()
    
    # Convert to hours and calculate deviations
    stats['Average_Time'] = stats['Average'] / 60
    print("stats columns:", stats.columns)
    # Calculate deviations by matching shift averages
    vardiya_dict = vardiya_ort.to_dict()
    stats['Deviation_Hour'] = stats.apply(
        lambda x: x['Average_Time'] - vardiya_dict[x['Shift']], 
        axis=1
    )
    stats['Deviation_Percent'] = stats.apply(
        lambda x: x['Deviation_Hour'] / vardiya_dict[x['Shift']] * 100, 
        axis=1
    )
    
    # Create a pivot table
    pivot = stats.pivot_table(
        index='Raw Product Stock Code',
        columns='Shift',
        values=['Average_Time', 'Deviation_Hour', 'Deviation_Percent', 'Sample'],
        aggfunc='first'
    )
    
    # Edit column names
    pivot.columns = ['_'.join(col).strip() for col in pivot.columns.values]
    print("stats columns:", stats.columns)
    # Edit column sorting
    columns_ordered = []
    for vardiya in ['00:00-08:00', '08:00-16:00', '16:00-00:00']:
        for metric in ['Average_Time', 'Deviation_Hour', 'Deviation_Percent', 'Sample']:
            col_name = f"{metric}_{vardiya}"
            if col_name in pivot.columns:
                columns_ordered.append(col_name)
    
    return pivot[columns_ordered], vardiya_ort


def calculate_entropy(series, bins=10):
    """Shannon entropy calculate"""
    counts, _ = np.histogram(series, bins=bins, density=True)
    probabilities = counts / np.sum(counts)
    probabilities = probabilities[probabilities > 0]
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return entropy


def bayesian_posterior(mean_likelihood, var_likelihood, mean_prior, var_prior):
    var_post = 1 / (1/var_likelihood + 1/var_prior)
    mean_post = var_post * (mean_likelihood/var_likelihood + mean_prior/var_prior)
    return mean_post, var_post

def bayes_model_with_entropy(data):
    results = []
    grouped = data.groupby(['Raw Product Stock Code', 'Shift'])

    for (cig_kodu, vardiya), group in grouped:
        sureler = group['Transition_Time_Min'].dropna()
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
                    'Raw Product Stock Code': cig_kodu,
                    'Shift': vardiya,
                    'Entropy': entropy,
                    'Prior_Variance': var_prior,
                    'Likelihood_Mean': mean_likelihood,
                    'Likelihood_Variance': var_likelihood,
                    'Posterior_Prediction': mean_post,
                    'Posterior_Variance': var_post
                })

    df = pd.DataFrame(results)

    if not df.empty:
        # Determining confidence class based on posterior variance (Hybrid Z-score method)
        mean_var = df['Posterior_Variance'].mean()
        std_var = df['Posterior_Variance'].std()

        if std_var < 0.1:
            kesin_eşik = 0.75
            şüpheli_eşik = 0.95
        else:
            kesin_eşik = mean_var - std_var
            şüpheli_eşik = mean_var + std_var

        def classify(var):
            if var < kesin_eşik:
                return "Certain"
            elif var > şüpheli_eşik:
                return "Suspect"
            else:
                return "Medium"

        df['Trust_Score'] = df['Posterior_Variance'].apply(classify)

    return df

def generate_klasik_df(product_stats, bayes_model_df):
    """Converts classical means to product-shift long format and returns those that match the Bayesian model"""
    klasik_df = product_stats.reset_index().melt(
        id_vars=["Raw Product Stock Code"], 
        value_vars=[col for col in product_stats.columns if "Average_Time" in col],
        var_name="Shift", 
        value_name="Classic_Average"
    )
    klasik_df["Shift"] = klasik_df["Shift"].str.replace("Average_Time_", "", regex=False)

    # Get matches with Bayesian model
    print("bayes_model_df columns:", bayes_model_df.columns)
    print("klasik_df columns:", klasik_df.columns)
    merged_df = pd.merge(
        bayes_model_df[["Raw Product Stock Code", "Shift"]],
        klasik_df,
        on=["Raw Product Stock Code", "Shift"],
        how="left"
    )
    return merged_df

# MAIN PROCESS FLOW
def main():
    # Data loading
    file_path = r"C:/Users/ARDAN/Desktop/ucaylikcigmamul22.csv"
    data = load_and_preprocess(file_path)
    
    # Run analytics
    urun_tahminleri = urun_bazli_tahmin_araliklari(data)
    vardiya_stats, anova, tukey = basic_analyses(data)
    logrank_p, survival_plot = perform_survival_analysis(data)
    mixed_model = run_mixed_effects_model(data)
 
    product_stats, vardiya_ort = generate_product_shift_stats(data) 
    print("product_stats columns:", product_stats.columns)
    bayes_model_df = bayes_model_with_entropy(data)
    klasik_df = generate_klasik_df(product_stats, bayes_model_df)  # GÜNCELLENDİ
    # Bayesian Priority Model
    bayes_model_df = bayes_model_with_entropy(data)
    
    # Generating outputs
    desktop_path = os.path.join(os.path.expanduser('~'), 'Desktop')
    
    
    # Excel output
    output_path = os.path.join(desktop_path, 'fermentation_analysis_results.xlsx')
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # Raw Data
        final_data = data.merge(urun_tahminleri, on=['Raw Product Stock Code', 'Shift'], how='left')
        final_data.to_excel(writer, sheet_name='Raw_Data', index=False)
        # Bayesian Model Results
        bayes_model_df.to_excel(writer, sheet_name='Bayesian_Model', index=False)
        # Product-Shift Statistics
        product_stats.to_excel(writer, sheet_name='Product_Shift_Stats', index=True)
        # Entropy vs Posterior Variance Correlation
        posterior_corr = bayes_model_df[['Entropy', 'Posterior_Variance']].corr().iloc[0, 1]
        print(f"[POSTERIOR VARIANCE CORRELATION] Entropy vs Posterior Variance Correlation: {posterior_corr:.3f}")
        # Group info
        groups = pd.DataFrame({
            'Group': ['Group1', 'Group2', 'Group3', 'Group4', 'Group5', 'Group6'],
            'Range': ['0-1000', '1001-2000', '2001-3000', '3001-4000', '4001-5000', '5000+']
        })
        groups.to_excel(writer, sheet_name='Groups', index=False)

        # Shift averages
        pd.DataFrame({
            'Shift': vardiya_ort.index,
            'Average_Time_Hour': vardiya_ort.values
        }).to_excel(writer, sheet_name='Shift_Average', index=False)
        
        groups.to_excel(writer, sheet_name='Groups', index=False)
        

    
    # Graphics
    plot_path = create_main_plot(data, desktop_path)
    survival_plot.savefig(os.path.join(desktop_path, 'survival_analysis.png'), dpi=300)
    
    # Report
    results = (vardiya_stats, anova, tukey, logrank_p, mixed_model)
    report_path = generate_scientific_report(results, desktop_path)
    
    print("ANALİZ BAŞARIYLA TAMAMLANDI!")
    print(f"Excel çıktısı: {output_path}")
    print(f"Rapor: {report_path}")
    print(f"Grafikler: {plot_path} ve survival_analysis.png")


    # --- Entropy vs. Posterior Variance Plot ---
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x="Entropy", y="Posterior_Variance", data=bayes_model_df)
    plt.title("Entropy vs Posterior Variance")
    plt.xlabel("Entropy")
    plt.ylabel("Posterior Variance")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(desktop_path, "Entropy_vs_posterior_var.png"), dpi=300)
    plt.close()

    # --- BAYESIAN VS CLASSICAL MEAN COMPARISON ---

    # Convert to long format
    classic_means = product_stats.reset_index()[['Raw Product Stock Code'] + [col for col in product_stats.columns if 'Average_Time' in col]]
    classic_long = classic_means.melt(id_vars='Raw Product Stock Code', var_name='Shift', value_name='Classic_Average')
    classic_long['Shift'] = classic_long['Shift'].str.extract(r'Average_Time_(.*)')

    # Combine with Bayes output
    bayes_long = bayes_model_df[['Raw Product Stock Code', 'Shift', 'Posterior_Prediction', 'Entropy']]
    compare_df = pd.merge(bayes_long, classic_long, on=['Raw Product Stock Code', 'Shift'], how='inner')

    # Calculate the difference
    compare_df['Difference (Bayesian - Classic)'] = compare_df['Posterior_Prediction'] - compare_df['Classic_Average']

    # Correlation analysis
    correlation = compare_df[['Entropy', 'Difference (Bayesian - Classic)']].corr().iloc[0,1]

    # Chart
    plt.figure(figsize=(10, 6))
    plt.scatter(compare_df['Entropy'], compare_df['Difference (Bayesian - Classic)'], alpha=0.7)
    plt.axhline(0, color='gray', linestyle='--')
    plt.title(f'Entropy vs Bayesian–Classical Mean Difference (Correlation: {correlation:.2f})')
    plt.xlabel('Entropy')
    plt.ylabel('Bayesian – Classical Mean (hours)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(desktop_path, 'Entropy_vs_difference.png'), dpi=300)
    plt.close()

    # Save as attachment to Excel
    with pd.ExcelWriter(output_path, mode='a', engine='openpyxl') as writer:
        compare_df.to_excel(writer, sheet_name='Bayesian_vs_Classic', index=False)

    # Calculate the correlation between entropy and deviation difference
    bayes_model_df["Classic_Average"] = klasik_df["Classic_Average"].values
    bayes_model_df["Difference"] = bayes_model_df["Posterior_Prediction"] - bayes_model_df["Classic_Average"]

    corr = bayes_model_df["Entropy"].corr(bayes_model_df["Difference"])



    # Draw a graph
    plt.figure(figsize=(12, 6))
    sns.scatterplot(x="Entropy", y="Difference (Bayesian - Classic)", data=compare_df)  # DÜZELTTİM
    plt.axhline(0, linestyle='--', color='gray')
    plt.title(f"Entropy vs (Bayesian – Classical Estimate)  — Correlation: {corr:.2f}")
    plt.xlabel("Entropy")
    plt.ylabel("Bayesian – Classical Mean (hours)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(desktop_path, "entropy_vs_bayesian_classic_difference2.png"), dpi=300)
    plt.show()

    return compare_df

if __name__ == "__main__":
    compare_df = main()

from scipy.stats import linregress

# Regression for Bayesian vs Classical difference
slope, intercept, r_value, p_value, std_err = linregress(compare_df['Entropy'], compare_df['Difference (Bayesian - Classic)'])

print(f"[REGRESSION] Slope: {slope:.3f}")
print(f"[REGRESSION] p-value: {p_value:.4f}")
print(f"[REGRESSION] R-squared: {r_value**2:.3f}")
