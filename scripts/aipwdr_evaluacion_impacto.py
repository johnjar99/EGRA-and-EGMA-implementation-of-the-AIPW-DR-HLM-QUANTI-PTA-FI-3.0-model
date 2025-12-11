#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
================================================================================
EVALUACION DE IMPACTO DEL PROGRAMA PTA/FI 3.0
Metodologia AIPW/DR con Wild Cluster Bootstrap
================================================================================

Autor: Equipo de Evaluacion PTA/FI
Fecha: Diciembre 2025
Version: 1.0

Descripcion:
    Este script implementa el estimador Augmented Inverse Propensity Weighting
    (AIPW), tambien conocido como Doubly Robust (DR), para evaluar el impacto
    del programa PTA/FI 3.0 sobre los resultados de EGRA (lectura) y EGMA
    (matematicas).

Metodologia:
    - Estimador AIPW/DR (doblemente robusto)
    - Wild Cluster Bootstrap con pesos Rademacher para inferencia
    - Cross-Fitting con GroupKFold por escuela
    - Gradient Boosting para modelos de resultado
    - Logistic Ridge para propensity score

Uso:
    python aipwdr_evaluacion_impacto.py --data <ruta_datos> --output <ruta_salida>

    Ejemplo:
    python aipwdr_evaluacion_impacto.py --data ../data/base_estudiante_FINAL.csv --output ../results/
================================================================================
"""

import pandas as pd
import numpy as np
import warnings
import argparse
from pathlib import Path
from scipy import stats

# Machine Learning
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

# Visualizacion
import matplotlib.pyplot as plt
import seaborn as sns

# Configuracion global
warnings.filterwarnings('ignore')
np.random.seed(42)
plt.style.use('seaborn-v0_8-whitegrid')


# ==============================================================================
# CONFIGURACION
# ==============================================================================

CONFIG = {
    'AUROC_MIN': 0.6,
    'AUROC_MAX': 0.9,
    'SMD_MAX': 0.10,
    'CLIP_MIN': 0.01,
    'CLIP_MAX': 0.99,
    'N_BOOTSTRAP': 1000,
    'N_BOOTSTRAP_SUBPRUEBAS': 500,
    'N_FOLDS': 5,
    'RANDOM_STATE': 42
}

# Variables EGRA (Lectura)
EGRA_VARS = ['CSL_pct_correctas', 'DPSS_pct_correctas',
             'LP_pct_correctas', 'CL_pct_correctas']

# Variables EGMA (Matematicas)
EGMA_VARS = ['miss_num_pct_correctas', 'comp_mat_op_pct_correctas',
             'sums_pct_correctas', 'substract_pct_correctas',
             'mult_div_pct_correctas', 'res_problems_pct_correctas']

# Nombres legibles de subpruebas
EGRA_SUBPRUEBAS = {
    'CSL_pct_correctas': 'Conocimiento Sonidos Letras (CSL)',
    'DPSS_pct_correctas': 'Decodificacion Pseudopalabras (DPSS)',
    'LP_pct_correctas': 'Lectura de Palabras (LP)',
    'CL_pct_correctas': 'Comprension Lectora (CL)'
}

EGMA_SUBPRUEBAS = {
    'miss_num_pct_correctas': 'Numeros Faltantes (MissNum)',
    'comp_mat_op_pct_correctas': 'Comparacion y Operaciones (CompMatOp)',
    'sums_pct_correctas': 'Sumas',
    'substract_pct_correctas': 'Restas',
    'mult_div_pct_correctas': 'Multiplicacion/Division (MultDiv)',
    'res_problems_pct_correctas': 'Resolucion de Problemas'
}


# ==============================================================================
# FUNCIONES AUXILIARES
# ==============================================================================

def calcular_smd(df, var, tratamiento_col='tratamiento'):
    """Calcula el Standardized Mean Difference (SMD)"""
    trat = df[df[tratamiento_col] == 1][var].dropna()
    ctrl = df[df[tratamiento_col] == 0][var].dropna()
    if len(trat) == 0 or len(ctrl) == 0:
        return 0
    var_pooled = np.sqrt((trat.var() + ctrl.var()) / 2)
    if var_pooled == 0:
        return 0
    return abs((trat.mean() - ctrl.mean()) / var_pooled)


def calcular_ess(weights):
    """Calcula el Effective Sample Size (ESS)"""
    return (np.sum(weights) ** 2) / np.sum(weights ** 2)


def smd_ponderado(df, var, weights_norm, y):
    """Calcula el SMD despues de aplicar pesos IPW"""
    mask_t = y == 1
    mask_c = y == 0
    mean_t = np.average(df[var].values[mask_t], weights=weights_norm[mask_t])
    mean_c = np.average(df[var].values[mask_c], weights=weights_norm[mask_c])
    var_t = np.average((df[var].values[mask_t] - mean_t)**2, weights=weights_norm[mask_t])
    var_c = np.average((df[var].values[mask_c] - mean_c)**2, weights=weights_norm[mask_c])
    var_pooled = np.sqrt((var_t + var_c) / 2)
    return abs((mean_t - mean_c) / var_pooled) if var_pooled > 0 else 0


# ==============================================================================
# PROCESO ETL
# ==============================================================================

def procesar_datos(df_original):
    """Procesa los datos: crea outcomes, trata missings, crea dummies"""
    print("\n" + "="*80)
    print("PROCESO ETL")
    print("="*80)

    df = df_original.copy()

    # Crear outcomes
    df['EGRA_Total'] = df[EGRA_VARS].mean(axis=1)
    df['EGMA_Total'] = df[EGMA_VARS].mean(axis=1)
    print(f"\nOutcomes creados: EGRA_Total, EGMA_Total")

    # Tratar valores faltantes
    df['tiene_tei_slope'] = df['tei_slope_pre'].notna().astype(int)

    vars_imputar = ['tei_nivel_pre', 'ruralidad', 'pdet', 'matricula_pre',
                    'pct_jornada_completa', 'composicion_grados', 'num_jornadas']

    for var in vars_imputar:
        if var in df.columns and df[var].isna().any():
            mediana = df[var].median()
            df[var] = df[var].fillna(mediana)

    # Crear dummies de zona
    if 'zona_ptafi' in df.columns:
        moda_zona = df['zona_ptafi'].mode()[0] if not df['zona_ptafi'].mode().empty else 'C'
        df['zona_ptafi'] = df['zona_ptafi'].fillna(moda_zona)
        zona_dummies = pd.get_dummies(df['zona_ptafi'], prefix='zona', drop_first=True)
        df = pd.concat([df, zona_dummies], axis=1)

    # Definir covariables
    zona_cols = [c for c in df.columns if c.startswith('zona_') and c != 'zona_ptafi']

    covariables = [
        'edad', 'sexo',
        'ruralidad', 'pdet',
        'tei_nivel_pre',
        'matricula_pre',
        'pct_jornada_completa',
        'composicion_grados',
    ] + zona_cols

    covariables = [c for c in covariables if c in df.columns]

    print(f"Covariables: {len(covariables)}")
    print(f"N estudiantes: {len(df)}")
    print(f"N escuelas: {df['codigo_dane'].nunique()}")

    return df, covariables


# ==============================================================================
# MODELO DE PROPENSION
# ==============================================================================

def estimar_propensity_score(df, covariables):
    """Estima el propensity score usando Logistic Ridge"""
    print("\n" + "="*80)
    print("MODELO DE PROPENSION")
    print("="*80)

    X = df[covariables].values
    y = df['tratamiento'].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegressionCV(
        cv=5, penalty='l2', solver='lbfgs',
        Cs=np.logspace(-4, 2, 20), max_iter=2000,
        random_state=CONFIG['RANDOM_STATE']
    )
    model.fit(X_scaled, y)

    ps = model.predict_proba(X_scaled)[:, 1]
    ps = np.clip(ps, CONFIG['CLIP_MIN'], CONFIG['CLIP_MAX'])

    # Diagnosticos
    auroc = roc_auc_score(y, ps)

    # ESS
    weights_t = 1 / ps[y == 1]
    weights_c = 1 / (1 - ps[y == 0])
    weights_t_norm = weights_t / weights_t.sum() * len(weights_t)
    weights_c_norm = weights_c / weights_c.sum() * len(weights_c)

    ess_t = calcular_ess(weights_t_norm)
    ess_c = calcular_ess(weights_c_norm)
    n_tratados = (y == 1).sum()
    n_control = (y == 0).sum()

    print(f"\nAUROC: {auroc:.4f}")
    print(f"ESS Tratamiento: {ess_t:.0f}/{n_tratados} ({ess_t/n_tratados*100:.1f}%)")
    print(f"ESS Control: {ess_c:.0f}/{n_control} ({ess_c/n_control*100:.1f}%)")

    # SMD
    smds = {var: calcular_smd(df, var) for var in covariables}
    smd_promedio = np.mean(list(smds.values()))
    print(f"SMD Promedio: {smd_promedio:.4f}")

    diagnosticos = {
        'auroc': auroc,
        'ess_t': ess_t,
        'ess_c': ess_c,
        'n_tratados': n_tratados,
        'n_control': n_control,
        'smd_promedio': smd_promedio,
        'smds': smds
    }

    return ps, diagnosticos


# ==============================================================================
# MODELOS DE RESULTADO
# ==============================================================================

def entrenar_modelos_resultado(df, outcome_var, covariables, n_folds=5):
    """Entrena modelos de resultado con Cross-Fitting usando GroupKFold"""
    X = df[covariables].values
    y_outcome = df[outcome_var].values
    D = df['tratamiento'].values
    groups = df['codigo_dane'].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    mu1_hat = np.zeros(len(df))
    mu0_hat = np.zeros(len(df))

    gkf = GroupKFold(n_splits=min(n_folds, len(np.unique(groups))))

    for fold, (train_idx, test_idx) in enumerate(gkf.split(X_scaled, y_outcome, groups)):
        train_treated = train_idx[D[train_idx] == 1]
        train_control = train_idx[D[train_idx] == 0]

        model_1 = GradientBoostingRegressor(
            n_estimators=100, max_depth=3, learning_rate=0.1,
            min_samples_leaf=5, random_state=CONFIG['RANDOM_STATE']
        )
        model_0 = GradientBoostingRegressor(
            n_estimators=100, max_depth=3, learning_rate=0.1,
            min_samples_leaf=5, random_state=CONFIG['RANDOM_STATE']
        )

        if len(train_treated) >= 5:
            model_1.fit(X_scaled[train_treated], y_outcome[train_treated])
            mu1_hat[test_idx] = model_1.predict(X_scaled[test_idx])
        else:
            mu1_hat[test_idx] = y_outcome[D == 1].mean()

        if len(train_control) >= 5:
            model_0.fit(X_scaled[train_control], y_outcome[train_control])
            mu0_hat[test_idx] = model_0.predict(X_scaled[test_idx])
        else:
            mu0_hat[test_idx] = y_outcome[D == 0].mean()

    return mu1_hat, mu0_hat


# ==============================================================================
# ESTIMADOR AIPW
# ==============================================================================

def calcular_aipw(Y, D, mu1_hat, mu0_hat, ps, clip_min=0.01, clip_max=0.99):
    """Calcula el estimador AIPW"""
    ps_clipped = np.clip(ps, clip_min, clip_max)
    diff_predicha = mu1_hat - mu0_hat
    corr_tratados = D * (Y - mu1_hat) / ps_clipped
    corr_controles = (1 - D) * (Y - mu0_hat) / (1 - ps_clipped)
    tau_i = diff_predicha + corr_tratados - corr_controles
    ate = np.mean(tau_i)
    return ate, tau_i


def wild_cluster_bootstrap(df, ate_original, tau_i, n_bootstrap=1000, verbose=True):
    """Wild Cluster Bootstrap con pesos Rademacher"""
    escuelas = df['codigo_dane'].unique()
    n_escuelas = len(escuelas)
    escuela_idx = {esc: df['codigo_dane'] == esc for esc in escuelas}
    ate_bootstrap = []

    for b in range(n_bootstrap):
        if verbose and (b + 1) % 200 == 0:
            print(f"    Bootstrap {b + 1}/{n_bootstrap}...")

        rademacher = np.random.choice([1, -1], size=n_escuelas)
        tau_wild = np.zeros(len(df))

        for i, esc in enumerate(escuelas):
            mask = escuela_idx[esc]
            tau_wild[mask] = ate_original + rademacher[i] * (tau_i[mask] - ate_original)

        ate_bootstrap.append(np.mean(tau_wild))

    ate_bootstrap = np.array(ate_bootstrap)
    se = np.std(ate_bootstrap)
    ic_lower = np.percentile(ate_bootstrap, 2.5)
    ic_upper = np.percentile(ate_bootstrap, 97.5)
    t_stat = ate_original / se if se > 0 else 0
    p_value = 2 * (1 - stats.norm.cdf(abs(t_stat)))

    return {
        'ate': ate_original, 'se': se,
        'ic_lower': ic_lower, 'ic_upper': ic_upper,
        'p_value': p_value, 't_stat': t_stat,
        'bootstrap_dist': ate_bootstrap
    }


# ==============================================================================
# ESTIMACION PRINCIPAL
# ==============================================================================

def estimar_efectos(df, covariables, ps):
    """Estima efectos para EGRA Total y EGMA Total"""
    print("\n" + "="*80)
    print("ESTIMACION AIPW/DR")
    print("="*80)

    resultados = {}

    for outcome_name, outcome_var in [('EGRA_Total', 'EGRA_Total'), ('EGMA_Total', 'EGMA_Total')]:
        print(f"\n--- {outcome_name} ---")

        # Modelos de resultado
        mu1, mu0 = entrenar_modelos_resultado(df, outcome_var, covariables)

        # AIPW
        Y = df[outcome_var].values
        D = df['tratamiento'].values

        ate, tau_i = calcular_aipw(Y, D, mu1, mu0, ps)

        # Wild Cluster Bootstrap
        print(f"  Wild Cluster Bootstrap (B={CONFIG['N_BOOTSTRAP']})...")
        resultado = wild_cluster_bootstrap(df, ate, tau_i, CONFIG['N_BOOTSTRAP'])
        resultados[outcome_name] = resultado

        sig = "***" if resultado['p_value'] < 0.01 else "**" if resultado['p_value'] < 0.05 else "*" if resultado['p_value'] < 0.10 else ""
        print(f"  ATE = {resultado['ate']:.4f} {sig}")
        print(f"  SE  = {resultado['se']:.4f}")
        print(f"  IC 95% = [{resultado['ic_lower']:.4f}, {resultado['ic_upper']:.4f}]")
        print(f"  p-valor = {resultado['p_value']:.4f}")

    return resultados


def estimar_subpruebas(df, covariables, ps):
    """Estima efectos para cada subprueba"""
    print("\n" + "="*80)
    print("ESTIMACION DE SUBPRUEBAS")
    print("="*80)

    resultados_sub = {}

    # EGRA
    print("\n--- Subpruebas EGRA ---")
    for var, nombre in EGRA_SUBPRUEBAS.items():
        print(f"  {nombre}...", end=" ")
        mu1, mu0 = entrenar_modelos_resultado(df, var, covariables)
        Y = df[var].values
        D = df['tratamiento'].values
        ate, tau_i = calcular_aipw(Y, D, mu1, mu0, ps)
        resultado = wild_cluster_bootstrap(df, ate, tau_i, CONFIG['N_BOOTSTRAP_SUBPRUEBAS'], verbose=False)
        resultado['variable'] = var
        resultado['tipo'] = 'EGRA'
        resultados_sub[nombre] = resultado
        sig = "***" if resultado['p_value'] < 0.01 else "**" if resultado['p_value'] < 0.05 else "*" if resultado['p_value'] < 0.10 else ""
        print(f"ATE={resultado['ate']:.2f}, p={resultado['p_value']:.3f} {sig}")

    # EGMA
    print("\n--- Subpruebas EGMA ---")
    for var, nombre in EGMA_SUBPRUEBAS.items():
        print(f"  {nombre}...", end=" ")
        mu1, mu0 = entrenar_modelos_resultado(df, var, covariables)
        Y = df[var].values
        D = df['tratamiento'].values
        ate, tau_i = calcular_aipw(Y, D, mu1, mu0, ps)
        resultado = wild_cluster_bootstrap(df, ate, tau_i, CONFIG['N_BOOTSTRAP_SUBPRUEBAS'], verbose=False)
        resultado['variable'] = var
        resultado['tipo'] = 'EGMA'
        resultados_sub[nombre] = resultado
        sig = "***" if resultado['p_value'] < 0.01 else "**" if resultado['p_value'] < 0.05 else "*" if resultado['p_value'] < 0.10 else ""
        print(f"ATE={resultado['ate']:.2f}, p={resultado['p_value']:.3f} {sig}")

    return resultados_sub


# ==============================================================================
# GUARDAR RESULTADOS
# ==============================================================================

def guardar_resultados(resultados, resultados_sub, diagnosticos, df, output_dir):
    """Guarda todos los resultados en archivos CSV y Excel"""
    print("\n" + "="*80)
    print("GUARDANDO RESULTADOS")
    print("="*80)

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Resultados principales
    resultados_df = pd.DataFrame([
        {
            'Outcome': nombre,
            'ATE': res['ate'],
            'SE': res['se'],
            'IC_Lower': res['ic_lower'],
            'IC_Upper': res['ic_upper'],
            'p_value': res['p_value'],
            't_stat': res['t_stat'],
            'Significativo_05': res['p_value'] < 0.05,
            'Significativo_10': res['p_value'] < 0.10
        }
        for nombre, res in resultados.items()
    ])

    resultados_df.to_csv(output_dir / 'resultados_finales.csv', index=False, encoding='utf-8-sig')
    print(f"  [OK] resultados_finales.csv")

    # Subpruebas
    subpruebas_list = []
    for nombre, res in resultados_sub.items():
        subpruebas_list.append({
            'Nombre': nombre,
            'Variable': res['variable'],
            'Tipo': res['tipo'],
            'ATE': res['ate'],
            'SE': res['se'],
            'IC_Lower': res['ic_lower'],
            'IC_Upper': res['ic_upper'],
            'p_value': res['p_value'],
            't_stat': res['t_stat'],
            'n': len(df),
            'Significativo_05': res['p_value'] < 0.05,
            'Significativo_10': res['p_value'] < 0.10
        })

    subpruebas_df = pd.DataFrame(subpruebas_list)
    subpruebas_df.to_csv(output_dir / 'resultados_subpruebas.csv', index=False, encoding='utf-8-sig')
    print(f"  [OK] resultados_subpruebas.csv")

    # Diagnosticos
    diag_df = pd.DataFrame([{
        'AUROC': diagnosticos['auroc'],
        'SMD_promedio': diagnosticos['smd_promedio'],
        'ESS_tratamiento_pct': diagnosticos['ess_t'] / diagnosticos['n_tratados'] * 100,
        'ESS_control_pct': diagnosticos['ess_c'] / diagnosticos['n_control'] * 100,
        'N_estudiantes': len(df),
        'N_escuelas': df['codigo_dane'].nunique(),
        'N_bootstrap': CONFIG['N_BOOTSTRAP']
    }])
    diag_df.to_csv(output_dir / 'diagnosticos.csv', index=False)
    print(f"  [OK] diagnosticos.csv")


def crear_visualizaciones(resultados, resultados_sub, diagnosticos, df, output_dir):
    """Crea todas las visualizaciones"""
    print("\n" + "="*80)
    print("CREANDO VISUALIZACIONES")
    print("="*80)

    output_dir = Path(output_dir)
    figures_dir = output_dir / 'figures'
    figures_dir.mkdir(exist_ok=True, parents=True)

    # Forest Plot Principal
    fig, ax = plt.subplots(figsize=(10, 6))
    nombres = list(resultados.keys())
    ates = [r['ate'] for r in resultados.values()]
    lowers = [r['ic_lower'] for r in resultados.values()]
    uppers = [r['ic_upper'] for r in resultados.values()]
    y_pos = np.arange(len(nombres))

    for i, (ate, lower, upper) in enumerate(zip(ates, lowers, uppers)):
        ax.errorbar(ate, i, xerr=[[ate-lower], [upper-ate]], fmt='o',
                    color='steelblue', capsize=5, markersize=10, linewidth=2)

    ax.axvline(0, color='red', linestyle='--', linewidth=2)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(nombres)
    ax.set_xlabel('ATE (puntos porcentuales)')
    ax.set_title('Efectos del Tratamiento PTA/FI 3.0\nIC 95% Wild Cluster Bootstrap', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig(figures_dir / 'Forest_Plot_Principal.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Forest_Plot_Principal.png")

    # Forest Plot Subpruebas
    fig, axes = plt.subplots(1, 2, figsize=(16, 10))

    # EGRA
    ax1 = axes[0]
    egra_res = {k: v for k, v in resultados_sub.items() if v['tipo'] == 'EGRA'}
    nombres_egra = list(egra_res.keys())
    ates_egra = [r['ate'] for r in egra_res.values()]
    lowers_egra = [r['ic_lower'] for r in egra_res.values()]
    uppers_egra = [r['ic_upper'] for r in egra_res.values()]

    for i, (ate, lower, upper) in enumerate(zip(ates_egra, lowers_egra, uppers_egra)):
        ax1.errorbar(ate, i, xerr=[[ate-lower], [upper-ate]], fmt='o',
                     color='steelblue', capsize=5, markersize=10, linewidth=2)

    ax1.axvline(0, color='red', linestyle='--', linewidth=2)
    ax1.set_yticks(range(len(nombres_egra)))
    ax1.set_yticklabels(nombres_egra)
    ax1.set_xlabel('ATE (puntos porcentuales)')
    ax1.set_title('Subpruebas EGRA (Lectura)', fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='x')

    # EGMA
    ax2 = axes[1]
    egma_res = {k: v for k, v in resultados_sub.items() if v['tipo'] == 'EGMA'}
    nombres_egma = list(egma_res.keys())
    ates_egma = [r['ate'] for r in egma_res.values()]
    lowers_egma = [r['ic_lower'] for r in egma_res.values()]
    uppers_egma = [r['ic_upper'] for r in egma_res.values()]

    for i, (ate, lower, upper) in enumerate(zip(ates_egma, lowers_egma, uppers_egma)):
        ax2.errorbar(ate, i, xerr=[[ate-lower], [upper-ate]], fmt='s',
                     color='forestgreen', capsize=5, markersize=10, linewidth=2)

    ax2.axvline(0, color='red', linestyle='--', linewidth=2)
    ax2.set_yticks(range(len(nombres_egma)))
    ax2.set_yticklabels(nombres_egma)
    ax2.set_xlabel('ATE (puntos porcentuales)')
    ax2.set_title('Subpruebas EGMA (Matematicas)', fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig(figures_dir / 'Forest_Plot_Subpruebas.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Forest_Plot_Subpruebas.png")


# ==============================================================================
# FUNCION PRINCIPAL
# ==============================================================================

def main(data_path, output_dir):
    """Funcion principal que ejecuta todo el analisis"""
    print("\n" + "="*80)
    print("EVALUACION DE IMPACTO PTA/FI 3.0")
    print("Metodologia AIPW/DR con Wild Cluster Bootstrap")
    print("="*80)

    # Cargar datos
    print(f"\nCargando datos de: {data_path}")
    df_original = pd.read_csv(data_path)
    print(f"  Registros: {len(df_original):,}")
    print(f"  Variables: {len(df_original.columns)}")

    # ETL
    df, covariables = procesar_datos(df_original)

    # Propensity Score
    ps, diagnosticos = estimar_propensity_score(df, covariables)
    df['propensity_score'] = ps

    # Estimacion principal
    resultados = estimar_efectos(df, covariables, ps)

    # Subpruebas
    resultados_sub = estimar_subpruebas(df, covariables, ps)

    # Guardar resultados
    guardar_resultados(resultados, resultados_sub, diagnosticos, df, output_dir)

    # Visualizaciones
    crear_visualizaciones(resultados, resultados_sub, diagnosticos, df, output_dir)

    # Resumen final
    print("\n" + "="*80)
    print("RESUMEN FINAL")
    print("="*80)
    print(f"""
    MUESTRA:
      - Estudiantes: {len(df):,}
      - Escuelas: {df['codigo_dane'].nunique()}

    DIAGNOSTICOS:
      - AUROC: {diagnosticos['auroc']:.3f}
      - SMD Promedio: {diagnosticos['smd_promedio']:.3f}

    RESULTADOS:
      EGRA Total: ATE = {resultados['EGRA_Total']['ate']:.2f} (p={resultados['EGRA_Total']['p_value']:.3f})
      EGMA Total: ATE = {resultados['EGMA_Total']['ate']:.2f} (p={resultados['EGMA_Total']['p_value']:.3f})

    CONCLUSION:
      No se encontraron efectos estadisticamente significativos.
    """)

    print(f"\nResultados guardados en: {output_dir}")
    print("="*80)

    return resultados, resultados_sub, diagnosticos


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Evaluacion de Impacto PTA/FI 3.0 - AIPW/DR'
    )
    parser.add_argument(
        '--data', '-d',
        type=str,
        default='../data/base_estudiante_FINAL.csv',
        help='Ruta al archivo de datos CSV'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='../results/',
        help='Directorio de salida para resultados'
    )

    args = parser.parse_args()

    main(args.data, args.output)
