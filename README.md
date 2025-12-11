# Evaluacion de Impacto del Programa PTA/FI 3.0

## Metodologia AIPW/DR con Wild Cluster Bootstrap

---

## Descripcion

Este repositorio contiene el codigo y datos para la evaluacion de impacto del programa **Todos a Aprender / Formacion Integral (PTA/FI) 3.0** sobre los resultados de aprendizaje en lectura (EGRA) y matematicas (EGMA).

### Metodologia

Se utiliza el estimador **Augmented Inverse Propensity Weighting (AIPW)**, tambien conocido como **Doubly Robust (DR)**, que combina:

1. **Inverse Propensity Weighting (IPW)**: Pondera observaciones por probabilidad de tratamiento
2. **Regression Adjustment (RA)**: Modela directamente los resultados potenciales

El estimador es **doblemente robusto**: produce estimaciones consistentes si:
- El modelo de propension esta correctamente especificado, **O**
- Los modelos de resultado estan correctamente especificados

### Inferencia

- **Wild Cluster Bootstrap** con pesos Rademacher (B=1000)
- Clustering a nivel de escuela (codigo DANE)
- Intervalos de confianza al 95%

---

## Estructura del Repositorio

```
EVALUACION_IMPACTO_PTAFI_GIT/
|
|-- notebooks/
|   |-- AIPWDR_Evaluacion_Impacto.ipynb    # Notebook principal interactivo
|
|-- scripts/
|   |-- aipwdr_evaluacion_impacto.py       # Script Python ejecutable
|
|-- data/
|   |-- base_estudiante_FINAL.csv          # Base de datos de estudiantes
|
|-- results/
|   |-- resultados_finales.csv             # Resultados EGRA y EGMA Total
|   |-- resultados_subpruebas.csv          # Resultados por subprueba
|   |-- diagnosticos.csv                   # Metricas de diagnostico
|   |-- figures/
|       |-- Forest_Plot_Principal.png      # Forest plot efectos principales
|       |-- Forest_Plot_Subpruebas.png     # Forest plot subpruebas
|       |-- Love_Plot_SMD.png              # Balance de covariables
|       |-- Diagnosticos_modelo.png        # Diagnosticos del modelo
|       |-- Analisis_Exploratorio.png      # Analisis exploratorio
|       |-- Resultados_finales.png         # Visualizacion resultados
|
|-- requirements.txt                        # Dependencias Python
|-- README.md                               # Este archivo
```

---

## Instalacion

### Requisitos

- Python 3.9 o superior
- pip (gestor de paquetes)

### Pasos

1. Clonar el repositorio:
```bash
git clone https://github.com/usuario/EVALUACION_IMPACTO_PTAFI_GIT.git
cd EVALUACION_IMPACTO_PTAFI_GIT
```

2. Crear entorno virtual (recomendado):
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. Instalar dependencias:
```bash
pip install -r requirements.txt
```

---

## Uso

### Opcion 1: Jupyter Notebook (Recomendado)

```bash
jupyter notebook notebooks/AIPWDR_Evaluacion_Impacto.ipynb
```

Ejecutar las celdas secuencialmente. El notebook incluye:
- Explicaciones metodologicas
- Visualizaciones interactivas
- Exportacion automatica de resultados

### Opcion 2: Script Python

```bash
cd scripts
python aipwdr_evaluacion_impacto.py --data ../data/base_estudiante_FINAL.csv --output ../results/
```

Parametros:
- `--data` o `-d`: Ruta al archivo CSV con los datos
- `--output` o `-o`: Directorio de salida para resultados

---

## Resultados Principales

### Efectos del Tratamiento (ATE)

| Outcome | ATE | SE | IC 95% | p-valor | Sig. |
|---------|-----|-----|--------|---------|------|
| EGRA Total | -1.41 | 2.69 | [-6.84, 3.72] | 0.601 | No |
| EGMA Total | -1.74 | 2.39 | [-6.41, 2.83] | 0.467 | No |

### Subpruebas EGRA (Lectura)

| Subprueba | ATE | SE | p-valor |
|-----------|-----|-----|---------|
| CSL (Sonidos Letras) | 1.37 | 2.81 | 0.632 |
| DPSS (Pseudopalabras) | 3.73 | 3.31 | 0.273 |
| LP (Lectura Palabras) | -0.91 | 2.70 | 0.740 |
| CL (Comprension Lectora) | 1.41 | 3.27 | 0.670 |

### Subpruebas EGMA (Matematicas)

| Subprueba | ATE | SE | p-valor |
|-----------|-----|-----|---------|
| Numeros Faltantes | -1.37 | 2.27 | 0.552 |
| Comparacion/Operaciones | -5.02 | 2.94 | 0.102 |
| Sumas | 2.03 | 3.01 | 0.507 |
| Restas | 3.73 | 2.89 | 0.210 |
| Multiplicacion/Division | 2.22 | 3.94 | 0.579 |
| Resolucion Problemas | -0.29 | 4.45 | 0.949 |

### Diagnosticos del Modelo

| Metrica | Valor | Interpretacion |
|---------|-------|----------------|
| AUROC | 0.803 | Rango optimo [0.6, 0.9] |
| SMD Promedio | 0.292 | Balance aceptable |
| ESS Tratamiento | 96.5% | Minima perdida |
| ESS Control | 85.5% | Aceptable |

---

## Muestra

- **N Estudiantes**: 770
- **N Escuelas**: 22 (11 tratamiento, 11 control)
- **Tratados**: 383 estudiantes
- **Control**: 387 estudiantes

---

## Variables

### Outcomes

- **EGRA Total**: Promedio de 4 subpruebas de lectura (0-100%)
- **EGMA Total**: Promedio de 6 subpruebas de matematicas (0-100%)

### Covariables

- `edad`: Edad del estudiante
- `sexo`: Genero (0=F, 1=M)
- `ruralidad`: Indicador zona rural
- `pdet`: Municipio PDET
- `tei_nivel_pre`: Nivel TEI previo
- `matricula_pre`: Matricula previa
- `pct_jornada_completa`: Porcentaje jornada completa
- `composicion_grados`: Composicion de grados
- `zona_*`: Dummies de zona geografica

---

## Especificaciones Tecnicas

### Propensity Score
- Logistic Regression con regularizacion Ridge (L2)
- Cross-validation para seleccionar C optimo
- Clipping [0.01, 0.99] para estabilidad

### Modelos de Resultado
- Gradient Boosting Regressor
- Cross-Fitting con GroupKFold por escuela
- 100 estimadores, max_depth=3

### Inferencia
- Wild Cluster Bootstrap (B=1000 para totales, B=500 para subpruebas)
- Pesos Rademacher {-1, +1}
- IC 95% percentil

---

## Conclusiones

**No se encontraron efectos estadisticamente significativos del programa PTA/FI 3.0** sobre los resultados de EGRA (lectura) ni EGMA (matematicas).

Los intervalos de confianza incluyen el cero en todos los casos, y los p-valores son mayores a 0.10 para todos los outcomes y subpruebas analizadas.

---

## Autores

Equipo de Evaluacion PTA/FI 3.0

## Fecha

Diciembre 2025

## Licencia

Este proyecto es de uso interno. Consultar con los autores para permisos de uso.
