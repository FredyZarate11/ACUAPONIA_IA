import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import datetime
import os
import io
from contextlib import redirect_stdout

def calculate_metrics(y_true, y_pred) -> dict:
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    relative_error = (mae / np.mean(y_true)) * 100 if np.mean(y_true) != 0 else 0
    metrics = { 'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R2': r2, 'Error Relativo (%)': relative_error }
    return metrics


def plot_evaluation_results(model_name: str, history, y_test_original, y_pred_original, metrics: dict, save_path=None):
    fig, axes = plt.subplots(2, 2, figsize=(16, 13))
    fig.suptitle(f'Evaluación del Modelo: {model_name}', fontsize=18, fontweight='bold')

    axes[0, 0].plot(history.history['loss'], label='Pérdida de Entrenamiento', color='blue')
    if 'val_loss' in history.history:
        axes[0, 0].plot(history.history['val_loss'], label='Pérdida de Validación', color='red', linestyle='--')
    axes[0, 0].set_title('Curva de Aprendizaje (Pérdida)', fontsize=12)
    axes[0, 0].set_xlabel('Épocas')
    axes[0, 0].set_ylabel('Pérdida (MSE)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, linestyle='--', alpha=0.6)

    axes[0, 1].scatter(y_test_original, y_pred_original, alpha=0.7, color='green', edgecolor='k', s=50)
    min_val = min(y_test_original.min(), y_pred_original.min())
    max_val = max(y_test_original.max(), y_pred_original.max())
    axes[0, 1].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Predicción Perfecta')
    axes[0, 1].set_title('Predicciones vs. Valores Reales', fontsize=12)
    axes[0, 1].set_xlabel('Valores Reales (unidades originales)')
    axes[0, 1].set_ylabel('Predicciones (unidades originales)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, linestyle='--', alpha=0.6)

    errors = y_pred_original - y_test_original
    axes[1, 0].hist(errors, bins=30, alpha=0.75, color='orange', edgecolor='black')
    axes[1, 0].axvline(errors.mean(), color='red', linestyle='--', linewidth=2, label=f'Error Promedio: {errors.mean():.2f}')
    axes[1, 0].set_title('Distribución de Errores de Predicción', fontsize=12)
    axes[1, 0].set_xlabel('Error (Predicción - Real)')
    axes[1, 0].set_ylabel('Frecuencia')
    axes[1, 0].legend()
    axes[1, 0].grid(True, linestyle='--', alpha=0.6)

    metrics_text = (f"Métricas de Rendimiento:\n\n" f"  R² (Coef. de Determinación): {metrics['R2']:.4f}\n" f"  MAE (Error Absoluto Medio): {metrics['MAE']:.4f}\n" f"  RMSE (Raíz del Error Cuadrático): {metrics['RMSE']:.4f}\n" f"  MSE (Error Cuadrático Medio): {metrics['MSE']:.4f}\n\n" f"  Error Relativo Promedio: {metrics['Error Relativo (%)']:.2f}%\n")
    axes[1, 1].text(0.5, 0.5, metrics_text, fontsize=12, ha='center', va='center', bbox=dict(boxstyle='round,pad=0.5', fc='lightblue', alpha=0.5))
    axes[1, 1].set_title('Resumen de Métricas', fontsize=12)
    axes[1, 1].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Gráfico de evaluación guardado en: {save_path}")
    
    plt.show()


def generate_report(model, model_name, metrics, config, image_path, report_path, epochs_run, training_duration):
    """Genera un informe de resultados en formato Markdown."""
    
    summary_stream = io.StringIO()
    with redirect_stdout(summary_stream):
        model.summary()
    model_summary_str = summary_stream.getvalue()

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Formatear la duración para que sea más legible
    duration_min = int(training_duration // 60)
    duration_sec = int(training_duration % 60)
    formatted_duration = f"{duration_min} min y {duration_sec} seg"

    report_content = f"""
# Informe de Resultados de Entrenamiento

- **Modelo:** `{model_name}`
- **Fecha de Ejecución:** `{timestamp}`

---

## 1. Configuración del Experimento

| Parámetro | Valor |
|---|---|
| Archivo de Datos | `{config.get('DATA_PATH', 'No especificado')}` |
| Semilla Aleatoria (Seed) | `{config.get('RANDOM_STATE', 'No especificado')}` |
| Épocas Máximas Config. | `{config.get('EPOCHS', 'No especificado')}` |
| Tamaño de Lote (Batch Size) | `{config.get('BATCH_SIZE', 'No especificado')}` |
| **Época de Detención (real)** | **{epochs_run}** |
| **Duración del Entrenamiento** | **{formatted_duration}** |

---

## 2. Arquitectura del Modelo

A continuación se muestra la arquitectura detallada de la red neuronal utilizada:

```
{model_summary_str}
```

---

## 3. Métricas de Rendimiento Final

Estas métricas fueron calculadas sobre el conjunto de prueba, utilizando los pesos del modelo que obtuvieron el mejor rendimiento en el conjunto de validación.

| Métrica | Valor |
|---|---|
| **R² (Coef. de Determinación)** | **{metrics.get('R2', 0):.4f}** |
| RMSE (Raíz del Error Cuadrático) | {metrics.get('RMSE', 0):.4f} |
| MAE (Error Absoluto Medio) | {metrics.get('MAE', 0):.4f} |
| MSE (Error Cuadrático Medio) | {metrics.get('MSE', 0):.4f} |
| Error Relativo Promedio | {metrics.get('Error Relativo (%)', 0):.2f}% |

---

## 4. Gráficos de Evaluación

La siguiente imagen muestra las curvas de aprendizaje, la comparación entre valores reales y predichos, y la distribución de los errores.

![Gráfico de Evaluación]({os.path.basename(image_path)})

### Análisis de la Curva de Aprendizaje

- **Pérdida de Entrenamiento (Azul):** Muestra cómo disminuye el error del modelo en los datos que está viendo para aprender.
- **Pérdida de Validación (Roja):** Muestra el error del modelo en datos que no ha usado para aprender. Es el indicador más importante de la capacidad de generalización del modelo. Si esta curva empieza a subir mientras la azul baja, es un síntoma de sobreajuste.

"""

    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print(f"Informe de resultados guardado en: {report_path}")
