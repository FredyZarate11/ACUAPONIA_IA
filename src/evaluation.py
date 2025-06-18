import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def calculate_metrics(y_true, y_pred) -> dict:
    """Calcula un conjunto de métricas de regresión."""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # Error relativo promedio en porcentaje
    relative_error = (mae / np.mean(y_true)) * 100 if np.mean(y_true) != 0 else 0
    
    metrics = {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'Error Relativo (%)': relative_error
    }
    return metrics

def plot_evaluation_results(model_name: str, history, y_test_original, y_pred_original, metrics: dict):
    """Genera y muestra un panel de gráficos de evaluación del modelo."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 13))
    fig.suptitle(f'Evaluación del Modelo: {model_name}', fontsize=18, fontweight='bold')

    # 1. Historial de pérdida
    axes[0, 0].plot(history.history['loss'], label='Pérdida de Entrenamiento', color='blue')
    if 'val_loss' in history.history:
        axes[0, 0].plot(history.history['val_loss'], label='Pérdida de Validación', color='red', linestyle='--')
    axes[0, 0].set_title('Curva de Aprendizaje (Pérdida)', fontsize=12)
    axes[0, 0].set_xlabel('Épocas')
    axes[0, 0].set_ylabel('Pérdida (MSE)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, linestyle='--', alpha=0.6)

    # 2. Predicciones vs Valores Reales
    axes[0, 1].scatter(y_test_original, y_pred_original, alpha=0.7, color='green', edgecolor='k', s=50)
    min_val = min(y_test_original.min(), y_pred_original.min())
    max_val = max(y_test_original.max(), y_pred_original.max())
    axes[0, 1].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Predicción Perfecta')
    axes[0, 1].set_title('Predicciones vs. Valores Reales', fontsize=12)
    axes[0, 1].set_xlabel('Valores Reales (unidades originales)')
    axes[0, 1].set_ylabel('Predicciones (unidades originales)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, linestyle='--', alpha=0.6)

    # 3. Distribución de errores
    errors = y_pred_original - y_test_original
    axes[1, 0].hist(errors, bins=30, alpha=0.75, color='orange', edgecolor='black')
    axes[1, 0].axvline(errors.mean(), color='red', linestyle='--', linewidth=2, label=f'Error Promedio: {errors.mean():.2f}')
    axes[1, 0].set_title('Distribución de Errores de Predicción', fontsize=12)
    axes[1, 0].set_xlabel('Error (Predicción - Real)')
    axes[1, 0].set_ylabel('Frecuencia')
    axes[1, 0].legend()
    axes[1, 0].grid(True, linestyle='--', alpha=0.6)

    # 4. Métricas de rendimiento
    metrics_text = (
        f"Métricas de Rendimiento:\n\n"
        f"  R² (Coef. de Determinación): {metrics['R2']:.4f}\n"
        f"  MAE (Error Absoluto Medio): {metrics['MAE']:.4f}\n"
        f"  RMSE (Raíz del Error Cuadrático): {metrics['RMSE']:.4f}\n"
        f"  MSE (Error Cuadrático Medio): {metrics['MSE']:.4f}\n\n"
        f"  Error Relativo Promedio: {metrics['Error Relativo (%)']:.2f}%\n"
    )
    axes[1, 1].text(0.5, 0.5, metrics_text, fontsize=12, ha='center', va='center', 
                   bbox=dict(boxstyle='round,pad=0.5', fc='lightblue', alpha=0.5))
    axes[1, 1].set_title('Resumen de Métricas', fontsize=12)
    axes[1, 1].axis('off')

    plt.tight_layout(rect=(0, 0.03, 1, 0.95))
    plt.show()