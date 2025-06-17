import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def plot(model, model_name, history, X_test, y_test_original, y_pred_original=None):
    """
    Función para graficar resultados del modelo.
    
    Parámetros:
    - model: modelo entrenado
    - model_name: nombre del modelo para el título
    - history: historial de entrenamiento
    - X_test: datos de prueba (características)
    - y_test_original: valores reales desnormalizados
    - y_pred_original: predicciones desnormalizadas (opcional, se calculan si no se proporcionan)
    """
    
    # Si no se proporcionan las predicciones, las calculamos
    # NOTA: Esto solo funcionará si X_test está normalizado y el modelo espera datos normalizados
    if y_pred_original is None:
        y_pred_scaled = model.predict(X_test, verbose=0)
        # Aquí necesitarías el scaler_Y para desnormalizar, pero es mejor pasarlo como parámetro
        print("ADVERTENCIA: No se proporcionaron predicciones desnormalizadas. Calculando con datos normalizados.")
        y_pred_original = y_pred_scaled.flatten()
    
    # Crear figura con subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Evaluación del Modelo: {model_name}', fontsize=16, fontweight='bold')
    
    # 1. Historial de pérdida durante el entrenamiento
    axes[0, 0].plot(history.history['loss'], label='Pérdida de Entrenamiento', color='blue')
    if 'val_loss' in history.history:
        axes[0, 0].plot(history.history['val_loss'], label='Pérdida de Validación', color='red')
    axes[0, 0].set_title('Pérdida Durante el Entrenamiento')
    axes[0, 0].set_xlabel('Épocas')
    axes[0, 0].set_ylabel('Pérdida (MSE)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Predicciones vs Valores Reales (scatter plot)
    axes[0, 1].scatter(y_test_original, y_pred_original, alpha=0.6, color='green')
    
    # Línea de predicción perfecta (y = x)
    min_val = min(y_test_original.min(), y_pred_original.min())
    max_val = max(y_test_original.max(), y_pred_original.max())
    axes[0, 1].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Predicción Perfecta')
    
    axes[0, 1].set_title('Predicciones vs Valores Reales')
    axes[0, 1].set_xlabel('Valores Reales (g)')
    axes[0, 1].set_ylabel('Predicciones (g)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Distribución de errores
    errors = y_pred_original - y_test_original
    axes[1, 0].hist(errors, bins=30, alpha=0.7, color='orange', edgecolor='black')
    axes[1, 0].axvline(errors.mean(), color='red', linestyle='--', linewidth=2, label=f'Error Promedio: {errors.mean():.2f}')
    axes[1, 0].set_title('Distribución de Errores')
    axes[1, 0].set_xlabel('Error (Predicción - Real)')
    axes[1, 0].set_ylabel('Frecuencia')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Métricas de rendimiento
    mse = mean_squared_error(y_test_original, y_pred_original)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_original, y_pred_original)
    r2 = r2_score(y_test_original, y_pred_original)
    
    # Crear texto con métricas
    metrics_text = f"""
    Métricas de Rendimiento:
    
    MSE: {mse:.4f}
    RMSE: {rmse:.4f}
    MAE: {mae:.4f}
    R²: {r2:.4f}
    
    Error Relativo: {(mae/y_test_original.mean())*100:.2f}%
    
    Rango de Datos:
    Reales: {y_test_original.min():.2f} - {y_test_original.max():.2f}
    Predicciones: {y_pred_original.min():.2f} - {y_pred_original.max():.2f}
    """
    
    axes[1, 1].text(0.1, 0.9, metrics_text, transform=axes[1, 1].transAxes, 
                    fontsize=10, verticalalignment='top', 
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    axes[1, 1].set_title('Métricas de Rendimiento')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Imprimir resumen en consola
    print(f"\n=== RESUMEN DE EVALUACIÓN: {model_name} ===")
    print(f"Número de muestras de prueba: {len(y_test_original)}")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R²: {r2:.4f}")
    print(f"Error relativo promedio: {(mae/y_test_original.mean())*100:.2f}%")
    print("="*50)