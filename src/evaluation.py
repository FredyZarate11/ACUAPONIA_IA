import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def evaluate_and_plot(model, model_name, history, X_test, y_test):
    """Evalúa el modelo y genera gráficas y métricas de resultados."""
    print(f"\n--- Evaluando: {model_name} ---")

    # 1. Gráfica de Pérdida
    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.title(f'Curva de Aprendizaje - {model_name}')
    plt.ylabel("Pérdida (MSE)")
    plt.xlabel("Época")
    plt.legend(['Entrenamiento', 'Validación'])
    plt.grid(True)
    # plt.savefig(f'./results/{model_name}_loss.png') # Descomenta para guardar la gráfica
    plt.show()

    # 2. Métricas de Error
    loss_mse = model.evaluate(X_test, y_test, verbose=0)
    rmse = np.sqrt(loss_mse)
    predictions = model.predict(X_test, verbose=0).flatten()
    mae = np.mean(np.abs(y_test - predictions))
    
    print(f"Error (RMSE) en prueba: {rmse:.2f} g")
    print(f"Error (MAE) en prueba: {mae:.2f} g")

    # 3. Gráfica de Dispersión
    plt.figure(figsize=(8, 5))
    plt.scatter(y_test, predictions, alpha=0.7)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', linewidth=2)
    plt.title(f'Predicción vs. Real - {model_name}')
    plt.xlabel('Valor Real (g)')
    plt.ylabel('Predicción (g)')
    plt.grid(True)
    # plt.savefig(f'./results/{model_name}_scatter.png') # Descomenta para guardar
    plt.show()