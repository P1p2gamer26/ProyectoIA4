## Análisis comparativo del experimento

**Mejor técnica en accuracy (mejor corrida por técnica):** MLP (RF=0.7070, SVM=0.6718, MLP=0.7593).

**Mejor técnica en F1 ponderado (criterio principal por desbalance 73/27):** MLP.

**Técnica más sensible a hiperparámetros** (mayor rango de Accuracy entre las 8 configuraciones): **Random Forest** (ΔAcc ≈ 0.0281). Rangos: RF=0.0281, SVM=0.0220, MLP=0.0036.

**Hiperparámetro más influyente por técnica** (mayor diferencia de Accuracy promedio al variar solo ese parámetro):
- RF: `max_depth` (ΔAcc promedio entre niveles ≈ 0.0245)
- SVM: `kernel` (Δ ≈ 0.0122)
- MLP: `learning_rate` (Δ ≈ 0.0017)

**Nota metodológica:** SVM se entrenó con muestra estratificada de 15,000 registros (de 203,400 en train) por costo computacional O(n²); RF y MLP usaron el train completo. La comparación entre técnicas debe interpretarse considerando esta diferencia.

**Conclusión general:**
- MLP obtuvo la mayor accuracy en el experimento; MLP destaca en F1 ponderado, más adecuado con clases desbalanceadas.
- Random Forest mostró mayor variabilidad ante cambios de hiperparámetros.
- El diseño factorial 2³ permitió comparar 24 configuraciones de forma sistemática sin ANOVA, identificando combinaciones favorables para el PPT y la presentación en clase.