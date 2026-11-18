# Sentiment140 · aplicación Streamlit

Aplicación local de inferencia con V3. No entrena modelos ni modifica las evidencias de V1, V2 o V3. Publicación en Community Cloud pendiente.

## Instalación y ejecución

Usa Python 3.12 (entorno original: 3.12.10). Desde la raíz del proyecto, en PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Si el entorno virtual ya existe, omite su creación. Abre http://localhost:8501. Las versiones de scikit-learn, NumPy y pandas coinciden con `configuracion_experimento.json`; joblib y SciPy corresponden al entorno local comprobado. No cambies versiones del modelo serializado sin validar compatibilidad. La instalación en Linux/Community Cloud queda por comprobar durante el despliegue.

## Uso

- **Analizar texto:** escribe una publicación en inglés. La limpieza es exactamente `limpiar_texto` de `preprocesamiento.py`. Entradas vacías o sin características reconocidas no reciben una clase forzada.
- **Dashboard:** carga un CSV con cabecera, selecciona texto y opcionalmente fecha y pulsa Analizar CSV. No requiere etiquetas. Límite: 20 MB y 50.000 filas; inferencia en lotes de 1.000. Admite UTF-8 o latin-1, coma, punto y coma o tabulación. Para un subconjunto del Sentiment140 original, marca el formato sin cabecera de seis columnas y usa latin-1.
- Los gráficos usan la **última carga analizada**, identificada por nombre de archivo. El filtro busca literalmente en el texto original. Las filas excluidas conservan su motivo; las fechas inválidas no impiden clasificar el texto. La descarga incluye todas las filas del filtro, incluso las excluidas. Los textos que podrían ser fórmulas llevan un apóstrofo de protección en la descarga.
- **Evolución:** requiere fechas válidas. Muestra cantidades diarias o semanales (lunes a domingo). Acepta ISO y el formato histórico PDT; se conserva el día local de Sentiment140. Otros timestamps con zona se normalizan a UTC. Evita fechas ambiguas: usa AAAA-MM-DD; para fechas ambiguas el analizador usa mes/día.
- **Modelo:** lee métricas y matrices guardadas. Mantiene separadas la comparación inicial (15.000) y final V2/V3 (30.000).

Las columnas `original: ...` son valores del archivo cargado. `prediccion` es la clase estimada por V3. `fecha_analisis` es la fecha parseada de la publicación, no la fecha de ejecución. Los datos cargados permanecen en la sesión; no se escriben como dataset ni se usan para entrenamiento.

## Código y artefactos

- `app.py`: navegación, caché del modelo, gráficos Plotly y explicación de resultados.
- `app_core.py`: verificación de hashes/configuración, lectura CSV, inferencia, fechas, filtros y descarga; funciones separadas para poder probarlas.
- `preprocesamiento.py`: función original, reutilizada sin modificaciones.
- `artefactos/`: solo el pipeline seleccionado y evidencias pequeñas; `manifiesto.json` registra SHA-256 y procedencia. La copia del modelo pesa aproximadamente 2,8 MB y es idéntica a `salidas_v3/busqueda_combinada_20261115_024536_bb2de0/comparacion_final_V2_V3/modelo_recomendado.joblib`.
- `.streamlit/config.toml`: tema grafito/turquesa y límite de carga.
- `tests/test_app.py`: comprobaciones técnicas con datos sintéticos, sin valor como evaluación académica.

Se verifican todos los hashes al cargar el modelo. Nunca se cargan joblib suministrados por visitantes. No se descarga el dataset al iniciar. Los archivos originales y la limpieza se conservan intactos.

## Verificación

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Detalle de cobertura y limitaciones en `evidencias_app/VERIFICACION.md`.

## Publicación desde GitHub (pendiente)

1. Revisa localmente la aplicación y las evidencias. Crea un repositorio en GitHub.
2. Sube `app.py`, `app_core.py`, `preprocesamiento.py`, `requirements.txt`, `.streamlit/config.toml`, `artefactos/`, `.gitignore` y este README. Puedes incluir `tests/` y las notas de verificación. Los scripts experimentales existentes pueden conservarse como código; no se ejecutan desde la app.
3. No subas `.venv`, `data`, `salidas`, `salidas_v2`, `salidas_v3`, credenciales o logs. `.gitignore` ya los excluye, sin borrarlos de tu computador. Verifica los archivos antes del primer commit.
4. Conecta el repositorio en Streamlit Community Cloud, selecciona la rama y `app.py` como entrada. Selecciona Python 3.12 en la configuración avanzada.
5. Comprueba que se instalen las dependencias fijadas, que cargue el modelo y que funcionen texto, CSV, descarga, filtros y fechas. No cambies el modelo ni sus versiones silenciosamente si falla la compatibilidad.
6. Solo después de comprobar el despliegue registra la URL pública, el repositorio y las capturas finales para el profesor.

Documentación oficial: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy
Dependencias: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies

## Limitaciones y sustentación

Solo inglés y clases negativa/positiva; no hay neutral. Sentiment140 contiene datos históricos de 2009 y etiquetas automáticas asociadas a emoticonos. El modelo puede fallar ante ironía, contexto, sesgos y lenguaje nuevo. Las predicciones de un CSV sin etiquetas no permiten calcular Accuracy ni F1.

Recorrido: publicación original → limpieza (HTML, minúsculas, negaciones, enlaces, menciones y espacios) → TF-IDF con vocabulario fijo → MultinomialNB entrenado → clase 0/4. No se vuelve a aprender nada durante la inferencia.

En la prueba inicial Naive Bayes obtuvo 77,64 % de Accuracy frente a 75,54 % de Random Forest. En **otra prueba**, V3 alcanzó 79,3767 % y F1 macro 0,793762, frente a V2 con 78,94 % y 0,789388. V3 obtuvo 131 aciertos adicionales y cumplió el criterio guardado de mejorar F1 sin reducir Accuracy. Es una mejora observada, no superioridad universal ni significancia estadística demostrada.

El informe académico completo, las capturas visuales definitivas, el repositorio remoto y el enlace público siguen pendientes; esta entrega prepara la aplicación y el recorrido para la sustentación.
