"""Inferencia únicamente: reutiliza la limpieza original y nunca ajusta modelos."""
from pathlib import Path
import hashlib
import io
import json
import joblib
import pandas as pd
from preprocesamiento import limpiar_texto

ROOT = Path(__file__).resolve().parent
ART = ROOT / 'artefactos'
LABELS = {0: 'Negativo', 4: 'Positivo'}

def cargar_modelo():
    # Verificar antes de deserializar: solo artefactos locales del proyecto.
    manifest = json.loads((ART / 'manifiesto.json').read_text(encoding='utf-8'))
    for name, info in manifest.items():
        if hashlib.sha256((ART / name).read_bytes()).hexdigest() != info['sha256']:
            raise ValueError(f'El artefacto {name} no coincide con la copia verificada.')
    selection = json.loads((ART / 'seleccion_modelo.json').read_text(encoding='utf-8'))
    results = pd.read_csv(ART / 'comparacion_final.csv').set_index('version')
    if selection['version_recomendada'] != 'V3' or selection['entrada'] != 'texto_limpio':
        raise ValueError('La selección guardada no corresponde a V3 con texto limpio.')
    for metric in ('accuracy_prueba', 'f1_macro_prueba'):
        if abs(results.loc['V3', metric] - selection['metricas_nueva_prueba'][metric]) > 1e-12:
            raise ValueError('Selección y resultados inconsistentes.')
    model = joblib.load(ART / 'modelo_recomendado.joblib')
    tfidf, nb = model.steps[0][1], model.steps[1][1]
    if (tfidf.max_features, tfidf.min_df, tfidf.ngram_range, nb.alpha) != (80000, 5, (1, 2), 0.5) or list(model.classes_) != [0, 4]:
        raise ValueError('Configuración del pipeline inesperada; no se sustituirá el modelo.')
    return model

def leer_csv(raw, encoding='utf-8', sep=',', sin_cabecera=False):
    args = {'header': None, 'names': ['etiqueta_original','id','fecha','consulta','usuario','texto']} if sin_cabecera else {}
    data = pd.read_csv(io.BytesIO(raw), encoding=encoding, sep=sep, dtype=str, keep_default_na=False, nrows=50001, **args)
    if len(data) > 50000:
        raise ValueError('Máximo 50.000 filas por carga. Divide el archivo en partes.')
    return data

def fechas_reales(series):
    # Sentiment140 registra PDT: conservar el día local histórico, sin inventar fechas.
    values = series.astype('string').str.strip().str.replace(' PDT ', ' ', regex=False)
    historical = pd.to_datetime(values, format='%a %b %d %H:%M:%S %Y', errors='coerce')
    other = pd.to_datetime(values.where(historical.isna()), format='mixed', errors='coerce', utc=True).dt.tz_localize(None)
    return historical.fillna(other)

def analizar(data, columna, modelo, fecha=None, lote=1000):
    # Las columnas originales se conservan con prefijo para evitar colisiones.
    result = data.reset_index(drop=True).add_prefix('original: ')
    result['fila_csv'] = range(1, len(result) + 1)
    result['texto_original'] = data[columna].reset_index(drop=True).fillna('').astype(str)
    result['texto_limpio'] = result['texto_original'].map(limpiar_texto)
    result['estado'] = 'Sin vocabulario reconocido'
    result['prediccion'] = pd.Series(pd.NA, index=result.index, dtype='string')
    empty = result.texto_original.str.strip().eq('')
    clean_empty = result.texto_limpio.eq('')
    result.loc[empty, 'estado'] = 'Texto vacío'
    result.loc[~empty & clean_empty, 'estado'] = 'Vacío después de limpiar'
    candidates = result.index[~empty & ~clean_empty]
    for start in range(0, len(candidates), lote):
        indices = candidates[start:start+lote]
        texts = result.loc[indices, 'texto_limpio']
        matrix = modelo.steps[0][1].transform(texts)
        recognized = matrix.getnnz(axis=1) > 0
        valid = indices[recognized]
        if len(valid):
            result.loc[valid, 'prediccion'] = [LABELS[int(v)] for v in modelo.steps[1][1].predict(matrix[recognized])]
            result.loc[valid, 'estado'] = 'Analizado'
    result['fecha_analisis'] = fechas_reales(data[fecha].reset_index(drop=True)) if fecha else pd.NaT
    return result

def filtrar(result, keyword):
    return result[result.texto_original.str.contains(keyword.strip(), case=False, regex=False, na=False)]

def evolucion(result, frecuencia):
    valid = result.loc[result.estado.eq('Analizado') & result.fecha_analisis.notna()].copy()
    valid['periodo'] = valid.fecha_analisis.dt.to_period('W-SUN' if frecuencia == 'Semana' else 'D').dt.start_time
    return valid.groupby(['periodo', 'prediccion']).size().rename('Publicaciones').reset_index()

def descargar(result):
    # Evitar ejecución de fórmulas al abrir texto externo en hojas de cálculo.
    safe = result.copy()
    for col in safe.select_dtypes(include=['object', 'string']).columns:
        safe[col] = safe[col].map(lambda x: "'" + x if isinstance(x, str) and x.lstrip().startswith(('=', '+', '-', '@')) else x)
    return safe.to_csv(index=False).encode('utf-8-sig')
