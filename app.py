"""Interfaz de presentación. La inferencia y las validaciones viven en app_core."""
import json
import pandas as pd
import plotly.express as px
import streamlit as st
from app_core import ART, cargar_modelo, leer_csv, analizar, filtrar, evolucion, descargar

st.set_page_config(page_title='Sentiment140 · Análisis', page_icon='◈', layout='wide')
COLORS = {'Positivo': '#38D6C5', 'Negativo': '#F49984'}
st.markdown('''<style>.block-container{max-width:1200px;padding-top:2rem}
h1{letter-spacing:-.04em} [data-testid="stMetric"]{background:#212B36;padding:18px;border-radius:12px}
</style>''', unsafe_allow_html=True)

@st.cache_resource
def modelo_local():
    # Se reutiliza en cada interacción; jamás se ejecuta fit.
    return cargar_modelo()

try:
    modelo = modelo_local()
except Exception as exc:
    st.error(f'No se pudo verificar el modelo seleccionado: {exc}')
    st.stop()

with st.sidebar:
    st.title('Sentiment140')
    st.caption('PROYECTO DE ANÁLISIS DE SENTIMIENTO')
    page = st.radio('Navegación', ['Analizar texto', 'Dashboard', 'Evolución', 'Modelo'])
    st.divider()
    st.markdown('**V3 · TF-IDF + Naive Bayes**')
    st.caption('Inglés · Dos clases · Datos históricos')
    st.caption('No es monitoreo actual de redes sociales.')


def datos_filtrados():
    if 'resultado' not in st.session_state:
        st.info('Carga y analiza un CSV en Dashboard para explorar sus predicciones y fechas.')
        return None
    result = st.session_state.resultado
    st.caption(f"Fuente: CSV cargado «{st.session_state.fuente}» · {len(result):,} filas recibidas.")
    keyword = st.text_input('Filtrar por marca o palabra clave', key='filtro_'+page,
                            help='Busca una coincidencia literal en el texto original; no identifica marcas automáticamente.')
    subset = filtrar(result, keyword)
    st.caption(f'{len(subset):,} filas coinciden con el filtro. Los gráficos muestran predicciones del modelo, no etiquetas originales.')
    return subset

if page == 'Analizar texto':
    st.caption('01 / INFERENCIA')
    st.title('Una publicación, dos posibles clases')
    st.write('Escribe una publicación en inglés. El modelo clasifica sentimiento positivo o negativo; no reconoce una clase neutral.')
    text = st.text_area('Publicación en inglés', height=160, placeholder='Escribe aquí el texto que deseas analizar…')
    if st.button('Analizar publicación', type='primary'):
        result = analizar(pd.DataFrame({'texto': [text]}), 'texto', modelo).iloc[0]
        if result.estado == 'Analizado':
            st.metric('Predicción del modelo', result.prediccion)
            st.caption('Es una estimación y puede equivocarse. La Accuracy corresponde a una prueba completa, no a esta publicación.')
        else:
            st.warning(f'No se generó una predicción: {result.estado.lower()}. Introduce un texto en inglés con palabras reconocibles.')
        with st.expander('Recorrido de la publicación', expanded=True):
            st.write('1. Se aplica limpiar_texto de preprocesamiento.py, conservando negaciones.')
            st.code(result.texto_limpio or '(vacío)', language=None)
            st.write('2. TF-IDF transforma las palabras y pares conocidos en características numéricas.')
            st.write('3. MultinomialNB usa sus parámetros ya entrenados para asignar una clase.')
    st.info('Limitaciones: inglés, clasificación binaria y entrenamiento histórico con etiquetas automáticas derivadas de emoticonos. Ironía, contexto y cambios de lenguaje pueden causar errores.')

elif page == 'Dashboard':
    st.caption('02 / EXPLORACIÓN')
    st.title('Sentimiento en tus publicaciones')
    st.write('Carga un CSV para obtener predicciones, filtrar textos y descargar resultados. No necesitas etiquetas de sentimiento.')
    with st.expander('Cargar publicaciones', expanded='resultado' not in st.session_state):
        uploaded = st.file_uploader('Archivo CSV · hasta 20 MB y 50.000 filas', type=['csv'])
        a, b = st.columns(2)
        encoding = a.selectbox('Codificación', ['utf-8', 'latin-1'])
        separator = b.selectbox('Separador', [',', ';', '\t'])
        historical = st.checkbox('Formato original Sentiment140 sin cabecera (6 columnas)')
        st.caption('Para Sentiment140 original usa latin-1 y coma. Carga un subconjunto; la aplicación no descarga el dataset.')
        if uploaded is not None:
            try:
                data = leer_csv(uploaded.getvalue(), encoding, separator, historical)
                if data.empty:
                    st.warning('El CSV no contiene filas.')
                else:
                    st.dataframe(data.head(5), hide_index=True)
                    columns = list(data.columns)
                    text_col = st.selectbox('Columna de texto', columns, index=columns.index('texto') if 'texto' in columns else 0)
                    date_col = st.selectbox('Columna de fecha (opcional)', ['Sin fecha'] + columns)
                    st.caption('Fechas ISO (AAAA-MM-DD) recomendadas; también se acepta el formato PDT de Sentiment140. Fechas ambiguas se interpretan mes/día.')
                    if st.button('Analizar CSV', type='primary'):
                        with st.spinner('Procesando por lotes de 1.000 publicaciones…'):
                            st.session_state.resultado = analizar(data, text_col, modelo, None if date_col == 'Sin fecha' else date_col)
                            st.session_state.fuente = uploaded.name
                            st.session_state.con_fecha = date_col != 'Sin fecha'
                        st.success('CSV procesado. Los resultados corresponden a la última carga analizada.')
            except (ValueError, UnicodeError, pd.errors.ParserError) as exc:
                st.error(f'No se pudo leer el CSV: {exc}')
    subset = datos_filtrados()
    if subset is not None:
        valid = subset[subset.estado.eq('Analizado')]
        a, b, c = st.columns(3)
        a.metric('Publicaciones analizadas', f'{len(valid):,}')
        b.metric('Positivas', int(valid.prediccion.eq('Positivo').sum()))
        c.metric('Negativas', int(valid.prediccion.eq('Negativo').sum()))
        st.caption(f'{len(subset)-len(valid):,} filas excluidas de las predicciones y gráficos por texto vacío o sin características reconocidas.')
        if st.session_state.con_fecha:
            st.caption(f'{subset.fecha_analisis.isna().sum():,} fechas inválidas o ausentes; no se incluyen en Evolución, pero pueden tener predicción.')
        if not valid.empty:
            counts = valid.prediccion.value_counts().rename_axis('Sentimiento').reset_index(name='Publicaciones')
            st.plotly_chart(px.bar(counts, x='Sentimiento', y='Publicaciones', color='Sentimiento', color_discrete_map=COLORS), width='stretch')
        else:
            st.info('No hay publicaciones analizables con este filtro.')
        st.dataframe(subset, hide_index=True)
        st.caption('Las columnas «original: …» conservan los valores del CSV, incluidas sus etiquetas si existen. «prediccion» es la salida de V3.')
        st.download_button('Descargar resultados filtrados', descargar(subset), file_name='predicciones_v3.csv', mime='text/csv')

elif page == 'Evolución':
    st.caption('03 / ANÁLISIS TEMPORAL')
    st.title('Cómo cambia la distribución en el tiempo')
    subset = datos_filtrados()
    if subset is not None:
        valid = subset[subset.estado.eq('Analizado') & subset.fecha_analisis.notna()]
        if valid.empty:
            st.info('Esta sección requiere una columna de fecha con valores válidos y publicaciones analizables. Selecciónala al cargar el CSV en Dashboard.')
        else:
            start, end = valid.fecha_analisis.min(), valid.fecha_analisis.max()
            st.write(f'Periodo real de los datos: **{start:%d/%m/%Y} — {end:%d/%m/%Y}** · {len(valid):,} publicaciones con fecha válida.')
            st.caption(f'{len(subset)-len(valid):,} filas del filtro excluidas de esta serie por fecha inválida o ausencia de predicción.')
            if (valid.fecha_analisis.dt.year == 2009).all():
                st.info('Periodo histórico de 2009. Estos datos no describen el sentimiento actual en redes sociales.')
            frequency = st.radio('Agrupar por', ['Día', 'Semana'], horizontal=True)
            grouped = evolucion(subset, frequency)
            st.plotly_chart(px.bar(grouped, x='periodo', y='Publicaciones', color='prediccion', barmode='stack', color_discrete_map=COLORS,
                                   labels={'periodo': 'Fecha (semana desde lunes)', 'prediccion': 'Predicción'}), width='stretch')
            st.caption('Se muestran cantidades, no una medición representativa de toda la población. Las semanas comienzan el lunes.')
            st.dataframe(grouped, hide_index=True)

else:
    st.caption('04 / EVIDENCIA Y ALCANCE')
    st.title('Por qué seleccionamos V3')
    initial = pd.read_csv(ART / 'comparacion_inicial.csv')
    final = pd.read_csv(ART / 'comparacion_final.csv')
    st.write('Naive Bayes funcionó mejor entre los clasificadores y configuraciones evaluados. Esto no demuestra que siempre sea superior a Random Forest.')
    def comparison(frame, name):
        display = frame[[name, 'accuracy_prueba', 'f1_macro_prueba']].copy()
        display.columns = ['Modelo', 'Accuracy (%)', 'F1 macro']
        display['Accuracy (%)'] *= 100
        st.dataframe(display.style.format({'Accuracy (%)': '{:.4f}', 'F1 macro': '{:.6f}'}), hide_index=True)
        st.plotly_chart(px.bar(display, x='Modelo', y='Accuracy (%)', range_y=[0,100], color_discrete_sequence=['#38D6C5']), width='stretch')
    st.subheader('Comparación inicial · misma prueba de 15.000 publicaciones')
    comparison(initial, 'modelo')
    st.subheader('Comparación final V2 / V3 · otra prueba de 30.000 publicaciones')
    comparison(final, 'version')
    summary = json.loads((ART / 'resumen_final.json').read_text(encoding='utf-8'))
    balance = summary['balance']
    st.write(f"V3 consiguió {balance['aciertos_netos_adicionales_V3']} aciertos adicionales: +{balance['cambio_accuracy_pp_V3_menos_V2']:.4f} puntos porcentuales de Accuracy. Aumentó F1 macro sin reducir Accuracy, cumpliendo el criterio guardado.")
    st.warning('Son dos pruebas distintas. No se combinan en una clasificación única. La mejora observada no demuestra significancia estadística ni superioridad universal.')
    st.subheader('Cómo interpretar las métricas')
    st.write('**Accuracy:** proporción total de publicaciones clasificadas correctamente. No es la certeza de una predicción individual.')
    st.write('**F1 macro:** calcula F1 para cada clase, combinando precisión y recuperación, y promedia ambas con el mismo peso. Ayuda a valorar el desempeño en las dos clases.')
    st.write('**Matriz de confusión:** las filas son etiquetas reales del conjunto de prueba y las columnas son predicciones. La diagonal representa aciertos y las otras celdas, errores. Las etiquetas de Sentiment140 son automáticas, no juicios humanos infalibles.')
    matrix_name = st.selectbox('Matriz de confusión', ['V3 · prueba final', 'V2 · prueba final', 'Naive Bayes · prueba inicial', 'Random Forest · prueba inicial'])
    file = dict(zip(['V3 · prueba final', 'V2 · prueba final', 'Naive Bayes · prueba inicial', 'Random Forest · prueba inicial'], ['matriz_V3.csv','matriz_V2.csv','matriz_inicial_nb.csv','matriz_inicial_rf.csv']))[matrix_name]
    matrix = pd.read_csv(ART / file, index_col=0)
    st.plotly_chart(px.imshow(matrix, text_auto=True, color_continuous_scale=['#212B36','#38D6C5'], aspect='auto', labels={'x':'Predicción','y':'Etiqueta original','color':'Publicaciones'}), width='stretch')
    st.subheader('Configuración y limitaciones')
    config = json.loads((ART / 'configuracion_experimento.json').read_text(encoding='utf-8'))
    st.write(f"Entrenamiento V3: {config['entrenamiento']:,} publicaciones. TF-IDF: hasta 80.000 características, palabras y pares consecutivos, min_df=5. MultinomialNB: alpha=0.5.")
    st.write('Solo inglés y dos clases (0 negativo, 4 positivo). Datos históricos de 2009, con etiquetas automáticas derivadas de emoticonos; posibles sesgos y errores. No hay clase neutral ni garantías ante ironía, otros idiomas o lenguaje actual.')
    st.caption('Fuente: artefactos/comparacion_inicial.csv y comparacion_final.csv, matrices y JSON copiados de los resultados originales. El manifiesto registra rutas originales y SHA-256. No se entrena ni se ajusta el modelo en esta aplicación.')
