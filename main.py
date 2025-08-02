import streamlit as st
import pandas as pd
import numpy as np
from transformers import pipeline
import torch

st.set_page_config(page_title="Análisis de Comentarios Docentes", layout="wide")

# === MODELO GLOBAL ===
@st.cache_resource
def cargar_modelo():
    return pipeline("sentiment-analysis",
                    model="nlptown/bert-base-multilingual-uncased-sentiment",
                    device=0 if torch.cuda.is_available() else -1)

modelo_sentimiento = cargar_modelo()

# === SUBIDA DE ARCHIVO ===
st.sidebar.header("1. Subir archivo")
archivo = st.sidebar.file_uploader("📄 Cargar archivo CSV con comentarios", type=["csv"])
if archivo:
    df = pd.read_csv(archivo)
    df['comentarios'] = df['comentarios'].astype(str)
else:
    st.warning("Por favor sube un archivo CSV para comenzar.")
    st.stop()

# === BARRA LATERAL ===
opcion = st.sidebar.selectbox(
    "Selecciona módulo:",
    ["🔍 Palabras clave de riesgo", "👨‍🏫 Análisis por docente", "📊 Ranking por severidad"]
)

# === MÓDULO 1: Palabras clave de riesgo ===
if opcion == "🔍 Palabras clave de riesgo":
    st.title("🔍 Detección de palabras de riesgo")
    
    palabras_riesgo = {
        "riesgo_psicosocial": ["estrés", "ansiedad", "depresión", "cansancio", "agobio",
                               "presión", "burnout", "tensión", "desgaste", "agotamiento"],
        "violencia_acoso": ["acoso", "hostigamiento", "intimidación", "amenaza", "agresión",
                            "violencia", "golpear", "forzar", "manoseo", "imposición"],
        "maltrato_verbal_fisico": ["gritar", "insultar", "ofender", "ridiculizar", "menospreciar",
                                   "burlarse", "humillar", "descalificar", "pegar", "empujar"],
        "vulnerabilidad_discriminación": ["discriminación", "exclusión", "racismo", "clasismo", "marginación",
                                          "desigualdad", "inequidad", "vulnerable", "preferencia", "estigmatizar"]
    }

    df['comentario_valido'] = ~df['comentarios'].str.strip().isin(['.', '-', '', ' '])
    df_validos = df[df['comentario_valido']].copy()
    df_validos['comentarios'] = df_validos['comentarios'].str.lower().str.strip()

    def detectar_categoria(texto):
        categorias = []
        for cat, palabras in palabras_riesgo.items():
            if any(p in texto for p in palabras):
                categorias.append(cat)
        return categorias

    df_validos['categorias_riesgo'] = df_validos['comentarios'].apply(detectar_categoria)
    df_riesgo = df_validos[df_validos['categorias_riesgo'].apply(lambda x: len(x) > 0)]

    st.subheader("👥 Comentarios con riesgo identificado")
    st.dataframe(df_riesgo[['id_docente', 'id_asignatura', 'comentarios', 'categorias_riesgo']])

    st.subheader("🔎 Búsqueda dentro de comentarios con riesgo")
    palabra_riesgo = st.text_input("🔍 Escribe una palabra para buscar entre los comentarios con riesgo:")
    if palabra_riesgo:
        coincidencias = df_riesgo[df_riesgo['comentarios'].str.contains(palabra_riesgo.lower(), na=False)]
        if not coincidencias.empty:
            st.success(f"Se encontraron {len(coincidencias)} coincidencias.")
            st.dataframe(coincidencias[['id_docente', 'comentarios', 'categorias_riesgo']])
        else:
            st.warning("No se encontraron coincidencias.")

    st.subheader("📌 Búsqueda en todos los comentarios")
    palabra_general = st.text_input("📌 Palabra a buscar en todos los comentarios:")
    if palabra_general:
        df['comentarios'] = df['comentarios'].astype(str)
        df['coincide_palabra'] = df['comentarios'].str.contains(palabra_general, case=False, na=False)
        df_coincidencias = df[df['coincide_palabra']].copy()

        if df_coincidencias.empty:
            st.warning(f"❌ No se encontró la palabra '{palabra_general}' en ningún comentario.")
        else:
            st.success(f"✅ Se encontraron {len(df_coincidencias)} coincidencias.")
            st.dataframe(df_coincidencias[["id_docente", "id_asignatura", "comentarios"]],
                         use_container_width=True)

# === MÓDULO 2: Análisis por docente ===
elif opcion == "👨‍🏫 Análisis por docente":
    st.title("👨‍🏫 Análisis de sentimiento por docente")
    id_docente = st.number_input("Escribe el ID del docente:", min_value=0, step=1)

    if id_docente in df['id_docente'].unique():
        df_doc = df[df['id_docente'] == id_docente].copy()
        df_doc['comentario_valido'] = ~df_doc['comentarios'].str.strip().isin(['.', '-', '', ' '])
        df_validos = df_doc[df_doc['comentario_valido']].copy()
        df_validos['comentario_limpio'] = df_validos['comentarios'].str.strip().str.lower().str.replace(r"[\.\-]", "", regex=True).str[:510]

        st.write(f"Total de comentarios: {len(df_doc)}")
        st.write(f"Comentarios válidos: {len(df_validos)}")

        predicciones = modelo_sentimiento(df_validos['comentario_limpio'].tolist())

        def mapear_sentimiento(label):
            estrellas = int(label.split()[0])
            if estrellas <= 2:
                return "NEG"
            elif estrellas == 3:
                return "NEU"
            else:
                return "POS"

        df_validos['sentimiento'] = [mapear_sentimiento(p['label']) for p in predicciones]
        conteo = df_validos['sentimiento'].value_counts()

        st.subheader("📋 Resumen de sentimientos")
        st.write(conteo.to_frame())

        for asignatura in sorted(df_validos['id_asignatura'].unique()):
            st.markdown(f"### Asignatura {asignatura}")
            for sent in ['NEG', 'NEU', 'POS']:
                subset = df_validos[(df_validos['id_asignatura'] == asignatura) & (df_validos['sentimiento'] == sent)]
                for c in subset['comentario_limpio']:
                    st.markdown(f"- ({sent}) {c}")
    else:
        st.warning("Ese ID de docente no está en el archivo.")

# === MÓDULO 3: Ranking por severidad ===
elif opcion == "📊 Ranking por severidad":
    st.title("📊 Ranking de docentes por severidad de comentarios")

    id_min = int(df['id_docente'].min())
    id_max = int(df['id_docente'].max())
    rango = st.slider("Selecciona rango de ID de docentes", min_value=id_min, max_value=id_max, value=(id_min, id_max))

    df = df[(df['id_docente'] >= rango[0]) & (df['id_docente'] <= rango[1])]
    df['comentario_valido'] = ~df['comentarios'].str.strip().isin(['.', '-', '', ' '])
    df_validos = df[df['comentario_valido']].copy()
    df_validos['comentario_limpio'] = df_validos['comentarios'].str.strip().str.lower().str.replace(r"[\.\-]", "", regex=True).str[:510]

    predicciones = modelo_sentimiento(df_validos['comentario_limpio'].tolist())
    def mapear_sentimiento(label):
        estrellas = int(label.split()[0])
        if estrellas <= 2:
            return "NEG"
        elif estrellas == 3:
            return "NEU"
        else:
            return "POS"
    df_validos['sentimiento'] = [mapear_sentimiento(p['label']) for p in predicciones]

    resumen_list = []
    for docente_id in df['id_docente'].unique():
        sub = df[df['id_docente'] == docente_id]
        sub_valid = df_validos[df_validos['id_docente'] == docente_id]
        total_validos = len(sub_valid)
        neg = (sub_valid['sentimiento'] == 'NEG').sum()
        if total_validos > 0:
            prop_neg = neg / total_validos
            indice = prop_neg * np.log1p(neg)
        else:
            prop_neg = 0
            indice = 0
        resumen_list.append({
            'id_docente': docente_id,
            'asignaturas': sub['id_asignatura'].nunique(),
            'alumnos': len(sub),
            'comentarios_validos': total_validos,
            'negativos': neg,
            'neutros': (sub_valid['sentimiento'] == 'NEU').sum(),
            'positivos': (sub_valid['sentimiento'] == 'POS').sum(),
            'proporcion_negativa': round(prop_neg, 2),
            'indice_severidad': round(indice, 4)
        })

    df_resumen = pd.DataFrame(resumen_list).sort_values(by='indice_severidad', ascending=False)
    st.dataframe(df_resumen)
