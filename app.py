import json
import os
import io
from pathlib import Path
from typing import Any
from xml.etree import ElementTree
from zipfile import ZipFile

import streamlit as st
from openai import OpenAI
from pypdf import PdfReader

MODEL = os.getenv("OPENAI_MODEL", "gpt-6.2-astra")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
MAX_MB = int(os.getenv("MAX_FILE_MB", "45"))

RUBRIC = [
    {"id": 1, "name": "Planteamiento y justificación del problema", "weight": 10},
    {"id": 2, "name": "Innovación y selección de tecnologías emergentes", "weight": 10},
    {"id": 3, "name": "Aplicación de hiperautomatización", "weight": 15},
    {"id": 4, "name": "Análisis de requisitos, diseño y arquitectura", "weight": 15},
    {"id": 5, "name": "Desarrollo e implementación de la solución", "weight": 15},
    {"id": 6, "name": "Pruebas, validación y resultados", "weight": 15},
    {"id": 7, "name": "Estándares técnicos, seguridad y responsabilidad tecnológica", "weight": 10},
    {"id": 8, "name": "Calidad documental, fundamentos académicos y referencias", "weight": 10},
]
FACTORS = {0: 0.0, 1: 0.30, 2: 0.60, 3: 0.80, 4: 1.0}

SYSTEM_PROMPT = """
Eres ACADEMIC REPORT EVALUATOR, un evaluador académico estricto, objetivo y basado
exclusivamente en evidencia documental. Evalúas SOLO el informe escrito del proyecto
final del curso Tecnologías Emergentes e Innovación Tecnológica, código 101552,
Ingeniería de Sistemas, Universidad Nacional Federico Villarreal, semestre 2026-2.

EXCLUSIONES:
- No evalúes exposición oral, diapositivas, defensa, participación ni desempeño individual.
- No supongas que todas las tecnologías de la Unidad 3 deben implementarse. Evalúa
  pertinencia y justificación de las tecnologías elegidas.
- No otorgues crédito por funcionalidades planificadas como si estuvieran implementadas.

CONTEXTO DEL CURSO:
- Unidad 3: hiperautomatización (RPA, process mining, Low-Code/No-Code y Digital Twins
  of Organization / DTO).
- Unidad 4: innovación y uso de tecnologías emergentes, identificación de oportunidades,
  evaluación de alternativas, toma de decisiones y estándares técnicos.
- El proyecto final debe ser implementado y su informe se entrega en semana 15.
- El sílabo no exige incorporar todas las tecnologías de la Unidad 3 en un único proyecto.

RÚBRICA (100 puntos):
1. Planteamiento y justificación del problema: 10
2. Innovación y selección de tecnologías emergentes: 10
3. Aplicación de hiperautomatización: 15
4. Análisis de requisitos, diseño y arquitectura: 15
5. Desarrollo e implementación de la solución: 15
6. Pruebas, validación y resultados: 15
7. Estándares técnicos, seguridad y responsabilidad tecnológica: 10
8. Calidad documental, fundamentos académicos y referencias: 10

NIVELES:
4 Excelente = 100% del peso; 3 Bueno = 80%; 2 En desarrollo = 60%;
1 Insuficiente = 30%; 0 No cumple = 0%.
La aplicación calculará los puntos a partir del nivel; no intentes alterar los pesos.

PROTOCOLO OBLIGATORIO:
1. Determina qué tipo de documento recibiste: informe final, cuestionario, avance,
   anexo o evidencia complementaria. Si no es el informe final, declara que la
   evaluación es diagnóstica/provisional y limita el dictamen a ese material.
2. Examina el documento completo y usa referencias localizables: página, capítulo,
   sección, tabla o figura. Si no puedes determinar una página, dilo expresamente.
3. Distingue: (a) exigencia expresa del sílabo, (b) criterio de esta rúbrica propuesta,
   (c) buena práctica académica/técnica. No presentes una buena práctica como mandato
   literal del sílabo.
4. Para cada hallazgo importante informa: ubicación, evidencia observada, problema,
   impacto y recomendación.
5. Diferencia ausencia de evidencia de evidencia de incumplimiento. Usa NE/no evaluable
   cuando corresponda; no conviertas automáticamente NE en cero. Si un criterio es NE,
   explica qué material falta y marca la nota como provisional/no cerrada.
6. No inventes fuentes, citas, páginas, resultados, pruebas, métricas, implementaciones,
   enlaces verificados ni características técnicas.
7. No declares plagio ni uso de IA sin evidencia verificable.
8. No dupliques penalizaciones por un mismo defecto salvo que afecte independientemente
   criterios distintos; explica la razón.
9. Distingue claramente entre implementado, parcialmente implementado, simulado,
   propuesto y futuro.
10. Si el documento afirma que una prueba está pendiente, no la describas como ejecutada.
11. No verifiques sitios web, repositorios ni aplicaciones externas a menos que se te
    proporcione acceso/herramienta específica; indica que no fueron verificados.
12. Mantén tono profesional, severo y constructivo. No evalúes la presentación oral.

CASO HIPERAUTOMATIZACIÓN:
La sola descripción de varios agentes, un diagrama o una arquitectura propuesta no
demuestra colaboración multiagente funcional. Para afirmar colaboración deben existir
evidencias de ejecución: entrada, decisión/orquestación, delegación, respuesta del
especialista, recepción por el supervisor y respuesta final; preferiblemente trazas
identificables. No obstante, no reduzcas todo el criterio a multiagentes si el proyecto
usa otra forma pertinente de hiperautomatización: evalúa lo que realmente implementó
y justificó.

FORMATO:
Devuelve exclusivamente un objeto JSON válido, sin Markdown ni texto fuera del JSON.
Incluye:
- document_type, scope_note, is_final_report (boolean), provisional (boolean)
- executive_summary
- rubric: lista con criterion_id, criterion, weight, level (0-4 o null si NE),
  level_name, evidence (lista de objetos con location, observed, issue, impact,
  recommendation), strengths, gaps, rationale
- findings: hallazgos transversales priorizados (priority, location, evidence, impact,
  recommendation)
- missing_evidence: lista de evidencias necesarias
- syllabus_alignment: separate explicit_syllabus_requirements, rubric_expectations,
  good_practices
- verification_limits
- final_score_out_of_100 (null si no se puede cerrar por NE)
- grade_out_of_20 (null si no se puede cerrar por NE)
- verdict (provisional / satisfactorio / requiere mejoras / insuficiente / no evaluable)
- remediation_plan: acciones ordenadas por prioridad.

No calcules ni inventes puntajes de nivel. El código calculará el total. Si un criterio
es NE, usa level=null y justifica por qué.
"""

def validate_and_score(data: dict[str, Any]) -> dict[str, Any]:
    rubric = data.get("rubric", [])
    by_id = {int(x.get("criterion_id")): x for x in rubric if x.get("criterion_id") is not None}
    scored = []
    total = 0.0
    has_ne = False

    for item in RUBRIC:
        row = by_id.get(item["id"], {})
        level = row.get("level")
        if level is None:
            has_ne = True
            points = None
        else:
            level = int(level)
            if level not in FACTORS:
                raise ValueError(f"Nivel inválido en criterio {item['id']}: {level}")
            points = round(item["weight"] * FACTORS[level], 2)
            total += points
        scored.append({
            **row,
            "criterion_id": item["id"],
            "criterion": item["name"],
            "weight": item["weight"],
            "level": level,
            "points": points,
        })

    data["rubric"] = scored
    if has_ne:
        data["final_score_out_of_100"] = None
        data["grade_out_of_20"] = None
        data["provisional"] = True
        data["verdict"] = "provisional / no evaluable parcialmente"
    else:
        data["final_score_out_of_100"] = round(total, 2)
        data["grade_out_of_20"] = round(total / 5, 2)
    return data

def extract_document_text(uploaded_file) -> str:
    suffix = Path(uploaded_file.name).suffix.lower()
    if uploaded_file.size > MAX_MB * 1024 * 1024:
        raise ValueError(f"El archivo supera el límite local de {MAX_MB} MB.")

    content = uploaded_file.getvalue()
    if suffix in {".txt", ".md"}:
        text = content.decode("utf-8-sig")
    elif suffix == ".pdf":
        reader = PdfReader(io.BytesIO(content))
        text = "\n\n".join(page.extract_text() or "" for page in reader.pages)
    elif suffix == ".docx":
        with ZipFile(io.BytesIO(content)) as document:
            root = ElementTree.fromstring(document.read("word/document.xml"))
        paragraphs = []
        for paragraph in root.iter("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}p"):
            paragraph_text = "".join(
                node.text or ""
                for node in paragraph.iter(
                    "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t"
                )
            )
            if paragraph_text:
                paragraphs.append(paragraph_text)
        text = "\n".join(paragraphs)
    else:
        raise ValueError("DeepSeek admite PDF, DOCX, TXT o MD; el formato DOC no se puede extraer.")

    if not text.strip():
        raise ValueError("No se pudo extraer texto del documento.")
    return text


def evaluate(uploaded_file, api_key: str, provider: str):
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix not in {".pdf", ".docx", ".doc", ".txt", ".md"}:
        raise ValueError("Formato no admitido. Usa PDF, DOCX, DOC, TXT o MD.")
    if uploaded_file.size > MAX_MB * 1024 * 1024:
        raise ValueError(f"El archivo supera el límite local de {MAX_MB} MB.")

    if provider == "DeepSeek":
        text = extract_document_text(uploaded_file)
        client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        response = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        "Evalúa el documento adjunto conforme a las instrucciones. "
                        "Identifica primero su naturaleza y devuelve únicamente JSON válido.\n\n"
                        f"TEXTO EXTRAÍDO DEL DOCUMENTO:\n{text}"
                    ),
                },
            ],
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content
        if not raw:
            raise ValueError("DeepSeek devolvió una respuesta vacía.")
        data = validate_and_score(json.loads(raw))
        data["model_used"] = DEEPSEEK_MODEL
        return data

    if provider != "OpenAI":
        raise ValueError(f"Proveedor no admitido: {provider}")

    client = OpenAI(api_key=api_key)
    # Upload using the Files API, then pass the file ID to Responses API.
    uploaded_file.seek(0)
    api_file = client.files.create(
        file=(uploaded_file.name, uploaded_file.getvalue()),
        purpose="user_data",
    )

    try:
        response = client.responses.create(
            model=MODEL,
            instructions=SYSTEM_PROMPT,
            input=[{
                "role": "user",
                "content": [
                    {"type": "input_text", "text":
                     "Evalúa el documento adjunto conforme a las instrucciones. "
                     "No asumas que es el informe final: identifica primero su naturaleza. "
                     "Devuelve únicamente JSON válido."},
                    {"type": "input_file", "file_id": api_file.id},
                ],
            }],
            text={"format": {"type": "json_object"}},
            store=False,
        )
        raw = response.output_text
        data = json.loads(raw)
        data = validate_and_score(data)
        data["model_used"] = MODEL
        return data
    finally:
        # Elimina el archivo remoto al terminar, incluso si falla el análisis.
        try:
            client.files.delete(api_file.id)
        except Exception:
            pass

st.set_page_config(page_title="Academic Report Evaluator", page_icon="📘", layout="wide")
st.title("ACADEMIC REPORT EVALUATOR")
st.caption("Evaluación estricta del informe escrito · Tecnologías Emergentes e Innovación Tecnológica")

with st.sidebar:
    st.subheader("Configuración")
    provider = st.selectbox("Proveedor de IA", ["OpenAI", "DeepSeek"])
    if provider == "DeepSeek":
        key_input = st.text_input(
            "DEEPSEEK_API_KEY",
            value=os.getenv("DEEPSEEK_API_KEY", ""),
            type="password",
            help="La clave se usa solo durante esta sesión. No la publiques.",
        )
        st.caption(f"Modelo configurado: {DEEPSEEK_MODEL}")
        st.caption("DeepSeek recibe el texto extraído localmente del documento. DOC no está disponible.")
    else:
        key_input = st.text_input(
            "OPENAI_API_KEY",
            value=os.getenv("OPENAI_API_KEY", ""),
            type="password",
            help="La clave se usa solo durante esta sesión. No la publiques.",
        )
        st.caption(f"Modelo configurado: {MODEL}")
    st.caption("La API puede generar costos según el modelo y el tamaño del archivo.")

st.markdown("""
**Alcance:** únicamente el informe escrito. No se evalúan exposición, diapositivas ni defensa.
La rúbrica suma 100 puntos y la nota vigesimal se calcula dividiendo entre 5.
""")

accepted_types = ["pdf", "docx", "txt", "md"]
if provider == "OpenAI":
    accepted_types.append("doc")
uploaded = st.file_uploader("Carga el informe o documento que deseas evaluar",
                            type=accepted_types)
if uploaded and st.button("Evaluar documento", type="primary", disabled=not key_input):
    with st.spinner("Analizando el documento y construyendo la evaluación..."):
        try:
            result = evaluate(uploaded, key_input, provider)
            st.session_state["evaluation"] = result
        except Exception as exc:
            st.error(f"No se pudo completar la evaluación: {exc}")

if "evaluation" in st.session_state:
    result = st.session_state["evaluation"]
    st.divider()
    st.subheader("Resultado")
    c1, c2, c3 = st.columns(3)
    c1.metric("Puntaje / 100", result.get("final_score_out_of_100")
              if result.get("final_score_out_of_100") is not None else "NE")
    c2.metric("Nota / 20", result.get("grade_out_of_20")
              if result.get("grade_out_of_20") is not None else "NE")
    c3.metric("Dictamen", result.get("verdict", "No indicado"))
    st.write(result.get("executive_summary", ""))
    st.write("**Tipo de documento:**", result.get("document_type", "No identificado"))
    st.write("**Alcance:**", result.get("scope_note", ""))

    rows = []
    for item in result.get("rubric", []):
        rows.append({
            "Criterio": item.get("criterion"),
            "Peso": item.get("weight"),
            "Nivel": item.get("level", "NE"),
            "Puntos": item.get("points", "NE"),
            "Justificación": item.get("rationale", ""),
        })
    st.dataframe(rows, use_container_width=True, hide_index=True)

    with st.expander("Informe completo"):
        st.json(result)

    payload = json.dumps(result, ensure_ascii=False, indent=2)
    st.download_button("Descargar evaluación JSON", payload,
                       file_name="evaluacion_academica.json",
                       mime="application/json")
