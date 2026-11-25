# ACADEMIC REPORT EVALUATOR — API OpenAI

Aplicación inicial en Python/Streamlit para evaluar exclusivamente informes escritos
del curso Tecnologías Emergentes e Innovación Tecnológica (101552).

## 1. Requisitos
- Python 3.10 o superior
- Una API key de OpenAI con acceso al modelo configurado
- Conexión a internet

## 2. Instalación
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

## 3. Configurar la API key

No escribas la clave directamente en el código.

Windows PowerShell:
```powershell
$env:OPENAI_API_KEY="tu_api_key"
$env:OPENAI_MODEL="gpt-6.2-astra"
streamlit run app.py
```

Linux/macOS:
```bash
export OPENAI_API_KEY="tu_api_key"
export OPENAI_MODEL="gpt-6.2-astra"
streamlit run app.py
```

También puedes pegar temporalmente la clave en el campo protegido de la barra lateral.
Para despliegue, configura secretos en el gestor de secretos de tu plataforma.

## 4. Uso
1. Ejecuta `streamlit run app.py`.
2. Carga el informe en PDF, DOCX, DOC, TXT o MD.
3. Pulsa **Evaluar documento**.
4. Revisa la tabla por criterios y descarga el JSON.

## 5. Rúbrica
| Criterio | Peso |
|---|---:|
| Planteamiento y justificación del problema | 10 |
| Innovación y selección de tecnologías emergentes | 10 |
| Aplicación de hiperautomatización | 15 |
| Requisitos, diseño y arquitectura | 15 |
| Desarrollo e implementación | 15 |
| Pruebas, validación y resultados | 15 |
| Estándares, seguridad y responsabilidad | 10 |
| Calidad documental, fundamentos y referencias | 10 |
| **Total** | **100** |

Niveles: 4=100%, 3=80%, 2=60%, 1=30%, 0=0%. La nota vigesimal es total/5.
Si existe un criterio no evaluable (NE), el programa no cierra la nota final.

## 6. Consideraciones importantes
- La aplicación utiliza la Responses API y el endpoint de archivos de OpenAI.
- PDF es preferible cuando el informe contiene diagramas, capturas, tablas o gráficos.
  En DOCX, la extracción de entrada de la API procesa texto, pero no necesariamente
  las imágenes incrustadas. Si la evidencia visual importa, exporta el documento a PDF.
- Los archivos se envían a la API de OpenAI para procesarlos. Revisa las políticas de
  privacidad y retención aplicables a tu cuenta y organización antes de usar documentos
  confidenciales.
- El parámetro `store=False` evita almacenar la respuesta como estado de conversación,
  pero no sustituye una revisión de las políticas de datos de la API.
- El archivo subido queda asociado al uso de Files API. Para un sistema de producción,
  añade un proceso de eliminación del archivo remoto después de procesarlo, control de
  acceso, autenticación, límites de tamaño, registros, manejo de errores y auditoría.
- La salida del modelo es una evaluación asistida, no una calificación oficial automática.
  El docente debe revisar los hallazgos, especialmente las referencias a páginas.
- Los enlaces a repositorios, sitios web o aplicaciones no se verifican por defecto.
- El modelo se configura con `OPENAI_MODEL`. Si tu cuenta no tiene acceso al valor
  predeterminado, sustitúyelo por un modelo disponible en tu proyecto API.

## 7. Arquitectura
Usuario -> interfaz Streamlit -> Files API (subida del documento) -> Responses API
(instrucciones del evaluador) -> validación y cálculo local de puntajes -> JSON descargable.

## 8. Referencias técnicas
- Responses API: https://developers.openai.com/api/docs/guides/migrate-to-responses
- Entradas de archivos: https://developers.openai.com/api/docs/guides/file-inputs
- SDK Python: https://developers.openai.com/api/reference/python
