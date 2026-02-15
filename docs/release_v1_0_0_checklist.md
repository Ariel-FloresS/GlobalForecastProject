# GlobalForecastProject – Plan de salida a v1.0.0

## Evaluación rápida del estado actual

Fortalezas observadas:

- Existe una visión de arquitectura clara por capas (data, feature store, model, infrastructure).
- El `README.md` documenta objetivos del proyecto y ejemplos de uso orientados a Spark/Databricks.
- Ya hay una base de dependencias para correr el proyecto en entorno de forecasting distribuido.

Riesgos para lanzar una `v1.0.0` funcional:

- El repositorio todavía no incluye una batería de pruebas automatizadas mínimas de salud.
- No hay checklist de release versionado dentro del repo.
- No hay definición explícita de “Definition of Done” para declarar una versión estable.

## Criterios mínimos para declarar v1.0.0

### 1) Calidad y validación técnica

- [ ] Agregar pruebas automáticas mínimas (smoke tests + tests de contratos de entrada/salida).
- [ ] Definir y ejecutar un comando único de validación local antes de release.
- [ ] Verificar que no existan imports rotos ni módulos faltantes.

### 2) Confiabilidad del pipeline

- [ ] Validar esquema esperado de datos de entrada (columnas clave y tipos).
- [ ] Validar generación de datasets de entrenamiento, prueba y futuro.
- [ ] Validar que el pipeline maneje correctamente series con datos faltantes o inactividad.

### 3) Observabilidad y gobernanza

- [ ] Registrar metadata de ejecución (`process_date`, `version`, parámetros de corrida).
- [ ] Estandarizar logs mínimos por etapa (data prep, feature store, model).
- [ ] Definir política de versionado semántico para modelos y artefactos.

### 4) Operación y despliegue

- [ ] Definir procedimiento de despliegue reproducible (entorno, variables, credenciales).
- [ ] Documentar rollback básico (cómo volver a una versión anterior).
- [ ] Definir responsables y checklist de aprobación de release.

### 5) Documentación de uso

- [ ] Incluir quickstart con prerequisitos mínimos y comando de ejecución.
- [ ] Documentar parámetros críticos del pipeline y defaults recomendados.
- [ ] Documentar limitaciones conocidas de la v1.0.0.

## Plan sugerido de implementación de pruebas (rápido y realista)

1. **Semana corta de hardening (2–4 días):**
   - tests de estructura y salud del repositorio;
   - tests de lectura de documentación base;
   - tests de “contrato” para configuración mínima.
2. **Segundo bloque (3–5 días):**
   - tests unitarios para transformaciones clave de series temporales;
   - smoke test de flujo extremo a extremo con dataset pequeño sintético.
3. **Pre-release:**
   - congelar cambios;
   - ejecutar checklist completo;
   - tag `v1.0.0` al aprobar criterios.

## Gate recomendado para aprobar release

Se recomienda aprobar `v1.0.0` cuando se cumpla al menos:

- 100% de smoke tests en verde.
- 0 errores críticos abiertos (P0/P1) asociados al flujo principal.
- Documentación de ejecución y rollback disponible.
- Evidencia de una corrida completa y reproducible del pipeline.
