# PREDWEEM LOLIUM — Plataforma multisitio

Repositorio integrador para la predicción operativa de emergencia de *Lolium* en nueve localidades.

## Arquitectura de repositorios

```text
PREDWEEM/MULTISITIO
└── aplicación regional, catálogo de sitios y actualización meteorológica integrada

PREDWEEM/LOLIUM_ZAVALLA2026
└── aplicación y meteorología independiente de Zavalla

PREDWEEM/lolium_sanpedro2026
└── motor local de referencia para la parametrización final de San Pedro
```

## Localidades y política operativa

| Localidad | Modelo |
|---|---|
| Azul | Sin lag |
| Balcarce | Sin lag |
| Bordenave | Sin lag |
| Lartigau | Sin lag |
| Olavarría | Sin lag |
| San Pedro | Sin lag · motor final SP-FINAL-2025-2026 |
| Tres Arroyos | Sin lag |
| Pergamino | Lag fijo de 15 días |
| Zavalla | Lag fijo de 15 días |

El catálogo geográfico y la política de modelo se definen en `sitios_lolium.py`. Las calibraciones generales de las nueve localidades se encuentran en `config_multisitio.py`. San Pedro incorpora además una capa local congelada y auditable en `sanpedro_calibracion_final.py`, sincronizada con `PREDWEEM/lolium_sanpedro2026`.

## San Pedro — versión final 2025–2026

MULTISITIO reproduce para San Pedro la parametrización final calibrada con las campañas completas 2025 y 2026. La ANN original permanece sin modificaciones.

Parámetros congelados:

- cobertura efectiva: `17.117654787448153 %`;
- Wmax superficial: `21.835296520312887 mm`;
- T0 termohídrico: `24.554776252844984 °C`;
- ventana termohídrica: `7 días`;
- alpha hídrica: `0.05239929751806027 °C`;
- pendiente logística: `0.36599312676568485 °C`;
- agotamiento causal de cohorte: `k = 0.4302565796601996`;
- choque hídrico: `45 mm / 3 días`;
- Kr: `0`;
- primer pico válido: `EMERREL > 0.20`.

La termoinhibición binaria se reemplaza exclusivamente en San Pedro por una interacción continua temperatura × humedad. Después del primer pico válido se aplica un reservorio causal de cohorte. La cobertura y Wmax se presentan bloqueados en la interfaz para impedir modificaciones accidentales de la calibración local. Los parámetros exactos se conservan también en `sanpedro_calibration_2025_2026.json`.

## Archivos operativos

```text
app.py                         entrada de Streamlit
app_sanpedro_final.py          integración del motor final de San Pedro
app_multisitio_principal.py    interfaz regional
app_multisitio.py              utilidades de simulación e interfaz
app_fuente_hibrida.py          trazabilidad meteorológica de Zavalla
app_detalle_1pct.py            detalle de baja emergencia
app_umbral_operativo.py        criterio EMERREL >= 0,0001
app_zoom_operativo.py          zoom, paneles y mapa

sanpedro_calibracion_final.py  termohidria + agotamiento causal de San Pedro
sanpedro_calibration_2025_2026.json
                               parámetros congelados y trazabilidad
sitios_lolium.py               catálogo de localidades
config_multisitio.py           calibraciones y parámetros operativos generales
predweem_core.py               motor ANN y ecofisiológico común
visualizacion_operativa.py     gráficos principales
visualizacion_pulsos.py        agrupación de pulsos
mapa_sitios.py                 mapa regional

update_meteo.py                punto de entrada meteorológico seguro
update_meteo_core.py           motor de actualización
update_meteo_runtime.py        alias compatible del actualizador

data/meteo_sitios/*.csv       series meteorológicas por localidad
data/estado_actualizacion_meteo.json

IW.npy, bias_IW.npy, LW.npy, bias_out.npy
                               activos de la red neuronal
```

Los módulos encadenados desde `app.py` se conservan porque forman parte de la interfaz multisitio activa. No son aplicaciones independientes.

## Meteorología

Para Azul, Balcarce, Bordenave, Lartigau, Olavarría, Pergamino, San Pedro y Tres Arroyos se mantienen copias exactas de los archivos meteorológicos de sus repositorios geográficos.

Para Zavalla se aplica prioridad por variable:

```text
SMN Rosario Aero 87480
→ NOAA NCEI, cuando esté disponible
→ Open-Meteo ECMWF IFS Archive para faltantes
→ Open-Meteo ECMWF IFS Forecast desde el día actual
```

Las columnas `Fuente_TMAX`, `Fuente_TMIN` y `Fuente_Prec` registran la procedencia efectiva. Todas las series se almacenan únicamente en `data/meteo_sitios/`; la antigua copia raíz `meteo_daily.csv` fue eliminada.

## Componentes eliminados

La plataforma no selecciona modelos mediante conteos de campo y no mantiene archivos propios de una única localidad en la raíz. Por ese motivo no deben existir en `main`:

```text
selector_adaptativo.py
data/inspecciones_campo.csv
data/selector_estado.json
config_zavalla.py
meteo_daily.csv
```

La ausencia de estos archivos y la presencia de las nueve localidades se controlan en `tests/test_multisitio_registry.py`.

## Ejecución

```bash
python -m pip install -r requirements.txt
python update_meteo.py
streamlit run app.py
```

## Validación

```bash
python -m py_compile \
  app.py app_sanpedro_final.py app_multisitio.py app_multisitio_principal.py \
  sanpedro_calibracion_final.py config_multisitio.py sitios_lolium.py \
  predweem_core.py visualizacion_operativa.py visualizacion_pulsos.py \
  update_meteo.py update_meteo_core.py

python -m pytest tests/test_multisitio_registry.py tests/test_sanpedro_final.py -q
```

La rama `pre-depuracion-multisitio-20260802` conserva el estado inmediatamente anterior a esta limpieza.

**PREDWEEM by Guillermo R. Chantre**
