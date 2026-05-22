# Changelog — simple_controller_H3.py

**Basado en:** `simple_controller_H2.py` (con todas sus mejoras)  
**Objetivo:** cumplir la rúbrica del módulo y mejorar la visualización diagnóstica

---

## Cambios respecto a H2

### 1. Escala de grises restaurada como paso real del pipeline

**Qué cambia:**  
En H2 la escala de grises había sido eliminada y reemplazada completamente por HSV.
En H3 se restaura como un paso explícito que alimenta tanto el display como el detector de bordes.

```python
grey_image = greyscale_cv2(resized_bgr)   # paso 2 requerido en la rubrica
```

**Por qué:** la rúbrica lista explícitamente *"Obtención de la imagen en escala de grises"*
como paso de la secuencia. Tenerlo solo en el display pero no en el pipeline
de detección podía cuestionarse en la revisión.

---

### 2. Canny combinado: escala de grises + máscara amarilla (bitwise_or)

**Qué cambia:**

```python
# H2
canny = cv2.Canny(yellow_mask, CANNY_LOW, CANNY_HIGH)

# H3
canny_grey   = cv2.Canny(grey_image,  CANNY_LOW, CANNY_HIGH)
canny_yellow = cv2.Canny(yellow_mask, CANNY_LOW, CANNY_HIGH)
canny        = cv2.bitwise_or(canny_grey, canny_yellow)
```

**Por qué:** con `bitwise_or` ambas transformaciones contribuyen realmente a la detección.
`canny_grey` cumple la secuencia del módulo; `canny_yellow` mantiene la reducción de
falsos positivos de orillas de asfalto y sombras lograda en H2.

---

### 3. Display principal: cámara en gris + proyecciones Hough en blanco

**Qué cambia:**  
H2 mostraba el resultado del debug de bordes (fondo de Canny + líneas).
H3 muestra la imagen de cámara en escala de grises con las líneas Hough superpuestas.

```python
display_frame = grey_image.copy()
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(display_frame, (x1, y1), (x2, y2), 255, 2)
display_image(display_img, display_frame)
```

**Por qué:** la rúbrica pide *"observar la imagen capturada por la cámara"*. Mostrar el
gris de la cámara con las proyecciones encima cumple ambos requisitos: ver la cámara
y ver qué está detectando Hough.

---

### 4. Velocímetro HUD en el display principal

**Qué cambia:**  
Se agrega texto blanco superpuesto sobre el display principal con tres valores en tiempo real.

```python
display_img.setColor(0xFFFFFF)
display_img.drawText(f"V:{speed}km/h",      2, 2)
display_img.drawText(f"St:{steering:.3f}r", 2, 12)
display_img.drawText(estado,                2, 22)
# estado = "SIN LINEA" o "E:<valor>" según si hay línea detectada
```

| Campo | Ejemplo | Descripción |
|---|---|---|
| `V:50km/h` | velocidad | Velocidad de crucero constante |
| `St:0.023r` | ángulo | Ángulo de dirección en radianes |
| `E:0.12` / `E:SIN LINEA` | error o estado | Error normalizado actual o estado sin línea |

---

### 5. Display secundario con visualización del ROI en color

**Qué cambia:**  
Se agrega un segundo display (`display_image2`) que muestra el ROI con código de color.

| Color | Zona | Significado |
|---|---|---|
| **Azul** | Fuera del trapecio | Región que el ROI excluye — HoughLinesP no la analiza |
| **Blanco/gris** | Dentro del trapecio | Bordes Canny que sí recibe HoughLinesP cada frame |

```python
roi_viz = np.zeros((display_h, display_w, 3), dtype=np.uint8)
roi_viz[roi_mask == 0]    = [0, 0, 255]   # azul fuera del ROI
roi_viz[inside, 0] = roi_edges[inside]    # gris dentro del ROI
roi_viz[inside, 1] = roi_edges[inside]
roi_viz[inside, 2] = roi_edges[inside]
display_image_rgb(display_roi, roi_viz)
```

La máscara del ROI se precomputa **una sola vez** antes del loop principal
(las dimensiones no cambian durante la simulación).

**Configuración necesaria en el mundo:**  
Se debe agregar un nodo `Display` con nombre `display_image2` en `sensorsSlotTop`
del vehículo en el archivo `.wbt`. Ya está agregado en `city_2025a.wbt`:

```
Display {
  name "display_image2"
  width 200
  height 150
}
```

Si el dispositivo no existe, el código lo detecta (`display_roi = None`) y omite
el segundo display sin errores.

---

### 6. Corrección del error del PID: normalizado

**Qué cambia:**

```python
# H3 — incorrecto en versión inicial (píxeles crudos con gains de H2)
error = lane_center_x - setpoint                   # rango [-64, 64]

# H3 — corregido
error = (lane_center_x - setpoint) / setpoint      # rango [-1, 1]
```

**Por qué:** los gains `Kp=0.28, Ki=0.01, Kd=0.01` están calibrados para error
normalizado en [-1, 1]. El error en píxeles crudos con los mismos gains producía
steering de hasta 18 rad → salida inmediata del carril.

---

### 7. Comentarios pedagógicos por paso

Todos los bloques del pipeline tienen comentarios con numeración explícita
(Paso 1 a Paso 8) para corresponder con la secuencia del módulo:

```
Paso 1 — Imagen de cámara
Paso 2 — Escala de grises
Paso 3 — Canny (gris + amarillo combinados)
Paso 4 — ROI con fillPoly
Paso 5 — HoughLinesP + filtro de slope
Paso 6 — Cálculo de error normalizado
Paso 7 — Controlador PID con rate limiter
Paso 8 — Aplicar velocidad y ángulo al driver
```

---

## Tabla resumen H2 → H3

| Aspecto | H2 | H3 |
|---|---|---|
| Escala de grises | eliminada | ✅ paso explícito, alimenta Canny |
| Canny | solo yellow_mask | ✅ bitwise_or(grey, yellow) |
| Display principal | debug edges + líneas | ✅ cámara gris + líneas Hough |
| Velocímetro | no | ✅ V / Steering / Error en HUD |
| Display secundario | no | ✅ ROI en azul/blanco |
| Error PID | normalizado ✅ | normalizado ✅ (corregido bug) |
| Comentarios | notas de cambios | ✅ pedagógicos por paso |
| Hereda de H2 | — | slope 0.4, hold 10 frames, rate limiter 0.03, Kp/Ki/Kd |

---

## Resultado

| Métrica | Valor |
|---|---|
| Velocidad constante | 50 km/h |
| Cumple secuencia del módulo | ✅ pasos 1–8 |
| Tiempo sin choques | > 30 min (heredado de H2) |
| Displays activos | 2 (principal + ROI) |
