# simple_controller_H3.py
# Controlador de seguimiento de carril con PID
# Pipeline: Camara -> Escala de grises (display) + HSV amarillo (deteccion)
#           -> Canny -> ROI -> HoughLinesP -> PID -> SteeringAngle

from controller import Display, Keyboard, Robot, Camera
from vehicle import Car, Driver
import numpy as np
import cv2
from datetime import datetime
import os
import time

# ── Constantes de configuracion ────────────────────────────────────────────────
DEBOUNCE_TIME = 0.1       # segundos minimos entre pulsaciones de tecla
MAX_ANGLE     = 0.5       # maximo angulo de direccion en radianes
SPEED_INCR    = 5         # incremento de velocidad en modo manual
ANGLE_INCR    = 0.05      # incremento de angulo en modo manual

# Filtro de slope: rechaza lineas casi horizontales (cebras, cruces peatonales).
# Una linea con |slope| < 0.4 forma un angulo menor a ~22 grados con el eje X
# y no representa la linea central de carril.
MIN_ABS_SLOPE = 0.4

# ── Parametros HSV para segmentacion de color amarillo ────────────────────────
# OpenCV representa H en [0,180] (escala completa 360 / 2).
# El amarillo cae en H ≈ 30 (60 grados reales / 2).
# Rango [15, 35] captura variaciones por iluminacion, distancia y perspectiva.
# S >= 80 y V >= 80 descartan amarillos palidos, blancos y sombras.
YELLOW_LOW  = np.array([15,  80,  80], dtype=np.uint8)
YELLOW_HIGH = np.array([35, 255, 255], dtype=np.uint8)

# ── Parametros Canny ───────────────────────────────────────────────────────────
CANNY_LOW  = 50    # gradiente debajo de este valor: descartado
CANNY_HIGH = 150   # gradiente encima de este valor: aceptado como borde

# ── Parametros HoughLinesP ────────────────────────────────────────────────────
HOUGH_RHO        = 1            # resolucion de distancia en pixeles
HOUGH_THETA      = np.pi / 180  # resolucion angular: 1 grado
HOUGH_THRESHOLD  = 20           # votos minimos para aceptar una linea
HOUGH_MIN_LENGTH = 20           # longitud minima del segmento en pixeles
HOUGH_MAX_GAP    = 15           # brecha maxima entre segmentos para unirlos

# ── Ganancias del controlador PID ─────────────────────────────────────────────
# Kp (proporcional): reaccion inmediata al error actual.
# Ki (integral):     correccion de deriva acumulada a lo largo del tiempo.
#                    Con error normalizado y dt~0.032 s, Ki=0.01 tarda ~40 s
#                    en saturar el integral — evita sesgo permanente.
# Kd (derivada):     amortigua oscilaciones reaccionando a la velocidad de cambio.
Kp = 0.28
Ki = 0.01
Kd = 0.01

# Maximo cambio de angulo de direccion por frame (~32 ms).
# Limita el efecto de cualquier deteccion erronea puntual.
MAX_STEER_RATE = 0.03


# ── Funciones de vision ────────────────────────────────────────────────────────

def get_image(camera):
    """Lee la imagen raw de la camara y la convierte a array numpy BGRA."""
    raw = camera.getImage()
    return np.frombuffer(raw, np.uint8).reshape(
        (camera.getHeight(), camera.getWidth(), 4)
    )


def greyscale_cv2(image):
    """
    Convierte imagen BGR a escala de grises.
    Paso requerido en la secuencia del modulo: colapsa los 3 canales de color
    en un unico canal de luminancia, simplificando el procesamiento posterior.
    Se usa para mostrar la imagen de camara en el display integrado.
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def display_image(display, image):
    """Muestra imagen en escala de grises en el display integrado de Webots."""
    image_rgb = np.dstack((image, image, image))
    image_ref = display.imageNew(
        image_rgb.tobytes(),
        Display.RGB,
        width=image_rgb.shape[1],
        height=image_rgb.shape[0],
    )
    display.imagePaste(image_ref, 0, 0, False)
    display.imageDelete(image_ref)


def display_image_rgb(display, image_rgb):
    """Muestra imagen RGB directamente en el display de Webots."""
    image_ref = display.imageNew(
        image_rgb.tobytes(),
        Display.RGB,
        width=image_rgb.shape[1],
        height=image_rgb.shape[0],
    )
    display.imagePaste(image_ref, 0, 0, False)
    display.imageDelete(image_ref)


def apply_roi(edges, height, width):
    """
    Define y aplica una Region de Interes (ROI) trapezoidal sobre la imagen de bordes.

    La ROI se enfoca en el 40% inferior de la imagen donde la linea del carril
    es visible desde la perspectiva de la camara. Pixeles fuera de la ROI se
    anulan con cero para que HoughLinesP solo analice la zona relevante.

    Vertices del trapecio (coordenadas imagen, origen arriba-izquierda):
      - Inferior-izquierdo : (10% del ancho, alto total)
      - Superior-izquierdo : (35% del ancho, 60% del alto)
      - Superior-derecho   : (65% del ancho, 60% del alto)
      - Inferior-derecho   : (90% del ancho, alto total)

    El techo estrecho (35%-65%) evita capturar lineas del horizonte o cielo.
    La base ancha (10%-90%) captura la linea en curvas moderadas.

    Implementacion: fillPoly dibuja el trapecio en blanco sobre una mascara
    negra, luego bitwise_and extrae solo los bordes dentro de la ROI.
    """
    mask = np.zeros_like(edges)
    vertices = np.array([[
        (int(width * 0.10), height),
        (int(width * 0.35), int(height * 0.6)),
        (int(width * 0.65), int(height * 0.6)),
        (int(width * 0.90), height)
    ]], dtype=np.int32)
    cv2.fillPoly(mask, vertices, 255)
    return cv2.bitwise_and(edges, mask)


def hough_lines(roi_edges):
    """
    Aplica HoughLinesP a la imagen de bordes con ROI.

    HoughLinesP detecta segmentos de linea (no lineas infinitas).
    Cada segmento debe acumular al menos HOUGH_THRESHOLD votos en el espacio
    de Hough y tener longitud >= HOUGH_MIN_LENGTH px.
    Segmentos separados por <= HOUGH_MAX_GAP px se unen en uno solo.

    Retorna array de forma (N, 1, 4) con [x1, y1, x2, y2] por segmento,
    o None si no se detectaron lineas.
    """
    return cv2.HoughLinesP(
        roi_edges,
        HOUGH_RHO,
        HOUGH_THETA,
        HOUGH_THRESHOLD,
        minLineLength=HOUGH_MIN_LENGTH,
        maxLineGap=HOUGH_MAX_GAP
    )


def filter_lines_by_slope(lines, min_abs_slope=MIN_ABS_SLOPE):
    """
    Filtra lineas que sean casi horizontales.

    Las marcas de cebras peatonales y cruces de interseccion producen
    segmentos con slope cercano a cero que no representan la linea de carril.
    Se descartan todos los segmentos con |slope| < min_abs_slope.
    Lineas verticales (x1==x2) se omiten para evitar division por cero.

    Retorna array filtrado o None si no quedan lineas.
    """
    if lines is None:
        return None

    filtered = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 == x1:
            continue
        slope = (y2 - y1) / (x2 - x1)
        if abs(slope) >= min_abs_slope:
            filtered.append(line)

    return np.array(filtered) if filtered else None


def compute_lane_center(lines):
    """
    Calcula el centro estimado del carril a partir de las lineas detectadas.

    Para cada linea se calcula el punto medio horizontal: mid_x = (x1 + x2) / 2.
    Las lineas se clasifican por el signo de su slope:
      - slope < 0: linea izquierda  (en imagen, pendiente negativa = borde izquierdo)
      - slope > 0: linea derecha

    Si hay puntos de ambos lados: centro = promedio(izquierda + derecha) / 2.
    Si solo hay puntos de un lado: centro = promedio de todos los puntos.
    Esto permite funcionar en tramos donde solo es visible un borde del carril.

    Retorna la posicion x del centro estimado, o None si no hay lineas validas.
    """
    if lines is None:
        return None

    left_x, right_x, all_x = [], [], []

    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 == x1:
            continue
        slope = (y2 - y1) / (x2 - x1)
        mid = (x1 + x2) / 2
        all_x.append(mid)
        if slope < 0:
            left_x.append(mid)
        else:
            right_x.append(mid)

    if left_x and right_x:
        return (np.mean(left_x) + np.mean(right_x)) / 2.0
    if all_x:
        return np.mean(all_x)
    return None


# ── Programa principal ─────────────────────────────────────────────────────────

def main():
    # Velocidad constante en km/h — se mantiene fija durante todo el recorrido
    speed = 50

    # Estado del PID
    integral       = 0.0
    previous_error = 0.0
    previous_time  = time.time()
    steering       = 0.0

    # Contador de frames consecutivos sin linea detectada
    # Permite mantener el angulo previo al cruzar zonas sin marcas (intersecciones)
    no_line_frames = 0

    last_press = {}

    # Inicializacion de Webots
    robot    = Car()
    driver   = Driver()
    timestep = int(robot.getBasicTimeStep())

    camera = robot.getDevice("camera")
    camera.enable(timestep)

    # Display principal: escala de grises + lineas Hough + velocimetro
    display_img = robot.getDevice("display_image")

    # Display secundario: visualizacion del ROI (bordes Canny enmascarados).
    # Requiere un segundo dispositivo "display_image2" en el mundo de Webots.
    # Si no existe en el .wbt, display_roi queda en None y se omite sin error.
    display_roi = robot.getDevice("display_image2")

    keyboard = Keyboard()
    keyboard.enable(timestep)

    # Setpoint: centro horizontal de la imagen.
    # El PID intentara mantener la linea detectada en esta posicion x.
    # Se calcula con las dimensiones reales del display para coincidir con
    # el espacio donde se detectan las lineas.
    display_w = display_img.getWidth()
    display_h = display_img.getHeight()
    setpoint  = display_w / 2.0

    # Mascara del ROI precomputada: mismos vertices que apply_roi.
    # Se calcula una sola vez fuera del loop porque las dimensiones no cambian.
    roi_mask = np.zeros((display_h, display_w), dtype=np.uint8)
    cv2.fillPoly(roi_mask, np.array([[
        (int(display_w * 0.10), display_h),
        (int(display_w * 0.35), int(display_h * 0.6)),
        (int(display_w * 0.65), int(display_h * 0.6)),
        (int(display_w * 0.90), display_h)
    ]], dtype=np.int32), 255)

    driver.setCruisingSpeed(speed)

    while robot.step() != -1:
        current_time = time.time()
        dt = current_time - previous_time
        if dt <= 0:
            dt = 1e-3

        # ── Paso 1: Obtener imagen de la camara ───────────────────────────────
        image = get_image(camera)

        # ── Paso 2: Escala de grises ──────────────────────────────────────────
        # Conversion de BGRA a BGR para compatibilidad con OpenCV, luego a gris.
        # La imagen en escala de grises se muestra en el display integrado para
        # que se pueda observar lo que capta la camara durante la simulacion.
        bgr_image   = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        resized_bgr = cv2.resize(bgr_image, (display_w, display_h))
        grey_image  = greyscale_cv2(resized_bgr)

        # ── Segmentacion HSV (mejora sobre el pipeline base) ──────────────────
        # En escala de grises, Canny detecta todos los bordes de contraste:
        # orillas del asfalto, sombras, texturas. Muchos generan lineas con
        # slope valido que corrompian el calculo del centro del carril.
        # El filtro HSV amarillo limita la entrada a Canny solo a objetos amarillos,
        # reduciendo drasticamente los falsos positivos sin alterar la secuencia
        # requerida (escala de grises sigue presente como paso de observacion).
        hsv         = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2HSV)
        yellow_mask = cv2.inRange(hsv, YELLOW_LOW, YELLOW_HIGH)

        # ── Paso 3: Deteccion de bordes con Canny ─────────────────────────────
        # Canny usa dos umbrales: gradiente > HIGH = borde seguro;
        # entre LOW y HIGH = aceptado si conecta con borde seguro; < LOW = descartado.
        #
        # Se aplica por separado sobre la imagen en escala de grises y sobre la
        # mascara amarilla, y se combinan con bitwise_or:
        #   - canny_grey   captura todos los bordes de contraste de la escena
        #   - canny_yellow filtra solo bordes de objetos amarillos
        # La union aprovecha ambas fuentes: la escala de grises aporta la secuencia
        # requerida en los materiales del modulo; el filtro amarillo reduce
        # los falsos positivos de orillas de asfalto y sombras.
        canny_grey   = cv2.Canny(grey_image,   CANNY_LOW, CANNY_HIGH)
        canny_yellow = cv2.Canny(yellow_mask,  CANNY_LOW, CANNY_HIGH)
        canny        = cv2.bitwise_or(canny_grey, canny_yellow)

        # ── Paso 4: Region de Interes con fillPoly ────────────────────────────
        # Enmascara el 60% superior de la imagen para enfocarse solo en el
        # tramo de carretera inmediatamente frente al vehiculo.
        roi_edges = apply_roi(canny, display_h, display_w)

        # ── Paso 5: Transformada de Hough (HoughLinesP) ───────────────────────
        # Detecta segmentos de linea en la ROI. El resultado es una lista de
        # segmentos [x1, y1, x2, y2]. Se filtran los casi-horizontales.
        raw_lines = hough_lines(roi_edges)
        lines     = filter_lines_by_slope(raw_lines)

        # ── Paso 6: Calculo del error para el PID ─────────────────────────────
        # Se obtiene la posicion x del centro estimado del carril.
        # Error = distancia entre ese centro y el setpoint (mitad de la imagen).
        # Error positivo = linea a la derecha del centro → girar a la derecha.
        # Error negativo = linea a la izquierda del centro → girar a la izquierda.
        lane_center_x = compute_lane_center(lines)

        # ── Display principal: escala de grises + lineas Hough + velocimetro ──
        # Fondo: imagen de camara en escala de grises (paso 2 del modulo).
        # Lineas blancas: segmentos Hough que pasaron el filtro — entran al PID.
        display_frame = grey_image.copy()
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(display_frame, (x1, y1), (x2, y2), 255, 2)
        display_image(display_img, display_frame)

        # Velocimetro HUD: texto blanco sobre el display principal.
        estado = "SIN LINEA" if lane_center_x is None else f"E:{round(previous_error, 2)}"
        display_img.setColor(0xFFFFFF)
        display_img.drawText(f"V:{speed}km/h",     2, 2)
        display_img.drawText(f"St:{steering:.3f}r", 2, 12)
        display_img.drawText(estado,                2, 22)

        # ── Display ROI (pantalla secundaria) ──────────────────────────────────
        # Visualizacion en color:
        #   Azul  — zona excluida por el ROI (el trapecio no la considera)
        #   Blanco/gris — bordes Canny dentro del ROI (lo que ve HoughLinesP)
        # Permite entender visualmente que parte de la imagen contribuye a la
        # deteccion de lineas y que parte es ignorada por diseno.
        if display_roi:
            roi_viz = np.zeros((display_h, display_w, 3), dtype=np.uint8)
            # Zona fuera del ROI: azul (R=0, G=0, B=255 en formato RGB)
            roi_viz[roi_mask == 0] = [0, 0, 255]
            # Zona dentro del ROI: bordes Canny en escala de grises
            inside = roi_mask == 255
            roi_viz[inside, 0] = roi_edges[inside]
            roi_viz[inside, 1] = roi_edges[inside]
            roi_viz[inside, 2] = roi_edges[inside]
            display_image_rgb(display_roi, roi_viz)

        if lane_center_x is not None:
            no_line_frames = 0
            # Error normalizado: divide por el setpoint para que el rango sea [-1, 1].
            # Esto hace que las ganancias del PID sean independientes del ancho de imagen.
            # Error > 0: linea a la derecha del centro → girar derecha.
            # Error < 0: linea a la izquierda del centro → girar izquierda.
            error = (lane_center_x - setpoint) / setpoint

            # ── Paso 7: Controlador PID ───────────────────────────────────────
            # Termino proporcional: reaccion inmediata al error actual.
            # Termino integral: corrige deriva acumulada. Clamp ±0.5 evita
            # que el integral saturado actue como sesgo permanente.
            # Termino derivativo: amortigua oscilaciones.
            integral  += error * dt
            integral   = max(-0.5, min(0.5, integral))
            derivative = (error - previous_error) / dt

            raw_steering = Kp * error + Ki * integral + Kd * derivative
            raw_steering = max(-MAX_ANGLE, min(MAX_ANGLE, raw_steering))

            # Rate limiter: el angulo no puede cambiar mas de MAX_STEER_RATE
            # radianes por frame. Protege contra detecciones erroneas puntuales.
            steering = max(
                steering - MAX_STEER_RATE,
                min(steering + MAX_STEER_RATE, raw_steering)
            )
            previous_error = error

        else:
            # ── Sin linea detectada (interseccion o zona sin marcas) ──────────
            # Durante los primeros 10 frames sin linea se mantiene el angulo
            # exacto del frame anterior — el vehiculo "recuerda" la curva que traia.
            # A 50 km/h una cebra de 3 m dura ~7 frames: el hold es suficiente.
            # Despues de 10 frames se aplica decaimiento lento hacia recto (x0.95)
            # para no acumular deriva si la perdida de linea es prolongada.
            no_line_frames += 1
            integral      *= 0.6    # conserva parte del historial de deriva
            previous_error = 0.0    # evita kick de derivada al recuperar la linea
            if no_line_frames > 10:
                steering *= 0.95

        previous_time = current_time

        # ── Paso 8: Aplicar velocidad y angulo al vehiculo ────────────────────
        driver.setCruisingSpeed(speed)
        driver.setSteeringAngle(steering)

        # ── Teclado: captura de imagen ────────────────────────────────────────
        key = keyboard.getKey()
        if key != -1:
            if not (key in last_press and current_time - last_press[key] < DEBOUNCE_TIME):
                last_press[key] = current_time
                if key == ord('A'):
                    ts = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
                    camera.saveImage(os.getcwd() + "/" + ts + ".png", 1)
                    print(f"Imagen guardada: {ts}.png")


if __name__ == "__main__":
    main()
