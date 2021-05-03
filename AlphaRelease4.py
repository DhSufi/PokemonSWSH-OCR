from multiprocessing import Process, Queue
import asyncio
import device
import numpy as np
import cv2
import pytesseract
import simpleobsws

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

hola = asyncio.get_event_loop()
ws = simpleobsws.obsws(host='127.1.1.1', port=4444, password='hola', loop=hola)

# Definir colores
rojo_min = np.array([170, 190, 140])  # Promedio del rango mínimo de color rojo con 3 capturadoras diferentes
rojo_max = np.array([179, 230, 213])  # Promedio del rango maximo de color rojo con 3 capturadoras diferentes
rojohd_min = np.array([170, 215, 130])
rojohd_max = np.array([179, 255, 190])
rojo2_min = np.array([0, 160, 190])  # TEAM BOX Red color
rojo2_max = np.array([10, 190, 255])  # TEAM BOX Red color
rojo2hd_min = np.array([170, 185, 220])
rojo2hd_max = np.array([179, 255, 255])
rojo3_min = np.array([174, 160, 190])  # TEAM BOX Red color
rojo3_max = np.array([179, 190, 255])  # TEAM BOX Red color
azul_min = np.array([105, 188, 188])  # TEAM BOX Blue color
azul_max = np.array([110, 255, 255])  # TEAM BOX Blue color
blanco_min = np.array([0, 0, 210])  # Blanco generico
blanco_max = np.array([179, 85, 255])  # Blanco generico
negro_min = np.array([0, 0, 0])  # Negro generico
negro_max = np.array([179, 255, 50])  # Negro generico
texto_min = np.array([0, 0, 80])  # Color Texto
texto_max = np.array([179, 28, 255])  # Color Texto


# Funciones input usuario
def ask_for_capturecard(recuento):
    mensaje = "Select a camera (0 to " + str(recuento) + "): "
    try:
        valor = int(input(mensaje))
        # select = int(select)
    except Exception:
        print("It's not a number!")
        return ask_for_capturecard(recuento)

    if valor > recuento:
        print("Invalid number! Retry!")
        return ask_for_capturecard(recuento)

    return valor


def chosen_capturecard():
    # print OpenCV version
    print("OpenCV version: " + cv2.__version__)

    # Get camera list
    device_list = device.getDeviceList()
    etiqueta = 0

    for name in device_list:
        print(str(etiqueta) + ': ' + name)
        etiqueta += 1

    recuento = etiqueta - 1

    if recuento < 0:
        print("No device is connected")
        return

    # Select a camera
    NumeroCapturadora = ask_for_capturecard(recuento)

    return NumeroCapturadora


# Funcion obtener frame
def get_capture(cola, variable):
    print(variable)
    capture = cv2.VideoCapture(variable, cv2.CAP_DSHOW)
    capture.set(3, 1280)
    capture.set(4, 720)

    cola.put(capture)
    # return capture


# Funcion deteccion de puntos de control
def checkpoint(frame, value):
    contornos, hierarchy = cv2.findContours(frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contornos:
        area = cv2.contourArea(cnt)
        if area > value:
            # cv2.drawContours(frame, cnt, -1, (255, 0, 0), 3)
            return True
        else:
            return False

# Funcion ocultar
def overlay_frame(imagen, overlay, y0, y1, x0, x1):
    img = cv2.imread("Recursos/" + overlay + '.png', cv2.IMREAD_UNCHANGED)  # Custom image import with RGB-A (4 Channels)
    imgResized = cv2.resize(img, ((x1-x0), (y1-y0)))  # Resize "img" to fit the area to hide (RGB-A)
    maskAlpha = imgResized[:, :, 3]  # Generate alpha mask of "imgResized"
    maskAlphaInv = cv2.bitwise_not(maskAlpha)  # Generate the inverse of the "maskAlpha"
    imgRGB = imgResized[:, :, 0:3]  # Convert "imgResized" RGB-A to RGB (3 Channels)

    Recorte = imagen[y0:y1, x0:x1]  # Crop the frame to fit the area to hide
    CapaInferior = cv2.bitwise_and(Recorte, Recorte, mask=maskAlphaInv)  # Apply "maskAlphaInv" to "Recorte"
    CapaSuperior = cv2.bitwise_and(imgRGB, imgRGB, mask=maskAlpha)  # Apply "maskAlpha" to "imgRGB"
    dst = cv2.add(CapaInferior, CapaSuperior)  # Add "Capainferior" + "CapaSuperior"
    imagen[y0:y1, x0:x1]  = dst  # Draw "dst" in the frame area to hide

    return imagen


# Funcion imagen a string
def read_text(imagen, y0, y1, x0, x1):
    frame_area = imagen[y0:y1, x0:x1]
    text = pytesseract.image_to_string(frame_area,
                                       config="-c tessedit_char_whitelist=2ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz --psm 6")
    # print(text1)
    # print(len(current_pokemon1[0]))
    try:
        pokemon = text.splitlines()
    except:
        return 'notext'
    if len(pokemon[0]) >= 3:
        return pokemon[0]
    else:
        return 'notext'


# Funciones OBS
async def obsset_image(sourcename, pokemonname):
    try:
        await ws.connect()
        await ws.emit('SetSourceSettings', {'sourceName': sourcename, 'sourceSettings': {
            'file': 'C:/Users/Flamerino/PycharmProjects/BarrasPokemon/Pokes/' + pokemonname + '.png'}})
        await ws.disconnect()
    except:
        return


# Funcion HIDE INFO
def hide_info(cola, valor):
    capture = cv2.VideoCapture(valor, cv2.CAP_DSHOW)
    capture.set(3, 1280)
    capture.set(4, 720)

    alpha_value = 0
    contadorframes = 0

    while True:
        contadorframes = contadorframes + 1
        # Leer la capturadora
        success, frame = capture.read()
        clean = frame.copy()

        if contadorframes == 1:
            cola.put(frame)
        elif contadorframes == 2:
            contadorframes = 0

        # Transformación y manipulación del frame
        FrameBlur = cv2.GaussianBlur(frame, (7, 7), 1)  # Difuminar el Frame
        FrameHSV = cv2.cvtColor(FrameBlur, cv2.COLOR_BGR2HSV)  # Conversión de color del Frame a gama de color HSV

        # Crear mascara de color
        maskWhite = cv2.inRange(FrameHSV, blanco_min, blanco_max)  # White tone detection mask  - for ALL
        maskBlack = cv2.inRange(FrameHSV, negro_min, negro_max)  # Black tone detection mask  - for ALL
        maskRed = cv2.inRange(FrameHSV, rojohd_min, rojohd_max)  # Red tone 2 detection mask  - for CHANGE SCREEN
        maskRed2 = cv2.inRange(FrameHSV, rojo2hd_min, rojo2hd_max)  # Red tone detection mask    - for TEAM BOX
        maskRed3 = cv2.inRange(FrameHSV, rojo3_min, rojo3_max)  # Red tone detection mask    - for TEAM BOX
        maskBlue = cv2.inRange(FrameHSV, azul_min, azul_max)  # Blue tone detection mask   - for TEAM BOX

        # Definir puntos de deteccion
        P1 = maskRed[15:30, 1220:1240]  # CHANGE SCREEN Canvas coordinates for Red    Checkpoint 1
        P2 = maskRed[15:30, 1130:1150]  # CHANGE SCREEN Canvas coordinates for Red    Checkpoint 2
        P3 = maskWhite[535:570, 1200:1235]  # CHANGE SCREEN Canvas coordinates for white  Checkpoint 3
        P4 = maskWhite[535:570, 1240:1265]  # CHANGE SCREEN Canvas coordinates for White  Checkpoint 4

        P10 = maskBlack[485:490, 1180:1210]  # MOVE SELECT   Canvas coordinates for Black  Checkpoint 1
        P11 = maskBlack[555:560, 1180:1210]  # MOVE SELECT   Canvas coordinates for Black  Checkpoint 2
        P12 = maskBlack[625:630, 1180:1210]  # MOVE SELECT   Canvas coordinates for Black  Checkpoint 3
        P13 = maskBlack[695:700, 1180:1210]  # MOVE SELECT   Canvas coordinates for Black  Checkpoint 4
        P14 = maskWhite[365:375, 1260:1270]  # MOVE SELECT   Canvas coordinates for White  Checkpoint 5

        P20 = maskBlack[323:330, 466:475]  # TARGET SLOT   Canvas coordinates for Black  Checkpoint 1
        P21 = maskBlack[323:330, 770:780]  # TARGET SLOT   Canvas coordinates for Black  Checkpoint 2
        P22 = maskBlack[468:475, 466:475]  # TARGET SLOT   Canvas coordinates for Black  Checkpoint 3
        P23 = maskBlack[468:475, 770:780]  # TARGET SLOT   Canvas coordinates for Black  Checkpoint 4
        P24 = maskWhite[65:70, 1250:1260]  # TARGET SLOT   Canvas coordinates for White  Checkpoint 5
        P25 = maskWhite[65:70, 910:920]  # TARGET SLOT   Canvas coordinates for White  Checkpoint 6

        P30 = maskBlue[100:110, 215:230]  # MY TEAM BOX   Canvas coordinates for Blue   Checkpoint 1
        P31 = maskBlue[100:110, 450:465]  # MY TEAM BOX   Canvas coordinates for Blue   Checkpoint 2
        P32 = maskWhite[375:385, 420:430]  # MY TEAM BOX   Canvas coordinates for White  Checkpoint 3

        P40 = maskRed2[140:150, 80:90]  # PICKING PKM   Canvas coordinates for Red1   Checkpoint 1
        P41 = maskRed2[140:150, 320:330]  # PICKING PKM   Canvas coordinates for Red1   Checkpoint 2
        P42 = maskRed3[140:150, 80:90]  # PICKING PKM   Canvas coordinates for Red2   Checkpoint 1
        P43 = maskRed3[140:150, 320:330]  # PICKING PKM   Canvas coordinates for Red2   Checkpoint 2

        P50 = maskWhite[75:85, 10:20]  # INFO SCREEN   Canvas coordinates for White  Checkpoint 1
        P51 = maskWhite[175:185, 10:20]  # INFO SCREEN   Canvas coordinates for White  Checkpoint 2
        P52 = maskBlack[640:650, 10:20]  # INFO SCREEN   Canvas coordinates for Black  Checkpoint 3
        P53 = maskBlack[640:650, 195:205]  # INFO SCREEN   Canvas coordinates for Black  Checkpoint 4
        P54 = maskBlack[700:710, 635:645]  # INFO SCREEN   Canvas coordinates for Black  Checkpoint 5

        P99 = maskBlack[60:660, 340:940]  # BLACK SCREEN BETWEEN TRAINER CARD AND GAME

        # HIDE CHANGE POKEMON SCREEN
        if checkpoint(P1, 30) and checkpoint(P2, 30) and checkpoint(P3, 30) and checkpoint(P4, 30) is True:
            overlay_frame(frame, 'change', 45, 600, 0, 1280)

        # HIDE MOVE SELECT
        if checkpoint(P10, 30) and checkpoint(P11, 30) and checkpoint(P12, 30) and checkpoint(P13, 30) and checkpoint(
                P14, 30) is True:
            overlay_frame(frame, 'moves', 410, 720, 650, 1280)

        # HIDE TARGET SLOT
        if checkpoint(P20, 30) and checkpoint(P21, 30) and checkpoint(P22, 30) and checkpoint(P23, 30) and (checkpoint(P24, 30) or checkpoint(P25, 30)) is True:
            overlay_frame(frame, 'target', 225, 500, 305, 935)

        # HIDE MY TEAM BOX
        if checkpoint(P30, 30) and checkpoint(P31, 30) and checkpoint(P32, 30) is True:
            overlay_frame(frame, 'team', 123, 604, 410, 480)

        # HIDE POKEMON PICKING
        if (checkpoint(P40, 30) and checkpoint(P41, 30)) or (checkpoint(P42, 30) and checkpoint(P43, 30)) is True:
            overlay_frame(frame, 'picking', 20, 600, 380, 1120)

        # HIDE INFO BATTLE STATUS (STATS, FIELD, WEATHER)
        if checkpoint(P50, 30) and checkpoint(P51, 30) and checkpoint(P52, 30) and checkpoint(P53, 30) and checkpoint(
                P54, 30) is True:
            overlay_frame(frame, 'info', 45, 600, 0, 1280)

        # FADE LOGO IN ANY BLACK SCREEN
        if checkpoint(P99, 90000) is True:
            img = cv2.imread("Recursos/logo.png", cv2.IMREAD_UNCHANGED)  # Custom image import with RGB-A (4 Channels)
            imgResized = cv2.resize(img, (1280, 720))  # Resize "img" to fit the area to hide (RGB-A)
            maskAlpha = imgResized[:, :, 3]  # Generate alpha mask of "imgResized"
            maskAlphaInv = cv2.bitwise_not(maskAlpha)  # Generate the inverse of the "maskAlpha"
            imgRGB = imgResized[:, :, 0:3]  # Convert "imgResized" RGB-A to RGB (3 Channels)

            Recorte = frame[0:720, 0:1280]  # Crop the frame to fit the area to hide
            CapaInferior = cv2.bitwise_and(Recorte, Recorte, mask=maskAlphaInv)  # Apply "maskAlphaInv" to "Recorte"
            CapaSuperior = cv2.bitwise_and(imgRGB, imgRGB, mask=maskAlpha)  # Apply "maskAlpha" to "imgRGB"
            dst = cv2.addWeighted(CapaInferior, 1, CapaSuperior, alpha_value, 0)
            frame[0:720, 0:1280] = dst  # Draw "dst" in the frame area to hide
            if alpha_value + 0.033 > 1:
                alpha_value = 1
            else:
                alpha_value = alpha_value + 0.033
        else:
            alpha_value = 0

        # invert = (255 - frame)
        # invertHSV = cv2.cvtColor(invert, cv2.COLOR_BGR2HSV)
        # maskText = cv2.inRange(invertHSV, texto_min, texto_max)

        cv2.imshow("frame", frame)   # SHOW HIDE VGC INFO window
        cv2.imshow("clean", clean)   # SHOW CLEANFEED window

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cola.put('finalizar')
            capture.release()
            cv2.destroyAllWindows()
            break


def main_ocr(cola):
    current_pokemon1 = ''
    previous_pokemon1 = ''
    current_pokemon2 = ''
    previous_pokemon2 = ''
    p1a_counter = 0
    p1b_counter = 0
    p2a_counter = 0
    p2b_counter = 0

    while True:
        frame = cola.get()

        if type(frame) == str:
            break

        invert = (255 - frame)
        invertHSV = cv2.cvtColor(invert, cv2.COLOR_BGR2HSV)
        maskText = cv2.inRange(invertHSV, texto_min, texto_max)

        FrameHSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        maskWhite = cv2.inRange(FrameHSV, blanco_min, blanco_max)

        Pcheck111 = maskWhite[665:675, 170:270]  # Check: Pokemon 1, Position 1, Point 1
        Pcheck112 = maskWhite[620:630, 290:310]  # Check: Pokemon 1, Position 1, Point 2
        Pcheck121 = maskWhite[675:685, 170:270]  # Check: Pokemon 1, Posiiton 2, Point 1
        Pcheck122 = maskWhite[630:640, 290:310]  # Check: Pokemon 1, Position 2, Point 2

        Pcheck211 = maskWhite[670:680, 500:600]  # Check: Pokemon 2, Position 1, Point 1
        Pcheck212 = maskWhite[620:630, 620:640]  # Check: Pokemon 2, Position 1, Point 2
        Pcheck221 = maskWhite[680:690, 500:600]  # Check: Pokemon 2, Position 2, Point 1
        Pcheck222 = maskWhite[630:640, 620:640]  # Check: Pokemon 2, Position 2, Point 2

        # MyPokemon1
        if checkpoint(Pcheck111, 800) and checkpoint(Pcheck112, 30) is True:
            p1a_counter = p1a_counter + 1
            if p1a_counter >= 10:
                current_pokemon1 = read_text(maskText, 600, 640, 0, 170)
                p1a_counter = 0
                if current_pokemon1 != previous_pokemon1:
                    if current_pokemon1 == 'notext':
                        previous_pokemon1 = current_pokemon1
                    else:
                        hola.run_until_complete(obsset_image('MyPokemon1', current_pokemon1))
                        previous_pokemon1 = current_pokemon1
                        print('MyPokemon1: '+ str(current_pokemon1))
        elif checkpoint(Pcheck121, 800) and checkpoint(Pcheck122, 30) is True:
            p1b_counter = p1b_counter + 1
            if p1b_counter >= 10:
                current_pokemon1 = read_text(maskText, 610, 650, 0, 170)
                p1b_counter = 0
                if current_pokemon1 != previous_pokemon1:
                    if current_pokemon1 == 'notext':
                        previous_pokemon1 = current_pokemon1
                    else:
                        hola.run_until_complete(obsset_image('MyPokemon1', current_pokemon1))
                        previous_pokemon1 = current_pokemon1
                        print('MyPokemon1: ' + str(current_pokemon1))
        # MyPokemon2
        if checkpoint(Pcheck211, 800) and checkpoint(Pcheck212, 30) is True:
            p2a_counter = p2a_counter + 1
            if p2a_counter >= 10:
                current_pokemon2 = read_text(maskText, 605, 640, 345, 495)
                p2a_counter = 0
                if current_pokemon2 != previous_pokemon2:
                    if current_pokemon2 == 'notext':
                        previous_pokemon2 = current_pokemon2
                    else:
                        hola.run_until_complete(obsset_image('MyPokemon2', current_pokemon2))
                        previous_pokemon2 = current_pokemon2
                        print('MyPokemon2: ' + str(current_pokemon2))
        elif checkpoint(Pcheck221, 800) and checkpoint(Pcheck222, 30) is True:
            p2b_counter = p2b_counter + 1
            if p2b_counter >= 10:
                current_pokemon2 = read_text(maskText, 615, 650, 345, 495)
                p2b_counter = 0
                if current_pokemon2 != previous_pokemon2:
                    if current_pokemon2 == 'notext':
                        previous_pokemon2 = current_pokemon2
                    else:
                        hola.run_until_complete(obsset_image('MyPokemon2', current_pokemon2))
                        previous_pokemon2 = current_pokemon2
                        print('MyPokemon2: ' + str(current_pokemon2))

#EJECUCION
if __name__ == '__main__':
    multiprocessing.freeze_support()

    capturadora = chosen_capturecard()

    colita = Queue()

    proceso_hide = Process(target=hide_info, args=(colita, capturadora,))
    proceso_hide.start()

    while colita.qsize() == 0:
        pass
    else:
        proceso_ocr = Process(target=main_ocr, args=(colita,))
        proceso_ocr.start()

        proceso_hide.join()
        proceso_ocr.join()
