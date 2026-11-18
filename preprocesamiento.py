import html
import re


def limpiar_texto(texto):
    """Normaliza un texto sin eliminar palabras de negación."""

    # Convierte entidades HTML: por ejemplo, &amp; pasa a ser &.
    texto = html.unescape(str(texto))

    # Unifica mayúsculas y apóstrofes.
    texto = texto.lower().replace("’", "'")

    # Conserva explícitamente la negación de contracciones inglesas.
    texto = re.sub(r"\bwon't\b", "will not", texto)
    texto = re.sub(r"\bcan't\b", "can not", texto)
    texto = re.sub(r"n't\b", " not", texto)

    # Elimina enlaces y menciones a cuentas.
    texto = re.sub(r"https?://\S+|www\.\S+", " ", texto)
    texto = re.sub(r"@\w+", " ", texto)

    # Quita el símbolo #, pero conserva la palabra del hashtag.
    texto = texto.replace("#", "")

    # Reduce espacios, tabulaciones y saltos de línea a un espacio.
    texto = re.sub(r"\s+", " ", texto).strip()

    # Conservamos puntuación y emoticonos en esta etapa.
    return texto