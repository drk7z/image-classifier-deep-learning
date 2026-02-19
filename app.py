"""Gradio web application for image classification.

Run with: python app.py
"""

from functools import lru_cache
from pathlib import Path
from urllib.request import urlopen
import math
import os
import shutil
from typing import Any

import gradio as gr
import numpy as np
from PIL import Image, UnidentifiedImageError


MAX_UPLOAD_SIZE_MB = 10
MAX_UPLOAD_SIZE_BYTES = MAX_UPLOAD_SIZE_MB * 1024 * 1024
MAX_IMAGE_PIXELS = 20_000_000
MODEL_FILENAME = "transfer_learning_final_20260216_162455.h5"
MODEL_PATH = Path("models") / MODEL_FILENAME


def get_remote_model_url() -> str | None:
    return os.getenv("TRANSFER_MODEL_URL")


def resolve_model_path() -> tuple[str | None, str | None]:
    if MODEL_PATH.exists():
        return str(MODEL_PATH), None

    remote_url = get_remote_model_url()
    if not remote_url:
        return None, "Modelo local não encontrado e TRANSFER_MODEL_URL não está configurado."

    MODEL_PATH.parent.mkdir(exist_ok=True)

    try:
        with urlopen(remote_url, timeout=8) as response:
            with MODEL_PATH.open("wb") as output_file:
                shutil.copyfileobj(response, output_file)
    except Exception as exc:
        return None, f"Falha ao baixar modelo remoto: {exc}"

    if not MODEL_PATH.exists():
        return None, "Download do modelo terminou sem criar arquivo local."

    return str(MODEL_PATH), None


@lru_cache(maxsize=1)
def get_classifier() -> tuple[Any | None, str | None]:
    model_path, model_error = resolve_model_path()
    if model_error:
        return None, model_error

    try:
        from src.predict import ImageClassifier

        classifier = ImageClassifier(model_path=model_path, class_names=["Gato", "Cachorro"])
        return classifier, None
    except Exception as exc:
        return None, f"Falha ao inicializar classificador: {exc}"


def load_validated_image(file_path: str) -> tuple[Image.Image | None, str | None]:
    try:
        file_size = Path(file_path).stat().st_size
    except Exception as exc:
        return None, f"Falha ao ler arquivo: {exc}"

    if file_size > MAX_UPLOAD_SIZE_BYTES:
        return None, f"Arquivo excede {MAX_UPLOAD_SIZE_MB} MB."

    try:
        with Image.open(file_path) as image_obj:
            width, height = image_obj.size
            if width * height > MAX_IMAGE_PIXELS:
                return None, f"Imagem muito grande ({width}x{height})."
            image_rgb = image_obj.convert("RGB")
    except UnidentifiedImageError:
        return None, "Arquivo inválido: não é uma imagem reconhecida."
    except Exception as exc:
        return None, f"Falha ao abrir imagem: {exc}"

    return image_rgb, None


def predict_from_image(classifier: Any, image_rgb: Image.Image) -> tuple[str | None, float | None, np.ndarray | None, str | None]:
    prediction_error = None
    pred_class = None
    confidence_value = None
    scores_array = None

    try:
        pred_class, confidence, scores = classifier.predict_pil_image(image_rgb, return_confidence=True)
        try:
            confidence_value = float(confidence)
        except Exception:
            confidence_value = 0.0

        if not math.isfinite(confidence_value):
            confidence_value = 0.0

        scores_array = np.asarray(scores, dtype=float).flatten()
        scores_array = np.nan_to_num(scores_array, nan=0.0, posinf=1.0, neginf=0.0)
        if scores_array.size == 1:
            p1 = float(np.clip(scores_array[0], 0.0, 1.0))
            scores_array = np.array([1.0 - p1, p1], dtype=float)
    except Exception as exc:
        prediction_error = f"Falha na inferência: {exc}"

    if prediction_error:
        return None, None, None, prediction_error

    return pred_class, confidence_value, scores_array, None


def analyze_images(file_paths: list[str] | None):
    if not file_paths:
        return [], "Envie imagens para iniciar a análise."

    classifier, load_error = get_classifier()
    if load_error:
        return [], load_error

    rows = []
    for file_path in file_paths:
        file_name = Path(file_path).name
        image_rgb, validation_error = load_validated_image(file_path)
        if validation_error:
            rows.append([file_name, "-", "-", "-", "-", validation_error])
            continue

        pred_class, confidence_value, scores_array, prediction_error = predict_from_image(classifier, image_rgb)
        if prediction_error:
            rows.append([file_name, "-", "-", "-", "-", prediction_error])
            continue

        cat_score = dog_score = 0.0
        if scores_array is not None and scores_array.size >= 2:
            cat_score = float(np.clip(scores_array[0], 0.0, 1.0))
            dog_score = float(np.clip(scores_array[1], 0.0, 1.0))

        rows.append(
            [
                file_name,
                pred_class,
                f"{(confidence_value or 0.0):.2%}",
                f"{cat_score:.2%}",
                f"{dog_score:.2%}",
                "OK",
            ]
        )

    return rows, f"Processamento concluído para {len(rows)} imagem(ns)."


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Classificador de Imagens") as demo:
        gr.Markdown(
            """
            # 🐱🐶 Classificador de Imagens
            Classifica imagens de pets com Transfer Learning (MobileNetV2).
            Envie uma ou mais imagens para obter classe prevista e confiança.
            """
        )

        file_input = gr.Files(
            label="Escolha uma ou mais imagens",
            file_types=["image"],
            file_count="multiple",
            type="filepath",
        )
        analyze_button = gr.Button("Analisar imagens", variant="primary")
        results = gr.Dataframe(
            headers=["Arquivo", "Classe", "Confiança", "Gato", "Cachorro", "Status"],
            datatype=["str", "str", "str", "str", "str", "str"],
            row_count=5,
            col_count=(6, "fixed"),
            wrap=True,
            label="Resultados",
        )
        status = gr.Textbox(label="Status", interactive=False)

        analyze_button.click(fn=analyze_images, inputs=file_input, outputs=[results, status])

        gr.Markdown(
            """
            ### 🔗 Links
            - [LinkedIn](https://www.linkedin.com/in/leandroandradeti/)
            - [GitHub](https://github.com/drk7z)
            """
        )

    return demo


demo = build_app()


if __name__ == "__main__":
    server_name = os.getenv("GRADIO_SERVER_NAME", "0.0.0.0")
    server_port = int(os.getenv("PORT", os.getenv("GRADIO_SERVER_PORT", "7860")))
    demo.launch(server_name=server_name, server_port=server_port, show_error=True)
