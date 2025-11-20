#!/usr/bin/env python3
"""
Gradio веб-интерфейс для транскрибатора.
Запуск: python ui.py
"""

import os
import gc
import json
import tempfile
import atexit
import torch
import gradio as gr
import numpy as np
import soundfile as sf
from datetime import datetime
from faster_whisper import WhisperModel

from nemo_utils import (
    get_device,
    create_nemo_manifest,
    run_nemo_diarization,
    assign_speakers_to_segments,
    NEMO_AVAILABLE
)

# Для предзагрузки моделей
if NEMO_AVAILABLE:
    from omegaconf import OmegaConf
    from nemo.collections.asr.models import ClusteringDiarizer

# Список временных файлов для очистки
_temp_files = []

def _cleanup_temp_files():
    """Очищает временные файлы при завершении программы."""
    for f in _temp_files:
        try:
            if os.path.exists(f):
                os.unlink(f)
        except Exception:
            pass

atexit.register(_cleanup_temp_files)


def preload_nemo_models():
    """Предзагружает модели NeMo при запуске приложения."""
    if not NEMO_AVAILABLE:
        return
    
    print("[INFO] Предзагрузка моделей NeMo...")
    try:
        device = get_device()
        
        # Создаем временную конфигурацию для предзагрузки
        with tempfile.TemporaryDirectory() as temp_dir:
            # Создаем минимальный тестовый аудиофайл (1 секунда тишины)
            temp_audio = os.path.join(temp_dir, "temp_audio.wav")
            # Создаем 1 секунду тишины (16kHz, моно)
            silence = np.zeros(16000, dtype=np.float32)
            sf.write(temp_audio, silence, 16000)
            
            temp_manifest = os.path.join(temp_dir, "temp_manifest.json")
            create_nemo_manifest(temp_audio, temp_manifest)
            
            config = OmegaConf.create({
                "device": device,
                "diarizer": {
                    "manifest_filepath": temp_manifest,
                    "out_dir": temp_dir,
                    "vad": {
                        "model_path": "vad_multilingual_marblenet"
                    },
                    "speaker_embeddings": {
                        "model_path": "titanet_large"
                    }
                }
            })
            
            # Создаем диаризатор - это заставит NeMo скачать модели
            print("[INFO] Загрузка VAD модели (vad_multilingual_marblenet)...")
            print("[INFO] Загрузка Speaker Embeddings модели (titanet_large)...")
            print("[INFO] Это может занять несколько минут при первом запуске...")
            sd_model = ClusteringDiarizer(cfg=config)
            # Устанавливаем атрибут verbose, если его нет (для совместимости с разными версиями NeMo)
            if not hasattr(sd_model, 'verbose'):
                sd_model.verbose = False
            # Не вызываем diarize(), просто инициализация загрузит модели
            print("[SUCCESS] Модели NeMo успешно загружены в кэш")
            
    except Exception as e:
        print(f"⚠ Предупреждение: не удалось предзагрузить модели NeMo: {e}")
        print("  Модели будут загружены при первом использовании диаризации")


def format_timestamp(seconds: float) -> str:
    """Форматирует время."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{ms:03d}"


def format_output(segments, format_type, diarize):
    """Форматирует вывод."""
    if format_type == "txt":
        lines = []
        for seg in segments:
            start = format_timestamp(seg["start"])
            end = format_timestamp(seg["end"])
            text = seg["text"].strip()
            if diarize:
                speaker = seg.get("speaker", "Unknown")
                lines.append(f"[{start} - {end}] {speaker}: {text}")
            else:
                lines.append(f"[{start} - {end}] {text}")
        return "\n".join(lines)

    elif format_type == "srt":
        lines = []
        for i, seg in enumerate(segments, 1):
            start = format_timestamp(seg["start"]).replace('.', ',')
            end = format_timestamp(seg["end"]).replace('.', ',')
            text = seg["text"].strip()
            if diarize:
                speaker = seg.get("speaker", "")
                text = f"[{speaker}] {text}"
            lines.append(f"{i}\n{start} --> {end}\n{text}\n")
        return "\n".join(lines)

    elif format_type == "json":
        return json.dumps({"segments": segments}, ensure_ascii=False, indent=2)

    else:
        raise ValueError(f"Неизвестный формат: {format_type}")


def transcribe(audio_file, language, model_size, diarize, output_format, progress=gr.Progress()):
    """Основная функция транскрибации."""

    if audio_file is None:
        return "Ошибка: загрузите аудиофайл", None, ""

    device = get_device()
    compute_type = "float16" if device == "cuda" else "int8"

    try:
        # 1. Загрузка модели Whisper
        progress(0.1, desc="Загрузка модели Whisper...")

        whisper_model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type
        )

        # 2. Транскрибация
        progress(0.3, desc="Транскрибация...")

        segments_gen, info = whisper_model.transcribe(
            audio_file,
            language=language,
            beam_size=5,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500)
        )

        segments = []
        for seg in segments_gen:
            segments.append({
                "start": seg.start,
                "end": seg.end,
                "text": seg.text
            })

        # Очистка
        del whisper_model
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

        # 3. Диаризация
        if diarize and NEMO_AVAILABLE:
            progress(0.6, desc="Диаризация спикеров...")

            try:
                with tempfile.TemporaryDirectory() as temp_dir:
                    diarization = run_nemo_diarization(audio_file, temp_dir, device)
                    segments = assign_speakers_to_segments(segments, diarization)
            except Exception as e:
                error_msg = str(e)
                if "download" in error_msg.lower() or "url" in error_msg.lower():
                    return (
                        f"Ошибка загрузки моделей NeMo: {error_msg}\n\n"
                        "Возможные решения:\n"
                        "1. Проверьте интернет-соединение\n"
                        "2. Попробуйте позже (серверы NeMo могут быть временно недоступны)\n"
                        "3. Отключите диаризацию спикеров и попробуйте без неё",
                        None,
                        ""
                    )
                else:
                    return f"Ошибка диаризации: {error_msg}", None, ""

            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()

        progress(0.9, desc="Форматирование результата...")

        # 4. Форматирование
        result_text = format_output(segments, output_format, diarize and NEMO_AVAILABLE)

        # 5. Создание файла для скачивания
        ext = {"txt": ".txt", "srt": ".srt", "json": ".json"}[output_format]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"transcript_{timestamp}{ext}"

        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix=ext, delete=False, encoding='utf-8')
        temp_file.write(result_text)
        temp_file.close()
        _temp_files.append(temp_file.name)  # Регистрируем для cleanup

        # Статистика
        num_segments = len(segments)
        duration = segments[-1]["end"] if segments else 0
        num_speakers = len(set(s.get("speaker") for s in segments if s.get("speaker"))) if diarize else 0

        stats = f"Сегментов: {num_segments} | Длительность: {format_timestamp(duration)}"
        if diarize and NEMO_AVAILABLE:
            stats += f" | Спикеров: {num_speakers}"

        progress(1.0, desc="Готово!")

        return result_text, temp_file.name, stats

    except Exception as e:
        return f"Ошибка: {str(e)}", None, ""


# Создание интерфейса
def create_ui():
    device = get_device()
    device_info = f"🖥 Устройство: {device.upper()}"
    if device == "cuda":
        try:
            device_info += f" ({torch.cuda.get_device_name(0)})"
        except Exception:
            device_info += " (Unknown GPU)"

    with gr.Blocks(title="Транскрибатор", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🎙 Транскрибатор аудио")
        gr.Markdown("Whisper + NeMo | Без регистрации и токенов")
        gr.Markdown(f"**{device_info}**")

        with gr.Row():
            with gr.Column(scale=1):
                audio_input = gr.Audio(
                    label="Аудиофайл",
                    type="filepath",
                    sources=["upload", "microphone"]
                )

                language = gr.Dropdown(
                    choices=[("Русский", "ru"), ("English", "en")],
                    value="ru",
                    label="Язык"
                )

                model_size = gr.Dropdown(
                    choices=["tiny", "base", "small", "medium", "large-v2", "large-v3"],
                    value="small",
                    label="Модель Whisper",
                    info="small рекомендуется для 6GB VRAM"
                )

                diarize = gr.Checkbox(
                    value=NEMO_AVAILABLE,
                    label="Диаризация спикеров",
                    interactive=NEMO_AVAILABLE,
                    info="Определение кто говорит" if NEMO_AVAILABLE else "NeMo не установлен"
                )

                output_format = gr.Radio(
                    choices=[("Текст", "txt"), ("Субтитры SRT", "srt"), ("JSON", "json")],
                    value="txt",
                    label="Формат вывода"
                )

                transcribe_btn = gr.Button("🚀 Транскрибировать", variant="primary", size="lg")

            with gr.Column(scale=2):
                output_text = gr.Textbox(
                    label="Результат",
                    lines=20,
                    max_lines=30,
                    show_copy_button=True
                )

                with gr.Row():
                    stats_text = gr.Textbox(label="Статистика", lines=1, interactive=False)
                    download_file = gr.File(label="Скачать файл")

        # Обработка
        transcribe_btn.click(
            fn=transcribe,
            inputs=[audio_input, language, model_size, diarize, output_format],
            outputs=[output_text, download_file, stats_text]
        )

        # Примеры
        gr.Markdown("### Рекомендации")
        gr.Markdown("""
        - **Для RTX 4050 (6GB):** используйте модель `small` или `medium`
        - **Время обработки:** ~10-15 минут на 1 час аудио
        - **Форматы:** MP3, WAV, M4A, FLAC и другие
        """)

    return demo


if __name__ == "__main__":
    # Предзагружаем модели NeMo в фоновом потоке
    if NEMO_AVAILABLE:
        import threading

        def preload_nemo_models_wrapper():
            """Обертка с обработкой ошибок для предзагрузки моделей."""
            try:
                preload_nemo_models()
            except Exception as e:
                print(f"[ERROR] Критическая ошибка при предзагрузке моделей NeMo: {e}")
                print("  Диаризация может быть недоступна. Модели будут загружены при первом использовании.")

        preload_thread = threading.Thread(target=preload_nemo_models_wrapper, daemon=True)
        preload_thread.start()
        print("[INFO] Предзагрузка моделей NeMo запущена в фоновом режиме...")

    demo = create_ui()
    demo.launch(
        server_name="127.0.0.1",  # Только локальный доступ для безопасности
        server_port=7860,
        share=False,
        inbrowser=True
    )
