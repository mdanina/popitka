#!/usr/bin/env python3
"""
Gradio веб-интерфейс для транскрибатора.
Запуск: python ui.py
"""

import os
import gc
import json
import tempfile
import torch
import gradio as gr
from pathlib import Path
from datetime import datetime
from faster_whisper import WhisperModel

# Проверяем доступность NeMo
try:
    from nemo.collections.asr.models import ClusteringDiarizer
    from omegaconf import OmegaConf
    NEMO_AVAILABLE = True
except ImportError:
    NEMO_AVAILABLE = False
    print("⚠ NeMo не установлен. Диаризация будет недоступна.")


def get_device():
    """Определяет доступное устройство."""
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def format_timestamp(seconds: float) -> str:
    """Форматирует время."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{ms:03d}"


def create_nemo_manifest(audio_path: str, manifest_path: str):
    """Создаёт manifest для NeMo."""
    meta = {
        "audio_filepath": audio_path,
        "offset": 0,
        "duration": None,
        "label": "infer",
        "text": "-",
        "num_speakers": None,
        "rttm_filepath": None,
        "uem_filepath": None
    }
    with open(manifest_path, 'w') as f:
        json.dump(meta, f)
        f.write('\n')


def run_nemo_diarization(audio_path: str, output_dir: str):
    """Запускает диаризацию NeMo."""
    manifest_path = os.path.join(output_dir, "manifest.json")
    create_nemo_manifest(audio_path, manifest_path)

    config = OmegaConf.create({
        "diarizer": {
            "manifest_filepath": manifest_path,
            "out_dir": output_dir,
            "oracle_vad": False,
            "collar": 0.25,
            "ignore_overlap": True,
            "vad": {
                "model_path": "vad_multilingual_marblenet",
                "external_vad_manifest": None,
                "parameters": {
                    "window_length_in_sec": 0.15,
                    "shift_length_in_sec": 0.01,
                    "smoothing": "median",
                    "overlap": 0.5,
                    "onset": 0.1,
                    "offset": 0.1,
                    "pad_onset": 0.1,
                    "pad_offset": 0,
                    "min_duration_on": 0.2,
                    "min_duration_off": 0.2,
                    "filter_speech_first": True
                }
            },
            "speaker_embeddings": {
                "model_path": "titanet_large",
                "parameters": {
                    "window_length_in_sec": [1.5, 1.25, 1.0, 0.75, 0.5],
                    "shift_length_in_sec": [0.75, 0.625, 0.5, 0.375, 0.25],
                    "multiscale_weights": [1, 1, 1, 1, 1],
                    "save_embeddings": False
                }
            },
            "clustering": {
                "parameters": {
                    "oracle_num_speakers": False,
                    "max_num_speakers": 8,
                    "enhanced_count_thres": 80,
                    "max_rp_threshold": 0.25,
                    "sparse_search_volume": 30,
                    "maj_vote_spk_count": False
                }
            }
        }
    })

    sd_model = ClusteringDiarizer(cfg=config)
    sd_model.diarize()

    rttm_file = os.path.join(output_dir, "pred_rttms",
                             Path(audio_path).stem + ".rttm")

    results = []
    if os.path.exists(rttm_file):
        with open(rttm_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 8:
                    start = float(parts[3])
                    duration = float(parts[4])
                    speaker = parts[7]
                    results.append({
                        "start": start,
                        "end": start + duration,
                        "speaker": speaker
                    })

    return results


def assign_speakers(segments, diarization):
    """Назначает спикеров сегментам."""
    for segment in segments:
        seg_mid = (segment["start"] + segment["end"]) / 2
        speaker = "SPEAKER_00"
        for diar in diarization:
            if diar["start"] <= seg_mid <= diar["end"]:
                speaker = diar["speaker"]
                break
        segment["speaker"] = speaker
    return segments


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


def transcribe(audio_file, language, model_size, diarize, output_format, progress=gr.Progress()):
    """Основная функция транскрибации."""

    if audio_file is None:
        return "Ошибка: загрузите аудиофайл", None

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

            with tempfile.TemporaryDirectory() as temp_dir:
                diarization = run_nemo_diarization(audio_file, temp_dir)
                segments = assign_speakers(segments, diarization)

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

        # Статистика
        num_segments = len(segments)
        duration = segments[-1]["end"] if segments else 0
        num_speakers = len(set(s.get("speaker", "") for s in segments)) if diarize else 0

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
        device_info += f" ({torch.cuda.get_device_name(0)})"

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
    demo = create_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=True
    )
