#!/usr/bin/env python3
"""
Локальный офлайн транскрибатор аудио с диаризацией спикеров.
Использует Faster-Whisper для транскрибации и NVIDIA NeMo для диаризации.
Не требует регистрации или токенов.
"""

import os
import gc
import json
import tempfile
import traceback
import click
import torch
from pathlib import Path
from datetime import datetime
from faster_whisper import WhisperModel
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

console = Console()


def get_device():
    """Определяет доступное устройство для вычислений."""
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def format_timestamp(seconds: float) -> str:
    """Форматирует время в читаемый формат."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{ms:03d}"


def create_nemo_manifest(audio_path: str, manifest_path: str):
    """Создаёт manifest файл для NeMo."""
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
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False)
        f.write('\n')


def run_nemo_diarization(audio_path: str, output_dir: str, device: str):
    """Запускает диаризацию с помощью NeMo."""
    from nemo.collections.asr.models import ClusteringDiarizer
    from omegaconf import OmegaConf

    # Создаём manifest
    manifest_path = os.path.join(output_dir, "manifest.json")
    create_nemo_manifest(audio_path, manifest_path)

    # Конфигурация NeMo диаризатора
    config = OmegaConf.create({
        "device": device,
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

    # Запуск диаризации
    sd_model = ClusteringDiarizer(cfg=config)
    sd_model.diarize()

    # Чтение результатов RTTM
    rttm_file = os.path.join(output_dir, "pred_rttms",
                             Path(audio_path).stem + ".rttm")

    diarization_results = []
    if os.path.exists(rttm_file):
        with open(rttm_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 8:
                    start = float(parts[3])
                    duration = float(parts[4])
                    speaker = parts[7]
                    diarization_results.append({
                        "start": start,
                        "end": start + duration,
                        "speaker": speaker
                    })

    return diarization_results


def assign_speakers_to_segments(transcription_segments, diarization_results):
    """Назначает спикеров сегментам транскрипции."""
    for segment in transcription_segments:
        seg_start = segment["start"]
        seg_end = segment["end"]
        seg_mid = (seg_start + seg_end) / 2

        # Находим спикера для середины сегмента
        speaker = "SPEAKER_00"
        for diar in diarization_results:
            if diar["start"] <= seg_mid <= diar["end"]:
                speaker = diar["speaker"]
                break

        segment["speaker"] = speaker

    return transcription_segments


def save_results(result: dict, output_path: Path, format_type: str):
    """Сохраняет результаты в указанном формате."""

    if format_type == "json":
        with open(output_path.with_suffix('.json'), 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2, default=str)
        return output_path.with_suffix('.json')

    elif format_type == "txt":
        with open(output_path.with_suffix('.txt'), 'w', encoding='utf-8') as f:
            for segment in result.get("segments", []):
                speaker = segment.get("speaker", "Unknown")
                start = format_timestamp(segment["start"])
                end = format_timestamp(segment["end"])
                text = segment["text"].strip()
                f.write(f"[{start} - {end}] {speaker}: {text}\n")
        return output_path.with_suffix('.txt')

    elif format_type == "srt":
        with open(output_path.with_suffix('.srt'), 'w', encoding='utf-8') as f:
            for i, segment in enumerate(result.get("segments", []), 1):
                start = format_timestamp(segment["start"]).replace('.', ',')
                end = format_timestamp(segment["end"]).replace('.', ',')
                speaker = segment.get("speaker", "")
                text = segment["text"].strip()

                if speaker:
                    text = f"[{speaker}] {text}"

                f.write(f"{i}\n{start} --> {end}\n{text}\n\n")
        return output_path.with_suffix('.srt')

    else:
        raise ValueError(f"Неизвестный формат: {format_type}")


@click.command()
@click.argument('audio_file', type=click.Path(exists=True))
@click.option('--language', '-l', type=click.Choice(['ru', 'en']), default='ru',
              help='Язык аудио (ru - русский, en - английский)')
@click.option('--model', '-m', default='small',
              type=click.Choice(['tiny', 'base', 'small', 'medium', 'large-v2', 'large-v3']),
              help='Размер модели Whisper (small/medium для 6GB VRAM)')
@click.option('--diarize/--no-diarize', default=True,
              help='Включить/выключить диаризацию спикеров')
@click.option('--output', '-o', type=click.Path(), default=None,
              help='Путь для сохранения результата')
@click.option('--format', '-f', 'format_type',
              type=click.Choice(['txt', 'json', 'srt']), default='txt',
              help='Формат вывода')
@click.option('--beam-size', '-b', default=5, type=int,
              help='Beam size для Whisper (меньше = быстрее)')
@click.option('--compute-type', '-c', default='float16',
              type=click.Choice(['int8', 'float16', 'float32']),
              help='Тип вычислений (float16 для GPU)')
def transcribe(audio_file, language, model, diarize, output,
               format_type, beam_size, compute_type):
    """
    Транскрибирует аудиофайл с диаризацией спикеров.

    Использует Faster-Whisper + NVIDIA NeMo.
    Не требует регистрации или токенов.

    Примеры:

        python transcriber.py audio.mp3 -l ru --diarize

        python transcriber.py meeting.wav -l en -m medium -f srt
    """

    audio_path = Path(audio_file).resolve()
    device = get_device()

    # Информация о настройках
    console.print("\n[bold blue]═══ Транскрибатор (Whisper + NeMo) ═══[/bold blue]\n")

    info_table = Table(show_header=False, box=None)
    info_table.add_column(style="cyan")
    info_table.add_column(style="white")
    info_table.add_row("Файл:", str(audio_path))
    info_table.add_row("Язык:", "Русский" if language == "ru" else "English")
    info_table.add_row("Модель:", model)
    info_table.add_row("Устройство:", device.upper())
    info_table.add_row("Диаризация:", "Да (NeMo)" if diarize else "Нет")
    info_table.add_row("Тип вычислений:", compute_type)
    console.print(info_table)
    console.print()

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:

            # 1. Загрузка модели Whisper
            task = progress.add_task("[cyan]Загрузка модели Whisper...", total=None)

            whisper_model = WhisperModel(
                model,
                device=device,
                compute_type=compute_type
            )
            progress.update(task, description="[green]✓ Whisper загружен")

            # 2. Транскрибация
            task = progress.add_task("[cyan]Транскрибация...", total=None)

            segments_generator, info = whisper_model.transcribe(
                str(audio_path),
                language=language,
                beam_size=beam_size,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500)
            )

            # Собираем сегменты
            transcription_segments = []
            for segment in segments_generator:
                transcription_segments.append({
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text
                })

            progress.update(task, description=f"[green]✓ Транскрибация завершена ({len(transcription_segments)} сегментов)")

            # Очистка памяти
            del whisper_model
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()

            # 3. Диаризация с NeMo
            if diarize:
                task = progress.add_task("[cyan]Диаризация спикеров (NeMo)...", total=None)

                with tempfile.TemporaryDirectory() as temp_dir:
                    diarization_results = run_nemo_diarization(
                        str(audio_path),
                        temp_dir,
                        device
                    )

                    # Назначаем спикеров сегментам
                    transcription_segments = assign_speakers_to_segments(
                        transcription_segments,
                        diarization_results
                    )

                num_speakers = len(set(s.get("speaker") for s in transcription_segments if s.get("speaker")))
                progress.update(task, description=f"[green]✓ Диаризация завершена ({num_speakers} спикеров)")

                gc.collect()
                if device == "cuda":
                    torch.cuda.empty_cache()

        # Формируем результат
        result = {
            "segments": transcription_segments,
            "language": language,
            "model": model
        }

        # Сохранение результатов
        if output:
            output_path = Path(output)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = audio_path.parent / f"{audio_path.stem}_transcript_{timestamp}"

        saved_path = save_results(result, output_path, format_type)

        console.print(f"\n[bold green]✓ Результат сохранён:[/bold green] {saved_path}")

        # Статистика
        if transcription_segments:
            total_duration = transcription_segments[-1]["end"]
            speakers = set(s.get("speaker") for s in transcription_segments if s.get("speaker"))

            stats_table = Table(title="Статистика", show_header=False, box=None)
            stats_table.add_column(style="cyan")
            stats_table.add_column(style="white")
            stats_table.add_row("Сегментов:", str(len(transcription_segments)))
            stats_table.add_row("Длительность:", format_timestamp(total_duration))
            if diarize:
                stats_table.add_row("Спикеров:", str(len(speakers)))
            console.print(stats_table)

        console.print("\n[bold blue]═══ Готово! ═══[/bold blue]\n")

    except Exception as e:
        console.print(f"\n[bold red]Ошибка:[/bold red] {str(e)}")
        traceback.print_exc()
        raise click.Abort() from e


if __name__ == "__main__":
    transcribe()
