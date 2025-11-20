#!/usr/bin/env python3
"""
Локальный офлайн транскрибатор аудио на базе WhisperX
с диаризацией спикеров для длинных аудиофайлов.
"""

import os
import gc
import json
import click
import whisperx
import torch
from pathlib import Path
from datetime import datetime
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


@click.command()
@click.argument('audio_file', type=click.Path(exists=True))
@click.option('--language', '-l', type=click.Choice(['ru', 'en']), default='ru',
              help='Язык аудио (ru - русский, en - английский)')
@click.option('--model', '-m', default='small',
              type=click.Choice(['tiny', 'base', 'small', 'medium', 'large-v2', 'large-v3']),
              help='Размер модели Whisper (small рекомендуется для слабых ПК)')
@click.option('--diarize/--no-diarize', default=True,
              help='Включить/выключить диаризацию спикеров')
@click.option('--hf-token', envvar='HF_TOKEN', default=None,
              help='HuggingFace токен для диаризации (или переменная HF_TOKEN)')
@click.option('--output', '-o', type=click.Path(), default=None,
              help='Путь для сохранения результата')
@click.option('--format', '-f', 'format_type',
              type=click.Choice(['txt', 'json', 'srt']), default='txt',
              help='Формат вывода')
@click.option('--batch-size', '-b', default=8, type=int,
              help='Размер батча (уменьшите при нехватке памяти)')
@click.option('--compute-type', '-c', default='int8',
              type=click.Choice(['int8', 'float16', 'float32']),
              help='Тип вычислений (int8 для экономии памяти)')
def transcribe(audio_file, language, model, diarize, hf_token, output,
               format_type, batch_size, compute_type):
    """
    Транскрибирует аудиофайл с диаризацией спикеров.

    Пример использования:

        python transcriber.py audio.mp3 -l ru --diarize

        python transcriber.py meeting.wav -l en -m base -f srt
    """

    audio_path = Path(audio_file)
    device = get_device()

    # Информация о настройках
    console.print("\n[bold blue]═══ WhisperX Транскрибатор ═══[/bold blue]\n")

    info_table = Table(show_header=False, box=None)
    info_table.add_column(style="cyan")
    info_table.add_column(style="white")
    info_table.add_row("Файл:", str(audio_path))
    info_table.add_row("Язык:", "Русский" if language == "ru" else "English")
    info_table.add_row("Модель:", model)
    info_table.add_row("Устройство:", device.upper())
    info_table.add_row("Диаризация:", "Да" if diarize else "Нет")
    info_table.add_row("Тип вычислений:", compute_type)
    console.print(info_table)
    console.print()

    # Проверка токена для диаризации
    if diarize and not hf_token:
        console.print("[yellow]⚠ Для диаризации нужен HuggingFace токен.[/yellow]")
        console.print("[yellow]  Установите переменную HF_TOKEN или используйте --hf-token[/yellow]")
        console.print("[yellow]  Диаризация будет отключена.[/yellow]\n")
        diarize = False

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:

            # 1. Загрузка модели
            task = progress.add_task("[cyan]Загрузка модели...", total=None)

            whisper_model = whisperx.load_model(
                model,
                device,
                compute_type=compute_type,
                language=language
            )
            progress.update(task, description="[green]✓ Модель загружена")

            # 2. Загрузка аудио
            task = progress.add_task("[cyan]Загрузка аудио...", total=None)
            audio = whisperx.load_audio(str(audio_path))
            progress.update(task, description="[green]✓ Аудио загружено")

            # 3. Транскрибация
            task = progress.add_task("[cyan]Транскрибация...", total=None)
            result = whisper_model.transcribe(audio, batch_size=batch_size)
            progress.update(task, description="[green]✓ Транскрибация завершена")

            # Очистка памяти
            del whisper_model
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()

            # 4. Выравнивание
            task = progress.add_task("[cyan]Выравнивание...", total=None)
            model_a, metadata = whisperx.load_align_model(
                language_code=language,
                device=device
            )
            result = whisperx.align(
                result["segments"],
                model_a,
                metadata,
                audio,
                device,
                return_char_alignments=False
            )
            progress.update(task, description="[green]✓ Выравнивание завершено")

            # Очистка памяти
            del model_a
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()

            # 5. Диаризация
            if diarize:
                task = progress.add_task("[cyan]Диаризация спикеров...", total=None)
                diarize_model = whisperx.DiarizationPipeline(
                    use_auth_token=hf_token,
                    device=device
                )
                diarize_segments = diarize_model(audio)
                result = whisperx.assign_word_speakers(diarize_segments, result)
                progress.update(task, description="[green]✓ Диаризация завершена")

                del diarize_model
                gc.collect()
                if device == "cuda":
                    torch.cuda.empty_cache()

        # Сохранение результатов
        if output:
            output_path = Path(output)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = audio_path.parent / f"{audio_path.stem}_transcript_{timestamp}"

        saved_path = save_results(result, output_path, format_type)

        console.print(f"\n[bold green]✓ Результат сохранён:[/bold green] {saved_path}")

        # Статистика
        segments = result.get("segments", [])
        if segments:
            total_duration = segments[-1]["end"]
            speakers = set(s.get("speaker", "Unknown") for s in segments)

            stats_table = Table(title="Статистика", show_header=False, box=None)
            stats_table.add_column(style="cyan")
            stats_table.add_column(style="white")
            stats_table.add_row("Сегментов:", str(len(segments)))
            stats_table.add_row("Длительность:", format_timestamp(total_duration))
            if diarize:
                stats_table.add_row("Спикеров:", str(len(speakers)))
            console.print(stats_table)

        console.print("\n[bold blue]═══ Готово! ═══[/bold blue]\n")

    except Exception as e:
        console.print(f"\n[bold red]Ошибка:[/bold red] {str(e)}")
        raise click.Abort()


if __name__ == "__main__":
    transcribe()
