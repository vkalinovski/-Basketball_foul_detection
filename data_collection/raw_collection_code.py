# !pip install pytubefix -q
# !pip install yt_dlp ffmpeg-python torch whisper requests nest_asyncio -q
# !pip install yt-dlp ffmpeg-python pydub requests nest_asyncio -q

#-------------------------------------------------------------------------
import os
import sys
import json
import math
import time
import re
import ffmpeg
import requests
import yt_dlp
from pydub import AudioSegment
from IPython.display import clear_output
import nest_asyncio
from pathlib import Path
from typing import List, Dict

nest_asyncio.apply()
#-------------------------------------------------------------------------
from google.colab import drive
drive.mount('/content/drive', force_remount=True)
#-------------------------------------------------------------------------
# Конфигурация
class Config:
    DRIVE_PATH = "/content/drive/MyDrive/meow/DS_TWO/DATA/NBA_BEST_FIGHTS"
    TEMP_DIR = "/content/temp"
    FRAME_RATE = 18
    MAX_AUDIO_SIZE_MB = 5
    AUDIO_BITRATE = '128k'
    SAMPLE_RATE = 16000
    TRANSCRIBE_RETRIES = 2
    TRANSCRIBE_TIMEOUT = 120
    VIDEO_CODEC = 'libx264'
    AUDIO_CODEC = 'aac'
    PIX_FMT = 'yuv420p'
    VIDEO_EXT = 'mp4'
    DEEPINFRA_API_KEY = "9kScTQSpFJCd6OdxblR8bHjw3iWSPZkV"
    WHISPER_API_URL = "https://api.deepinfra.com/v1/inference/openai/whisper-large"
    LLM_API_URL = "https://api.deepinfra.com/v1/inference/meta-llama/Meta-Llama-3-70B-Instruct"
    COOKIES_PATH = "/content/drive/MyDrive/meow/cookies.txt"
    VIDEO_URL = "https://youtu.be/nBfcBmmt--M?si=a-LyW-XkSD4nouVj"

    @classmethod
    def setup(cls):
        os.makedirs(cls.DRIVE_PATH, exist_ok=True)
        os.makedirs(cls.TEMP_DIR, exist_ok=True)

Config.setup()
#-------------------------------------------------------------------------
def convert_to_mp3(input_path: str) -> str:
    output_path = Path(input_path).with_suffix(".mp3")
    (
        ffmpeg
        .input(input_path)
        .output(
            str(output_path),
            acodec='libmp3lame',
            audio_bitrate=Config.AUDIO_BITRATE,
            ar=str(Config.SAMPLE_RATE)
        )
        .overwrite_output()
        .run(quiet=True)
    )
    return str(output_path)
#-------------------------------------------------------------------------
def split_audio(input_path: str) -> List[str]:
    chunk_size = Config.MAX_AUDIO_SIZE_MB * 1024 * 1024
    file_size = os.path.getsize(input_path)

    if file_size <= chunk_size:
        return [input_path]

    audio = AudioSegment.from_file(input_path)
    duration_ms = len(audio)
    chunk_duration_ms = int((chunk_size / file_size) * duration_ms)

    chunks = []
    for i in range(0, math.ceil(duration_ms / chunk_duration_ms)):
        start = i * chunk_duration_ms
        end = min((i+1) * chunk_duration_ms, duration_ms)
        chunk = audio[start:end]
        chunk_path = Path(input_path).with_name(f"{Path(input_path).stem}_part{i}.mp3")
        chunk.export(chunk_path, format="mp3", bitrate=Config.AUDIO_BITRATE)
        chunks.append(str(chunk_path))
    return chunks
#-------------------------------------------------------------------------
def transcribe_chunk(chunk_path: str) -> str:
    headers = {"Authorization": f"Bearer {Config.DEEPINFRA_API_KEY}"}
    for attempt in range(Config.TRANSCRIBE_RETRIES):
        try:
            with open(chunk_path, "rb") as f:
                response = requests.post(
                    Config.WHISPER_API_URL,
                    headers=headers,
                    files={"audio": f},
                    timeout=Config.TRANSCRIBE_TIMEOUT)
            if response.status_code == 200:
                return response.json().get("text", "")
            else:
                print(f"Ошибка транскрибации: {response.status_code}")
                return ""
        except requests.exceptions.ReadTimeout:
            print(f"⏱ Таймаут. Повтор {attempt+1}/{Config.TRANSCRIBE_RETRIES}")
            if attempt < Config.TRANSCRIBE_RETRIES - 1:
                time.sleep(5)
                continue
            else:
                print("❌ Превышено количество попыток.")
                return ""
        except Exception as e:
            print(f"❌ Ошибка: {e}")
            return ""
    return ""
#-------------------------------------------------------------------------
def transcribe_audio(audio_path: str) -> str:
    mp3_path = convert_to_mp3(audio_path)
    chunks = split_audio(mp3_path)
    full_text_list = []
    for i, chunk_path in enumerate(chunks, 1):
        print(f"🔊 Транскрибируем фрагмент {i}/{len(chunks)}")
        text = transcribe_chunk(chunk_path)
        full_text_list.append(text)
        os.remove(chunk_path)
    return " ".join(full_text_list)
#-------------------------------------------------------------------------
def extract_json_array(text: str) -> List[Dict]:
    # Удаляем блоки кода, ограниченные ```...```
    text_no_code = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
    start_idx = text_no_code.find('[')
    end_idx = text_no_code.rfind(']')
    if start_idx == -1 or end_idx == -1 or start_idx > end_idx:
        raise ValueError("Не найден JSON-массив")
    array_str = text_no_code[start_idx:end_idx+1].strip()
    return json.loads(array_str)

def process_transcription(full_text: str) -> List[Dict]:
    prompt = f"""
Analyze the basketball match transcript and extract foul moments. Return JSON array with objects containing:
  "start_time": start in seconds,
  "end_time": end in seconds,
  "text": foul description.

Transcript:
{full_text}
"""
    headers = {
        "Authorization": f"Bearer {Config.DEEPINFRA_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "meta-llama/Meta-Llama-3-70B-Instruct",
        "input": prompt,
        "temperature": 0.3,
        "max_tokens": 2000
    }
    response = requests.post(Config.LLM_API_URL, headers=headers, json=payload)
    if response.status_code != 200:
        print(f"Ошибка LLM: {response.status_code}")
        return []
    try:
        data = response.json()
        result = data.get("results", [{}])[0].get("generated_text", "")
        return extract_json_array(result)
    except Exception as e:
        print(f"Ошибка парсинга: {e}")
        return []

#-------------------------------------------------------------------------
def download_video(url: str) -> str:
    print("🔄 Скачиваем видео...")
    ydl_opts = {
        'format': 'bestvideo+bestaudio/best',
        'outtmpl': os.path.join(Config.TEMP_DIR, '%(title)s.%(ext)s'),
        'merge_output_format': 'mp4',
        'quiet': True,
    }
    if os.path.exists(Config.COOKIES_PATH):
        ydl_opts['cookies'] = Config.COOKIES_PATH
    else:
        print("⚠ Файл cookies не найден")
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            return ydl.prepare_filename(info)
    except Exception as e:
        print(f"❌ Ошибка скачивания: {e}")
        sys.exit(1)
#-------------------------------------------------------------------------
def extract_audio(video_path: str) -> str:
    audio_path = Path(video_path).with_suffix('.wav')
    (
        ffmpeg
        .input(video_path)
        .output(str(audio_path), acodec='pcm_s16le', ac=1, ar=Config.SAMPLE_RATE)
        .overwrite_output()
        .run(quiet=True)
    )
    return str(audio_path)
#-------------------------------------------------------------------------
def extract_frames_segment(video_path: str, start: float, end: float, idx: int):
    foul_dir = Path(Config.DRIVE_PATH) / f"foul_{idx}"
    video_dir = foul_dir / "video"
    frames_dir = foul_dir / "frames"

    video_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)

    video_file = video_dir / f"foul_{idx}.{Config.VIDEO_EXT}"
    metadata_file = video_dir / "metadata.json"

    try:
        # Сохраняем видео
        (
            ffmpeg
            .input(video_path, ss=start, to=end)
            .output(
                str(video_file),
                vcodec=Config.VIDEO_CODEC,
                acodec=Config.AUDIO_CODEC,
                pix_fmt=Config.PIX_FMT,
                **{'vsync': 'vfr', 'movflags': 'faststart'}
            )
            .overwrite_output()
            .run(quiet=True)
        )

        # Метаданные
        metadata = {
            "foul_id": idx,
            "original_video": Path(video_path).name,
            "start_time": start,
            "end_time": end,
            "frames_dir": str(frames_dir.relative_to(foul_dir)),
            "video_path": str(video_file.relative_to(foul_dir))
        }
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        # Извлекаем кадры
        (
            ffmpeg
            .input(str(video_file))
            .output(
                str(frames_dir / "frame_%04d.jpg"),
                r=Config.FRAME_RATE,
                **{'qscale:v': '2'}
            )
            .overwrite_output()
            .run(quiet=True)
        )

        print(f"✅ Фол #{idx} сохранен")
        print(f"Видео: {video_file}")
        print(f"Кадры: {len(list(frames_dir.glob('*.jpg')))} шт\n")

    except ffmpeg.Error as e:
        print(f"❌ Ошибка обработки фола #{idx}: {e.stderr.decode()}")
#-------------------------------------------------------------------------
def process_video(url: str):
    clear_output()
    print("🚀 Начало обработки...")
    try:
        # 1. Скачивание
        video_path = download_video(url)
        print(f"✅ Видео: {Path(video_path).name}")

        # 2. Аудио
        audio_path = extract_audio(video_path)
        print("✅ Аудио извлечено")

        # 3. Транскрибация
        full_text = transcribe_audio(audio_path)
        os.remove(audio_path)
        print("✅ Текст расшифрован")

        # 4. Анализ фолов
        fouls = process_transcription(full_text)
        print(f"🎯 Найдено фолов: {len(fouls)}")

        # 5. Обработка сегментов
        for i, foul in enumerate(fouls, start=1):
            start = foul.get("start_time", 0)
            end = foul.get("end_time", 0)
            text = foul.get("text", "")

            print(f"\n⚡ Фол #{i}: {start}-{end}с")
            print(f"📝 {text[:100]}...")
            extract_frames_segment(video_path, start, end, i)

        # Удаление исходного видео
        os.remove(video_path)
        print(f"\n✅ Все данные сохранены в: {Config.DRIVE_PATH}")

    except Exception as e:
        print(f"🔥 Критическая ошибка: {e}")
        sys.exit(1)
#-------------------------------------------------------------------------
if __name__ == "__main__":
    process_video(Config.VIDEO_URL)
