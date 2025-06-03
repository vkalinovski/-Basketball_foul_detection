# DeepInfra NBA Foul Detection & Frame Extraction

![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg) 
![FFmpeg](https://img.shields.io/badge/FFmpeg-5.x-green.svg)
![Requests](https://img.shields.io/badge/Requests-2.x-orange.svg)
![yt--dlp](https://img.shields.io/badge/yt_dlp-2023.07-yellow.svg)
![DeepInfra](https://img.shields.io/badge/DeepInfra-API-9cf.svg)
![Colab](https://img.shields.io/badge/Google%20Colab-compatible-brightgreen.svg)

<div align="center">
  <img src="https://raw.githubusercontent.com/github/explore/main/topics/youtube/youtube.png" alt="YouTube" width="70" />
  <img src="https://raw.githubusercontent.com/github/explore/main/topics/ffmpeg/ffmpeg.png" alt="FFmpeg" width="70" />
  <img src="https://raw.githubusercontent.com/github/explore/main/topics/python/python.png" alt="Python" width="70" />
</div>

## Overview

This project allows you to:
1. **Download videos** from YouTube (using `yt_dlp` with cookies).
2. **Extract audio** from the video and transcribe it via the [DeepInfra Whisper API](https://deepinfra.com).
3. **Analyze** the transcription with DeepInfra’s LLM to **identify foul moments** (segments containing key phrases).
4. **Extract frames** from each detected interval, **saving them into separate folders** on Google Drive.
5. Also save the video clip corresponding to each detected interval.

Important:
- Keep `cookies.txt` updated.
- Replace `DRIVE_PATH` with your Google Drive path.
- Adjust `FRAME_RATE` and `MAX_AUDIO_SIZE_MB` as needed.

## Main Pipeline

1. **Video Download**:
   - Use `yt_dlp` with cookie support (for YouTube authentication).
   - Upload `cookies.txt` (Netscape format) to Colab and set `COOKIES_PATH`.

2. **Audio Extraction**:
   - Use `ffmpeg` to convert the video to `*.wav`.

3. **Transcription**:
   - Convert WAV → MP3 (custom bitrate and sample rate).
   - Split into chunks (default `5` MB) to avoid timeouts.
   - Send each chunk to the DeepInfra Whisper API. After successful transcription, delete the temporary file.

4. **Text Analysis (LLM)**:
   - Send the complete transcription to DeepInfra’s LLM (Meta-Llama).
   - The LLM returns a **JSON array**:  
     ```
     [
       { "start_time": <float>, "end_time": <float>, "text": <string> },
       ...
     ]
     ```
   - Times represent the estimated start and end seconds of each foul.

5. **Frame Extraction**:
   - For each foul, create a folder `foul_i`.
   - Use `ffmpeg` to save frames (default `18` frames/sec) from `[start_time, end_time]`.
   - You’ll get files like `frame_0001.jpg, frame_0002.jpg, ...` in `foul_i/frames/`.
   - Also save the corresponding video segment as a `.mp4` file.

## Dependencies

- [Python 3.9+](https://www.python.org/downloads/)
- [Requests](https://requests.readthedocs.io/)
- [ffmpeg](https://ffmpeg.org/) 
- [yt_dlp](https://github.com/yt-dlp/yt-dlp)
- [pydub](https://github.com/jiaaro/pydub)
- [nest_asyncio](https://github.com/erdewit/nest_asyncio)
- [DeepInfra API](https://deepinfra.com)
