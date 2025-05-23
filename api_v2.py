import os
import sys
import traceback
from typing import Generator

now_dir = os.getcwd()
sys.path.append(now_dir)
sys.path.append(f"{now_dir}/GPT_SoVITS")

import argparse
import subprocess
import wave
import signal
import numpy as np
import soundfile as sf
from fastapi import FastAPI, Response
from fastapi.responses import StreamingResponse, JSONResponse
import uvicorn
from io import BytesIO
from tools.i18n.i18n import I18nAuto
from GPT_SoVITS.TTS_infer_pack.text_segmentation_method import get_method_names as get_cut_method_names
from pydantic import BaseModel

speaker_list = {}

class DefaultRefer:
    def __init__(self, path: str, text: str, lang: str):
        self.path = path
        self.text = text
        self.lang = lang

class Speaker:
    def __init__(self, name: str, tts_pipeline, default_refer: DefaultRefer):
        self.name = name
        self.tts_pipeline = tts_pipeline
        self.default_refer = default_refer


from app.services.voice.tts_model_registry import get_tts_pipeline
i18n = I18nAuto()
cut_method_names = get_cut_method_names()

parser = argparse.ArgumentParser(description="GPT-SoVITS api")
parser.add_argument("-c", "--tts_config", type=str, default="GPT_SoVITS/configs/tts_infer.yaml", help="tts_infer路径")
parser.add_argument("-a", "--bind_addr", type=str, default="127.0.0.1", help="default: 127.0.0.1")
parser.add_argument("-p", "--port", type=int, default="9880", help="default: 9880")
args = parser.parse_args()
config_path = args.tts_config
port = args.port
host = args.bind_addr
argv = sys.argv

if config_path in [None, ""]:
    config_path = "GPT-SoVITS/configs/tts_infer.yaml"

APP = FastAPI()


class TTS_Request(BaseModel):
    text: str = None
    text_lang: str = None
    ref_audio_path: str = None
    aux_ref_audio_paths: list = None
    prompt_lang: str = None
    prompt_text: str = ""
    top_k: int = 5
    top_p: float = 1
    temperature: float = 1
    text_split_method: str = "cut5"
    batch_size: int = 1
    batch_threshold: float = 0.75
    split_bucket: bool = True
    speed_factor: float = 1.0
    fragment_interval: float = 0.3
    seed: int = -1
    media_type: str = "wav"
    streaming_mode: bool = False
    parallel_infer: bool = True
    repetition_penalty: float = 1.35
    sample_steps: int = 32
    super_sampling: bool = False


def pack_ogg(io_buffer: BytesIO, data: np.ndarray, rate: int):
    with sf.SoundFile(io_buffer, mode="w", samplerate=rate, channels=1, format="ogg") as audio_file:
        audio_file.write(data)
    return io_buffer

def pack_raw(io_buffer: BytesIO, data: np.ndarray, rate: int):
    io_buffer.write(data.tobytes())
    return io_buffer

def pack_wav(io_buffer: BytesIO, data: np.ndarray, rate: int):
    io_buffer = BytesIO()
    sf.write(io_buffer, data, rate, format="wav")
    return io_buffer

def pack_aac(io_buffer: BytesIO, data: np.ndarray, rate: int):
    process = subprocess.Popen(
        ["ffmpeg","-f","s16le","-ar",str(rate),"-ac","1","-i","pipe:0",
         "-c:a","aac","-b:a","192k","-vn","-f","adts","pipe:1"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    out, _ = process.communicate(input=data.tobytes())
    io_buffer.write(out)
    return io_buffer

def pack_audio(io_buffer: BytesIO, data: np.ndarray, rate: int, media_type: str):
    if media_type == "ogg":
        io_buffer = pack_ogg(io_buffer, data, rate)
    elif media_type == "aac":
        io_buffer = pack_aac(io_buffer, data, rate)
    elif media_type == "wav":
        io_buffer = pack_wav(io_buffer, data, rate)
    else:
        io_buffer = pack_raw(io_buffer, data, rate)
    io_buffer.seek(0)
    return io_buffer

def wave_header_chunk(frame_input=b"", channels=1, sample_width=2, sample_rate=32000):
    wav_buf = BytesIO()
    with wave.open(wav_buf, "wb") as vfout:
        vfout.setnchannels(channels)
        vfout.setsampwidth(sample_width)
        vfout.setframerate(sample_rate)
        vfout.writeframes(frame_input)
    wav_buf.seek(0)
    return wav_buf.read()

def handle_control(command: str):
    if command == "restart":
        os.execl(sys.executable, sys.executable, *argv)
    elif command == "exit":
        os.kill(os.getpid(), signal.SIGTERM)
        exit(0)


def check_params(
    req: dict,
    supported_languages: list[str],
    version: str
):
    text = req.get("text", "")
    text_lang = req.get("text_lang", "")
    prompt_lang = req.get("prompt_lang", "")
    streaming_mode = req.get("streaming_mode", False)
    media_type = req.get("media_type", "wav")
    text_split_method = req.get("text_split_method", "cut5")

    # text
    if not text:
        return JSONResponse(status_code=400, content={"message": "text is required"})

    # text_lang
    if not text_lang:
        return JSONResponse(status_code=400, content={"message": "text_lang is required"})
    if text_lang.lower() not in supported_languages:
        return JSONResponse(
            status_code=400,
            content={"message": f"text_lang: {text_lang} is not supported in version {version}"},
        )

    # prompt_lang
    if not prompt_lang:
        return JSONResponse(status_code=400, content={"message": "prompt_lang is required"})
    if prompt_lang.lower() not in supported_languages:
        return JSONResponse(
            status_code=400,
            content={"message": f"prompt_lang: {prompt_lang} is not supported in version {version}"},
        )

    # media_type
    if media_type not in ["wav", "raw", "ogg", "aac"]:
        return JSONResponse(status_code=400, content={"message": f"media_type: {media_type} is not supported"})
    if media_type == "ogg" and not streaming_mode:
        return JSONResponse(status_code=400, content={"message": "ogg format is not supported in non-streaming mode"})

    # text_split_method
    if text_split_method not in cut_method_names:
        return JSONResponse(
            status_code=400,
            content={"message": f"text_split_method: {text_split_method} is not supported"},
        )

    return None

async def tts_handle(req: dict):
    spk = req.get("spk")
    if not spk:
        return JSONResponse(status_code=400, content={"message": "spk is required"})
    if spk not in speaker_list:
        return JSONResponse(status_code=400, content={"message": f"speaker '{spk}' not found"})
    speaker = speaker_list[spk]

    from app.services.voice.tts_model_registry import get_tts_pipeline
    tts_pipe = get_tts_pipeline(spk)
    tts_pipe.set_ref_audio(speaker.default_refer.path)

    req["ref_audio_path"] = speaker.default_refer.path
    req["prompt_text"]    = req.get("prompt_text") or speaker.default_refer.text
    req["prompt_lang"]    = speaker.default_refer.lang

    streaming_mode  = req.get("streaming_mode", False)
    return_fragment = req.get("return_fragment", False)
    media_type      = req.get("media_type", "wav")

    check_res = check_params(
        req,
        supported_languages=tts_pipe.configs.languages,
        version=tts_pipe.configs.version
    )
    if check_res is not None:
        return check_res
    if streaming_mode or return_fragment:
        req["return_fragment"] = True

    try:
        tts_generator = tts_pipe.run(req)

        if streaming_mode:
            first_chunk = True
            def streaming_generator():
                nonlocal first_chunk
                for sr, chunk in tts_generator:
                    if first_chunk and media_type == "wav":
                        yield wave_header_chunk(sample_rate=sr)
                        first_chunk = False
                    yield pack_audio(BytesIO(), chunk, sr, media_type).getvalue()
            return StreamingResponse(streaming_generator(), media_type=f"audio/{media_type}")

        else:
            sr, audio_data = next(tts_generator)
            audio_bytes = pack_audio(BytesIO(), audio_data, sr, media_type).getvalue()
            return Response(audio_bytes, media_type=f"audio/{media_type}")

    except Exception as e:
        return JSONResponse(status_code=400, content={"message": "tts failed", "exception": str(e)})

@APP.get("/control")
async def control_endpoint(command: str = None):
    if not command:
        return JSONResponse(status_code=400, content={"message": "command is required"})
    handle_control(command)

@APP.get("/tts")
async def tts_get_endpoint(
    text: str,
    text_lang: str,
    spk: str = "default",
    top_k: int = 5,
    top_p: float = 1.0,
    temperature: float = 1.0,
    text_split_method: str = "cut5",
    batch_size: int = 1,
    batch_threshold: float = 0.75,
    split_bucket: bool = True,
    speed_factor: float = 1.0,
    fragment_interval: float = 0.3,
    seed: int = -1,
    media_type: str = "wav",
    streaming_mode: bool = False,
    parallel_infer: bool = True,
    repetition_penalty: float = 1.35,
    sample_steps: int = 32,
    super_sampling: bool = False,
):
    req = {
        "text": text,
        "text_lang": text_lang.lower(),
        "aux_ref_audio_paths": [],
        "top_k": top_k,
        "top_p": top_p,
        "temperature": temperature,
        "text_split_method": text_split_method,
        "batch_size": batch_size,
        "batch_threshold": batch_threshold,
        "split_bucket": split_bucket,
        "speed_factor": speed_factor,
        "fragment_interval": fragment_interval,
        "seed": seed,
        "media_type": media_type,
        "streaming_mode": streaming_mode,
        "parallel_infer": parallel_infer,
        "repetition_penalty": repetition_penalty,
        "sample_steps": sample_steps,
        "super_sampling": super_sampling,
        "return_fragment": False,
        "spk": spk,
    }
    return await tts_handle(req)


@APP.post("/tts")
async def tts_post_endpoint(request: TTS_Request):
    req = request.dict()
    req["spk"] = req.get("spk", "default")
    return await tts_handle(req)

if __name__ == "__main__":
    try:
        if host == "None":
            host = None
        uvicorn.run(app=APP, host=host, port=port, workers=1)
    except Exception:
        traceback.print_exc()
        os.kill(os.getpid(), signal.SIGTERM)
        exit(0)
