from fastapi import FastAPI, WebSocket, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from starlette.websockets import WebSocketDisconnect
from pydantic import BaseModel
import uvicorn
import os
import time
import shutil
from typing import Optional, List
import logging
import sys
from logging.handlers import RotatingFileHandler
from voice_service import VoiceService
import base64
import json
import numpy as np
import asyncio  # 添加这个导入

# 创建logs目录
os.makedirs('logs', exist_ok=True)

# 获取当前时间戳作为日志文件名
log_filename = time.strftime('app_%Y%m%d_%H%M%S.log')
log_filepath = os.path.join('logs', log_filename)

# 配置日志格式
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# 创建文件处理器
file_handler = RotatingFileHandler(
    log_filepath,  # 使用带时间戳的文件名
    maxBytes=1024*1024,  # 1MB
    backupCount=5,
    encoding='utf-8'
)
file_handler.setFormatter(formatter)
file_handler.setLevel(logging.INFO)

# 创建控制台处理器
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(formatter)
console_handler.setLevel(logging.INFO)

# 配置根日志记录器
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# 确保不会重复添加处理器
logger.propagate = False

# 配置 uvicorn 访问日志
logging.getLogger("uvicorn.access").handlers = [file_handler, console_handler]
logging.getLogger("uvicorn.error").handlers = [file_handler, console_handler]

# 记录服务启动信息
logger.info(f"服务启动，日志文件: {log_filepath}")

# 强制刷新输出
sys.stdout.reconfigure(line_buffering=True)

# 获取项目根目录
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
STATIC_DIR = os.path.join(BASE_DIR, "backend", "static")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

# 确保必要的目录存在
os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 创建 FastAPI 应用
app = FastAPI(
    title="Voice Recognition System",
    description="语音识别系统 API",
    version="1.0.0"
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 创建语音服务实例
voice_service = VoiceService()

# 响应模型
class RecognitionResponse(BaseModel):
    text: str
    duration: float
    status: str

# API路由
@app.get("/")
async def read_root():
    """
    提供前端页面
    """
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))

@app.post("/api/recognize", response_model=RecognitionResponse)
async def recognize_audio(
    file: UploadFile = File(...),
    model: Optional[str] = "ggml-tiny.bin"
):
    try:
        # 生成临时文件
        temp_file = os.path.join(voice_service._output_dir, f"temp_{int(time.time())}.raw")
        
        # 保存上传的文件
        try:
            with open(temp_file, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            logger.info(f"Saved uploaded file to {temp_file}")
        except Exception as e:
            logger.error(f"Error saving uploaded file: {e}")
            raise HTTPException(status_code=500, detail=f"Error saving file: {str(e)}")
        
        try:
            # 处理音频
            start_time = time.time()
            result = voice_service.process_stream(temp_file, model)
            duration = time.time() - start_time
            
            if not result or not result.strip():
                logger.warning("No text detected in audio")
                return RecognitionResponse(
                    text="",
                    duration=duration,
                    status="no_text"
                )
            
            logger.info(f"Successfully processed audio. Result: {result}")
            return RecognitionResponse(
                text=result,
                duration=duration,
                status="success"
            )
            
        except Exception as e:
            logger.error(f"Audio processing error: {e}")
            raise HTTPException(status_code=500, detail=f"Audio processing failed: {str(e)}")
            
    except Exception as e:
        logger.error(f"Recognition error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
        
    finally:
        # 清理临时文件
        try:
            if os.path.exists(temp_file):
                os.remove(temp_file)
        except Exception as e:
            logger.error(f"Error cleaning up temp file {temp_file}: {e}")

@app.post("/api/listen", response_model=RecognitionResponse)
async def listen():
    """
    直接录音并识别
    """
    try:
        result, duration = voice_service.listen()
        return RecognitionResponse(
            text=result,
            duration=duration,
            status="success"
        )
    except Exception as e:
        logger.error(f"Listening error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/status")
async def get_status():
    """
    检查服务状态
    """
    return {
        "status": "running",
        "models": voice_service.get_available_models()
    }

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket connection accepted")
    
    # 用于控制任务的事件
    shutdown_event = asyncio.Event()
    
    # 用于累积音频数据
    audio_buffer = []
    buffer_duration = 0  # 累积的音频时长（秒）
    MIN_DURATION = 2.0   # 增加到2秒
    
    async def process_audio():
        try:
            nonlocal audio_buffer, buffer_duration
            
            while True:  # 移除 shutdown_event 检查
                try:
                    data = await websocket.receive_bytes()
                    header = np.frombuffer(data[:8], dtype=np.int32)
                    sample_rate, num_samples = header
                    audio_data = np.frombuffer(data[8:], dtype=np.int16)
                    volume = np.abs(audio_data).mean()
                    
                    logger.info(f"音频信息: 采样率={sample_rate}Hz, 样本数={num_samples}, 平均音量={volume:.2f}")
                    
                    # 音量太小则跳过
                    if volume < 500:  # 调整音量阈值
                        continue
                        
                    # 累积音频数据
                    audio_buffer.append(audio_data)
                    buffer_duration += num_samples / sample_rate
                    
                    # 达到最小时长则处理
                    if buffer_duration >= MIN_DURATION:
                        # 合并音频数据
                        combined_audio = np.concatenate(audio_buffer)
                        logger.info(f"合并音频数据: {len(combined_audio)} 样本")
                        
                        # 计算音量统计
                        mean_volume = np.abs(combined_audio).mean()
                        max_volume = np.abs(combined_audio).max()
                        logger.info(f"音频统计: 平均音量={mean_volume:.2f}, 最大音量={max_volume:.2f}")
                        
                        # 保存为临时文件
                        temp_file = os.path.join(voice_service._output_dir, f"temp_{int(time.time())}.raw")
                        combined_audio.tofile(temp_file)
                        logger.info(f"保存累积音频数据: {temp_file}, 时长: {buffer_duration:.2f}秒")
                        
                        try:
                            # 处理音频
                            result = voice_service.process_stream(temp_file)
                            logger.info(f"处理结果: {result}")
                            
                            # 无论是否有识别结果都发送消息
                            if result:
                                for sentence in result:
                                    if sentence:
                                        await websocket.send_json({
                                            "status": "success",
                                            "text": sentence
                                        })
                            else:
                                # 发送空结果消息
                                await websocket.send_json({
                                    "status": "no_text",
                                    "text": "未识别到文本"
                                })
                                logger.info("发送空结果消息")
                                
                        except Exception as e:
                            logger.error(f"音频处理错误: {e}")
                            # 发送错误消息
                            await websocket.send_json({
                                "status": "error",
                                "text": f"处理错误: {str(e)}"
                            })
                        finally:
                            # 清理临时文件
                            if os.path.exists(temp_file):
                                os.remove(temp_file)
                                logger.info(f"Cleaned up {temp_file}")
                            
                        # 重置缓冲区
                        audio_buffer = []
                        buffer_duration = 0
                        
                except WebSocketDisconnect:
                    logger.info("WebSocket disconnected in audio processing")
                    break
        except Exception as e:
            logger.error(f"处理音频数据时出错: {e}")
        finally:
            logger.info("音频处理任务结束")
    
    try:
        # 创建并等待音频处理任务
        audio_task = asyncio.create_task(process_audio())
        await audio_task
            
    except Exception as e:
        logger.error(f"WebSocket错误: {e}")
    finally:
        # 清理资源
        try:
            await websocket.close()
            logger.info("WebSocket connection closed")
        except Exception as e:
            logger.error(f"关闭WebSocket连接出错: {e}")

# 添加优雅关闭处理
@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Application shutting down")
    # 这里可以添加其他清理代码

# 挂载静态文件
# 注意：这里要把静态文件路由放在最后，避免覆盖API路由
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        access_log=True
    )