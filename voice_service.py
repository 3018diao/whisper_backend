import os
import subprocess
import time
import wave
import speech_recognition as sr
from typing import Optional, Tuple, Dict
import logging
import re

# 获取记录器
logger = logging.getLogger(__name__)  # 使用 __name__ 而不是创建新的基础配置

class VoiceService:
    def __init__(self) -> None:
        self._whisper_path = "/home/huiyu/whisper.cpp"
        self._output_dir = "outputs"
        self._model_dir = os.path.join(self._whisper_path, "models")
        os.makedirs(self._output_dir, exist_ok=True)
        
    def listen(self) -> tuple[str, float]:
        """
        录制并处理音频
        返回: (转录文本, 处理时长)
        """
        logger.info('Listening (mode: offline)...')
        try:
            transcribed_file = f"{self._output_dir}/transcribed-audio.wav"
            r = sr.Recognizer()
            
            with sr.Microphone(sample_rate=16000) as source:
                # 配置语音识别参数
                r.dynamic_energy_threshold = True    # 动态能量阈值
                r.energy_threshold = 4000           # 音量阈值
                r.pause_threshold = 0.8             # 停顿阈值，超过这个时间认为说完一句话
                r.phrase_threshold = 0.3            # 短语阈值
                r.non_speaking_duration = 0.4       # 非说话持续时间
                
                # 调整环境噪音
                logger.info("正在调整环境噪音，请稍等...")
                r.adjust_for_ambient_noise(source, duration=1)
                
                logger.info("开始录音，请说话...")
                try:
                    # 等待说话并录制
                    audio = r.listen(
                        source,
                        timeout=None,           # 无超时限制
                        phrase_time_limit=None  # 无短语时间限制
                    )
                    logger.info("录音完成，开始处理...")
                    
                except sr.WaitTimeoutError:
                    logger.error("等待超时，未检测到语音输入")
                    return "未检测到语音输入", 0
                    
                except Exception as e:
                    logger.error(f"录音过程出错: {e}")
                    return f"录音错误: {str(e)}", 0
            
            try:
                # 保存音频文件
                with open(transcribed_file, "wb") as f:
                    f.write(audio.get_wav_data())
                
                # 处理音频文件
                start_time = time.time()
                result = self.process_audio(transcribed_file)
                duration = time.time() - start_time
                
                logger.info(f"处理完成，结果: {result}")
                
                # 清理临时文件
                if os.path.exists(transcribed_file):
                    os.remove(transcribed_file)
                    
                return result, duration
                
            except Exception as e:
                logger.error(f"处理过程出错: {e}")
                if os.path.exists(transcribed_file):
                    os.remove(transcribed_file)
                return f"处理错误: {str(e)}", 0
                
        except Exception as e:
            logger.error(f"语音识别过程出错: {e}")
            return f"识别错误: {str(e)}", 0
        
    def process_audio(self, wav_file: str, model_name: str = "ggml-tiny.bin") -> str:
        """
        处理音频文件
        参数:
            wav_file: WAV文件路径
            model_name: 模型文件名（默认使用 tiny 模型）
        返回:
            识别的文本
        """
        try:
            # 1. 检查音频文件
            if not os.path.exists(wav_file):
                raise FileNotFoundError(f"音频文件不存在: {wav_file}")
                
            # 2. 检查音频格式
            try:
                import wave
                with wave.open(wav_file, 'rb') as wf:
                    print(f"音频文件信息:")
                    print(f"- 声数: {wf.getnchannels()}")
                    print(f"- 采样宽度: {wf.getsampwidth()}")
                    print(f"- 采样率: {wf.getframerate()}")
                    print(f"- 总帧数: {wf.getnframes()}")
                    print(f"- 参数: {wf.getparams()}")

                    # 检查采样率
                    if wf.getframerate() != 16000:
                        raise ValueError(f"音频采样率必须是16kHz，当前是{wf.getframerate()}Hz")
            except Exception as e:
                print(f"音频文件检查错误: {e}")

            # 3. 检查模型文件
            model_path = os.path.join(self._whisper_path, "models", model_name)
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"模型文件不存在: {model_path}")
                
            print(f"使用模型: {model_path}")
                
            # 4. 构建命令 - 只用支持的参数
            command = [
                os.path.join(self._whisper_path, "main"),
                "-m", model_path,
                "-f", wav_file,
                "-l", "auto",     # 自动检测语言
                "-np",           # 不显示进度条
                "-nt",          # 不显示时间戳
                "--max-len", "0"  # 不限制输出长度
            ]
                
            print(f"执行命令: {' '.join(command)}")
                
            # 5. 执行命令
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
                
            # 6. 获取输出
            stdout, stderr = process.communicate()
                
            print(f"标准输出: {stdout}")
            if stderr:
                print(f"标准错误: {stderr}")
                
            # 7. 检查返回码
            if process.returncode != 0:
                error_msg = f"Whisper处理失败 (返回码: {process.returncode}): {stderr}"
                print(error_msg)
                raise RuntimeError(error_msg)
                
            # 8. 处理输出文本
            text = stdout.strip()
                
            # 9. 特���情况处理
            if not text:
                print("没有检测到文本")
                return "未检测到语音内容"
                
            if text == '[BLANK_AUDIO]':
                print("检测到空白音频")
                return "检测到空白音频"
                
            # 10. 清理并返回结果
            result = text.replace('[BLANK_AUDIO]', '').strip()
            print(f"最终识别结果: {result}")
                
            return result
                
        except Exception as e:
            print(f"处理音频时发生错误: {str(e)}")
            raise RuntimeError(f"音频处理失败: {str(e)}")

    def get_available_models(self) -> Dict[str, str]:
        """
        获取可用的模型列表
        返回: 模型名称和路径的字典
        """
        models = {}
        try:
            for file in os.listdir(self._model_dir):
                if file.startswith("ggml-") and file.endswith(".bin"):
                    model_name = file.replace("ggml-", "").replace(".bin", "")
                    models[model_name] = os.path.join(self._model_dir, file)
        except Exception as e:
            logger.error(f"Error listing models: {e}")
        return models

    def check_model_exists(self, model_name: str) -> bool:
        """
        检查指定模型是否存在
        """
        model_path = os.path.join(self._model_dir, f"ggml-{model_name}.bin")
        return os.path.exists(model_path)

    def process_stream(self, audio_file: str, model_name: str = "ggml-tiny.bin") -> list[str]:
        try:
            # 直接使用 RAW 文件
            cmd = [
                os.path.join(self._whisper_path, "stream"),
                "-m", os.path.join(self._model_dir, model_name),
                "-f", audio_file,     # 直接使用原始音频文件
                "-t", "8",            # 8线程
                "--step", "500",      # 步长500ms
                "--length", "5000",   # 处理窗口5000ms
                "-nr",                # 指定为 RAW 格式
                "-sr", "16000",       # 采样率
                "-ch", "1",           # 单声道
                "-bd", "16",          # 16位深度
            ]
            
            logger.info(f"执行命令: {' '.join(cmd)}")
            
            # 运行命令
            process = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                check=True,
                cwd=self._whisper_path
            )
            
            # 检查输出
            if process.stderr:
                logger.info(f"whisper stderr输出: {process.stderr}")
            
            result = process.stdout.strip()
            logger.info(f"whisper 原始输出: {result}")
            
            return self._clean_whisper_output(result)
            
        except subprocess.CalledProcessError as e:
            logger.error(f"whisper 命令行错误: {e.stderr}")
            raise RuntimeError(f"Whisper处理失败: {e.stderr}")
        except Exception as e:
            logger.error(f"处理错误: {e}")
            raise RuntimeError(f"音频处理失败: {str(e)}")

    def _clean_whisper_output(self, text: str) -> list[str]:
        """
        清理 whisper.cpp 输出中的特殊标记，并按句子分割
        返回: 句子列表
        """
        logger.info(f"清理前的文本: {text}")
        
        # 移除 ANSI 转义序列
        text = re.sub(r'\x1b\[[0-9;]*[a-zA-Z]', '', text)
        
        # 移除所有特殊标记
        text = re.sub(r'\[.*?\]', '', text)
        
        # 移除多余的空白行
        text = re.sub(r'\n\s*\n', '\n', text)
        
        # 移除开头和结尾的空白
        text = text.strip()
        
        # 移除每行开头和结尾的空白
        lines = [line.strip() for line in text.split('\n')]
        text = ' '.join(line for line in lines if line)
        
        logger.info(f"清理后的文本: {text}")
        
        # 按句子分割（使用句号、问号、感叹号作为分隔符）
        sentences = re.split(r'[.。!！?？]+', text)
        
        # 清理每个句子并过滤掉空句子和重复句子
        seen = set()
        cleaned_sentences = []
        for s in sentences:
            s = s.strip()
            if s and s not in seen:
                cleaned_sentences.append(s)
                seen.add(s)
        
        logger.info(f"最终句子列表: {cleaned_sentences}")
        return cleaned_sentences