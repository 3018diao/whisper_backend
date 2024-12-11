import subprocess
import threading
import queue
import time
import pyaudio
import wave
import numpy as np
from typing import Optional

class WhisperStream:
    def __init__(self, 
                 model_path: str,
                 whisper_cpp_path: str,
                 sample_rate: int = 44100,
                 chunk_size: int = 1024,
                 channels: int = 1,
                 input_device_index: Optional[int] = None,
                 language: str = "en"):
        self.model_path = model_path
        self.whisper_cpp_path = whisper_cpp_path
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.channels = channels
        self.input_device_index = input_device_index
        self.language = language
        self.resample_ratio = sample_rate / 16000  # whisper需要16kHz
        
        self.audio_queue = queue.Queue()
        self.is_recording = False
        self.stream: Optional[pyaudio.Stream] = None
        self.p: Optional[pyaudio.PyAudio] = None
        
    @staticmethod
    def list_audio_devices():
        """列出所有可用的音频输入设备"""
        p = pyaudio.PyAudio()
        devices = []
        
        print("\n可用的音频输入设备:")
        try:
            for i in range(p.get_device_count()):
                try:
                    dev_info = p.get_device_info_by_index(i)
                    if dev_info['maxInputChannels'] > 0:  # 只显示输入设备
                        print(f"设备 {i}: {dev_info['name']}")
                        print(f"    输入通道: {dev_info['maxInputChannels']}")
                        print(f"    采样率: {dev_info['defaultSampleRate']}")
                        devices.append(i)
                except Exception as e:
                    print(f"警告：无法获取设备 {i} 的信息: {e}")
                    continue
        except Exception as e:
            print(f"警告：枚举设备时出错: {e}")
        
        p.terminate()
        return devices
        
    def start_recording(self):
        """开始录音并将音频数据放入队列"""
        self.is_recording = True
        self.p = pyaudio.PyAudio()
        
        def audio_callback(in_data, frame_count, time_info, status):
            self.audio_queue.put(in_data)
            return (in_data, pyaudio.paContinue)
        
        try:
            # 优先使用pulse设备
            try:
                self.stream = self.p.open(
                    format=pyaudio.paFloat32,
                    channels=self.channels,
                    rate=self.sample_rate,
                    input=True,
                    input_device_index=0,  # pulse设备通常是索引0
                    frames_per_buffer=self.chunk_size,
                    stream_callback=audio_callback
                )
                print("使用 PulseAudio 设备进行录音")
                self.stream.start_stream()
                return
            except OSError as e:
                print(f"无法使用 PulseAudio 设备: {e}")
                print("尝试使用其他设备...")
                
            # 如果指定设备失败，尝试使用默认设备
            try:
                self.stream = self.p.open(
                    format=pyaudio.paFloat32,
                    channels=self.channels,
                    rate=self.sample_rate,
                    input=True,
                    frames_per_buffer=self.chunk_size,
                    stream_callback=audio_callback
                )
                print("使用默认音频设备")
                self.stream.start_stream()
            except OSError as e:
                print(f"无法使用默认设备: {e}")
                raise Exception("无法打开任何音频输入设备，请检查系统音频设置")
                
        except Exception as e:
            print(f"启动录音时发生错误: {e}")
            raise
            
    def stop_recording(self):
        """停止录音"""
        self.is_recording = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        if self.p:
            self.p.terminate()
            
    def process_audio(self):
        """处理音频队列中的数据并调用whisper.cpp进行识别"""
        temp_wav = "temp_chunk.wav"
        print("开始处理音频...")
        print("请对着麦克风说话...")
        
        # 存储上一次的音频电平，用于检测变化
        last_db = -100
        
        while self.is_recording:
            # 收集音频数据（8秒）
            audio_data = b''
            chunks_to_collect = int(self.sample_rate * 8 / self.chunk_size)
            
            for _ in range(chunks_to_collect):
                if not self.audio_queue.empty():
                    chunk = self.audio_queue.get()
                    audio_data += chunk
                else:
                    time.sleep(0.1)
                    continue
            
            if not audio_data:
                continue
                
            try:
                # 转换为numpy数组进行处理
                audio_array = np.frombuffer(audio_data, dtype=np.float32)
                
                # 计算分贝值
                rms = np.sqrt(np.mean(np.square(audio_array)))
                db = 20 * np.log10(rms) if rms > 0 else -100
                
                # 只在音量显著变化时更新显示
                if abs(db - last_db) > 3:
                    volume_bars = "█" * int((-db//10))
                    print(f"\r音量: {volume_bars} {db:.1f} dB         ", end='')
                    last_db = db
                
                # 如果音量太小，跳过处理
                if db < -50:  # 调整静音阈值
                    continue
                
                # 音频预处理
                # 1. 去除直流偏置
                audio_array = audio_array - np.mean(audio_array)
                
                # 2. 归一化到 [-1, 1]
                max_amp = np.max(np.abs(audio_array))
                if max_amp > 0:
                    audio_array = audio_array / max_amp * 0.9
                
                # 3. 重采样到16kHz (whisper要求)
                resampled = audio_array[::int(self.resample_ratio)]
                
                # 4. 转换到16位整数格式 (-32768 到 32767)
                int16_data = (resampled * 32767).astype(np.int16)
                
                # 保存为16位WAV文件
                with wave.open(temp_wav, 'wb') as wf:
                    wf.setnchannels(self.channels)
                    wf.setsampwidth(2)  # 16-bit = 2 bytes
                    wf.setframerate(16000)  # Whisper需要16kHz
                    wf.writeframes(int16_data.tobytes())
                
                print("\n正在识别...")
                
                # 调用whisper.cpp进行识别
                cmd = [
                    self.whisper_cpp_path,
                    "-m", self.model_path,
                    "-f", temp_wav,
                    "--no-timestamps",
                    "--print-special",
                    "--language", self.language,
                    "-t", "8",        # 8线程
                    "-p", "1",        # 1个处理器
                ]
                
                result = subprocess.run(cmd, 
                                     capture_output=True, 
                                     text=True,
                                     encoding='utf-8',
                                     check=True)
                
                if result.stdout.strip():
                    print(f"[识别结果] >>> {result.stdout.strip()}")
                    print("-" * 50)
                    print("请继续说话...")
                
            except subprocess.CalledProcessError as e:
                if "failed to decode" not in str(e.stderr):  # 忽略常见的解码错误
                    print(f"\nWhisper错误: {e}")
            except Exception as e:
                print(f"\n处理错误: {e}")
                
            time.sleep(0.1)  # 短暂暂停
                
    def run(self):
        """启动流式识别"""
        try:
            self.start_recording()
            # 在新线程中处理音频
            processing_thread = threading.Thread(target=self.process_audio)
            processing_thread.start()
            
            # 等待用户输入来停止
            input("按回车键停止识别...")
            
        finally:
            self.stop_recording()

if __name__ == "__main__":
    # 首先列出所有可用的音频设备
    available_devices = WhisperStream.list_audio_devices()
    
    if not available_devices:
        print("未找到可用的音频输入设备！")
        print("\n如果您在 WSL 环境下，请确保：")
        print("1. 确保 Windows 系统中有可用的麦克风")
        print("2. 检查 Windows 的隐私设置，确保允许应用访问麦克风")
        print("3. 在 Windows 中测试麦克风是否正常工作")
        exit(1)
    
    # 让用户选择输入设备
    device_index = None
    if len(available_devices) > 1:
        while True:
            try:
                choice = int(input("\n请选择要使用的输入设备编号: "))
                if choice in available_devices:
                    device_index = choice
                    break
                else:
                    print("无效的设备编号，请重试")
            except ValueError:
                print("请输入有效的数字")
    else:
        device_index = available_devices[0]
    
    # 创建 WhisperStream 实例
    whisper = WhisperStream(
        model_path="/home/huiyu/whisper.cpp/models/ggml-tiny.bin",
        whisper_cpp_path="/home/huiyu/whisper.cpp/main",
        input_device_index=device_index,
        language="en"  # 设置为英语
    )
    
    try:
        whisper.run()
    except Exception as e:
        print(f"\n错误: {e}")
        print("如果您在使用 WSL，请确保已经正确设置了音频环境")