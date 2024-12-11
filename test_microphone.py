import pyaudio
import numpy as np
import time

class MicrophoneTester:
    def __init__(self, 
                 rate=44100, 
                 chunk_size=1024, 
                 channels=1,
                 device_index=None):
        self.rate = rate
        self.chunk_size = chunk_size
        self.channels = channels
        self.device_index = device_index
        self.p = None
        self.stream = None
        
    def list_devices(self):
        """列出所有可用的音频输入设备"""
        p = pyaudio.PyAudio()
        print("\n可用的音频输入设备:")
        for i in range(p.get_device_count()):
            try:
                dev_info = p.get_device_info_by_index(i)
                if dev_info['maxInputChannels'] > 0:  # 只显示输入设备
                    print(f"设备 {i}: {dev_info['name']}")
                    print(f"    输入通道: {dev_info['maxInputChannels']}")
                    print(f"    采样率: {dev_info['defaultSampleRate']}")
            except Exception as e:
                print(f"获取设备 {i} 信息时发生错误: {e}")
        p.terminate()
        
    def start_monitoring(self, duration=None):
        """开始监测麦克风输入"""
        self.p = pyaudio.PyAudio()
        
        def callback(in_data, frame_count, time_info, status):
            # 将音频数据转换为numpy数组
            audio_data = np.frombuffer(in_data, dtype=np.float32)
            # 计算音量
            volume_norm = np.linalg.norm(audio_data) / len(audio_data)
            # 计算分贝值 (参考值为1.0)
            db = 20 * np.log10(volume_norm) if volume_norm > 0 else -100
            
            # 创建音量条
            bar_length = 50
            bar_count = int((volume_norm * 1000) * bar_length)
            bar_count = min(bar_count, bar_length)
            
            # 清除当前行并打印音量条
            print(f'\r音量: {"█" * bar_count + "░" * (bar_length - bar_count)} {db:.1f} dB   ', end='')
            
            return (in_data, pyaudio.paContinue)
        
        try:
            print("正在打开音频流...")
            self.stream = self.p.open(
                format=pyaudio.paFloat32,
                channels=self.channels,
                rate=self.rate,
                input=True,
                input_device_index=self.device_index,
                frames_per_buffer=self.chunk_size,
                stream_callback=callback
            )
            
            print("\n开始监测麦克风输入...")
            print("请对着麦克风说话，音量条会实时显示声音大小")
            print("按 Ctrl+C 停止监测\n")
            
            # 如果设置了持续时间，就等待指定时间
            if duration:
                time.sleep(duration)
            else:
                # 否则一直运行直到被中断
                while self.stream.is_active():
                    time.sleep(0.1)
                    
        except KeyboardInterrupt:
            print("\n\n停止监测")
        except Exception as e:
            print(f"\n打开音频设备时发生错误: {e}")
        finally:
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
            if self.p:
                self.p.terminate()

if __name__ == "__main__":
    # 创建麦克风测试器实例
    tester = MicrophoneTester()
    
    # 显示可用设备
    tester.list_devices()
    
    # 让用户选择设备
    try:
        device_index = int(input("\n请选择要测试的设备编号（直接回车使用默认设备）: ") or "-1")
        if device_index >= 0:
            tester = MicrophoneTester(device_index=device_index)
    except ValueError:
        print("使用默认设备")
        tester = MicrophoneTester()
    
    # 开始监测
    tester.start_monitoring()