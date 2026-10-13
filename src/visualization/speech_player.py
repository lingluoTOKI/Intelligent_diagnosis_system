"""In-process playback with asynchronous synthesis; never launches an external player."""
import sys,re,tempfile,os,importlib.util,shutil
from PyQt5.QtCore import QObject,QProcess,QUrl,QTimer

class SpeechPlayer(QObject):
    def __init__(self,window):
        super().__init__(window)
        from PyQt5.QtMultimedia import QMediaPlayer,QMediaContent
        self.window=window;self.player=QMediaPlayer(self);self.process=None;self.path=None
        self.player.error.connect(lambda _:window.status_bar.showMessage('语音播放失败：'+self.player.errorString()))
    def stop(self):
        from PyQt5.QtMultimedia import QMediaContent
        if self.process:
            process=self.process;self.process=None
            process.kill();process.waitForFinished(1000);process.deleteLater()
        self.player.stop();self.player.setMedia(QMediaContent())
        if self.path:
            try:os.remove(self.path)
            except OSError:pass
            self.path=None
    def speak(self,text):
        from PyQt5.QtMultimedia import QMediaContent
        self.stop()
        clean=re.sub(r'<[^>]+>|[*#`]', '',text).strip()
        if not clean:return
        executable=shutil.which('edge-tts')
        if executable:
            program,arguments=executable,[]
        elif importlib.util.find_spec('edge_tts') is not None:
            program,arguments=sys.executable,['-m','edge_tts']
        else:
            self.window.status_bar.showMessage('未找到原来的 edge-tts 程序，请检查运行环境；未切换离线音色');return
        fd,self.path=tempfile.mkstemp(suffix='.mp3');os.close(fd)
        process=QProcess(self);self.process=process;path=self.path
        def ready(code,status):
            if self.process is not process:return
            self.process=None;process.deleteLater()
            if code!=0:
                self.window.status_bar.showMessage('晓晓语音生成失败，请检查网络后重试；未切换离线音色');return
            self.player.setMedia(QMediaContent(QUrl.fromLocalFile(path)));self.player.play()
            self.window.status_bar.showMessage('正在应用内朗读，可点击消息下方“停止”')
        process.finished.connect(ready)
        timer=QTimer(process);timer.setSingleShot(True);timer.setInterval(20000)
        timer.timeout.connect(lambda:process.kill() if self.process is process else None)
        process.finished.connect(timer.stop);timer.start()
        process.errorOccurred.connect(lambda _:self.window.status_bar.showMessage('无法启动语音生成，请检查运行环境'))
        process.start(program,arguments+['--voice','zh-CN-XiaoxiaoNeural','--rate=+10%','--text',clean,'--write-media',path])
        self.window.status_bar.showMessage('正在生成语音…')

    def speak_offline(self,text):
        self.stop()
        if importlib.util.find_spec('pyttsx3') is None:
            self.window.status_bar.showMessage('语音不可用：当前环境缺少 edge-tts 和 pyttsx3');return
        process=QProcess(self);self.process=process
        script="import sys,pyttsx3; text=sys.stdin.buffer.read().decode('utf-8'); e=pyttsx3.init(); voices=e.getProperty('voices'); v=next((v for v in voices if 'chinese' in v.name.lower() or 'zh' in v.id.lower()),None); e.setProperty('voice',v.id) if v else None; e.setProperty('rate',160); e.say(text); e.runAndWait()"
        def started():
            process.write(text.encode('utf-8'));process.closeWriteChannel()
            self.window.status_bar.showMessage('正在离线朗读，可点击消息下方“停止”')
        def finished(code,status):
            if self.process is not process:return
            self.process=None;process.deleteLater()
            self.window.status_bar.showMessage('朗读完成' if code==0 else '离线朗读失败，请检查 Windows 语音引擎与音频输出')
        process.started.connect(started);process.finished.connect(finished)
        process.errorOccurred.connect(lambda _:self.window.status_bar.showMessage('无法启动离线语音进程'))
        process.start(sys.executable,['-c',script])
