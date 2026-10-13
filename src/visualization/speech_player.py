"""Fast in-process speech playback using cached, sentence-sized MP3 segments."""
import hashlib
import importlib.util
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import time

from PyQt5.QtCore import QObject, QProcess, QTimer, QUrl


def split_for_speech(text, target=110, maximum=180):
    """Split at natural pauses so the first MP3 can start quickly."""
    clean = re.sub(r'<[^>]+>|[*#`]', '', text)
    clean = re.sub(r'\s+', ' ', clean).strip()
    if not clean:
        return []
    sentences = [part.strip() for part in re.split(r'(?<=[。！？；!?;])', clean) if part.strip()]
    chunks, current = [], ''
    for sentence in sentences:
        while len(sentence) > maximum:
            cut = max(sentence.rfind(mark, 0, maximum) for mark in '，、,:：') + 1
            if cut < target // 2:
                cut = maximum
            piece, sentence = sentence[:cut].strip(), sentence[cut:].strip()
            if current:
                chunks.append(current)
                current = ''
            if piece:
                chunks.append(piece)
        if current and len(current) + len(sentence) > target:
            chunks.append(current)
            current = sentence
        else:
            current += sentence
    if current:
        chunks.append(current)
    return chunks


class SpeechPlayer(QObject):
    def __init__(self, window):
        super().__init__(window)
        from PyQt5.QtMultimedia import QMediaPlayer
        self.window = window
        self.player = QMediaPlayer(self)
        self.process = None
        self.parts = []
        self.play_index = -1
        self.waiting_index = None
        self.session = 0
        self.temp_paths = set()
        self.player.error.connect(lambda _: window.status_bar.showMessage('语音播放失败：' + self.player.errorString()))
        self.player.mediaStatusChanged.connect(self._media_status_changed)

    def stop(self):
        from PyQt5.QtMultimedia import QMediaContent
        self.session += 1
        if self.process:
            process, self.process = self.process, None
            process.kill()
            process.waitForFinished(1000)
            process.deleteLater()
        self.player.stop()
        self.player.setMedia(QMediaContent())
        for path in tuple(self.temp_paths):
            try:
                os.remove(path)
            except OSError:
                pass
        self.temp_paths.clear()
        self.parts = []
        self.play_index = -1
        self.waiting_index = None

    def speak(self, text):
        self.stop()
        chunks = split_for_speech(text)
        if not chunks:
            return
        executable = shutil.which('edge-tts')
        if executable:
            program, prefix = executable, []
        elif importlib.util.find_spec('edge_tts') is not None:
            program, prefix = sys.executable, ['-m', 'edge_tts']
        else:
            self.window.status_bar.showMessage('未找到原来的 edge-tts 程序，请检查运行环境；未切换离线音色')
            return
        cache = Path(__file__).resolve().parents[2] / 'runtime' / 'speech_cache'
        cache.mkdir(parents=True, exist_ok=True)
        self.program, self.prefix = program, prefix
        self.parts = []
        for chunk in chunks:
            digest = hashlib.sha256(('zh-CN-XiaoxiaoNeural|+10%|' + chunk).encode('utf-8')).hexdigest()
            path = cache / (digest + '.mp3')
            self.parts.append({'text': chunk, 'path': str(path), 'ready': path.exists() and path.stat().st_size > 0})
        current_session = self.session
        if self.parts[0]['ready']:
            self._play(0)
        self._generate_next(current_session)

    def _generate_next(self, current_session):
        if current_session != self.session or self.process:
            return
        index = next((i for i, part in enumerate(self.parts) if not part['ready']), None)
        if index is None:
            if self.play_index < 0 and self.parts:
                self._play(0)
            return
        fd, temporary = tempfile.mkstemp(suffix='.mp3')
        os.close(fd)
        self.temp_paths.add(temporary)
        process = QProcess(self)
        self.process = process
        started = time.monotonic()
        timed_out = [False]

        def finished(code, _status):
            if current_session != self.session or self.process is not process:
                return
            self.process = None
            process.deleteLater()
            valid = code == 0 and os.path.exists(temporary) and os.path.getsize(temporary) > 0
            if valid:
                try:
                    os.replace(temporary, self.parts[index]['path'])
                    self.temp_paths.discard(temporary)
                    self.parts[index]['ready'] = True
                except OSError:
                    self.parts[index]['path'] = temporary
                    self.parts[index]['ready'] = True
                if index == 0 and self.play_index < 0:
                    self._play(0)
                elif self.waiting_index == index:
                    self._play(index)
                self._generate_next(current_session)
            else:
                message = '语音生成超过 20 秒，请稍后重试或检查网络' if timed_out[0] else '晓晓语音生成失败，请检查网络后重试'
                self.window.status_bar.showMessage(message)

        process.finished.connect(finished)
        timeout = QTimer(process)
        timeout.setSingleShot(True)
        timeout.setInterval(20000)
        def kill_on_timeout():
            if self.process is process:
                timed_out[0] = True
                process.kill()
        timeout.timeout.connect(kill_on_timeout)
        process.finished.connect(timeout.stop)
        timeout.start()
        progress = QTimer(process)
        progress.setInterval(1000)
        progress.timeout.connect(lambda: self.window.status_bar.showMessage(
            '正在生成第 %d/%d 段 MP3（%d 秒），首段完成即播放' %
            (index + 1, len(self.parts), int(time.monotonic() - started))) if self.process is process else None)
        process.finished.connect(progress.stop)
        progress.start()
        process.errorOccurred.connect(lambda _: self.window.status_bar.showMessage('无法启动语音生成，请检查运行环境'))
        arguments = self.prefix + ['--voice', 'zh-CN-XiaoxiaoNeural', '--rate=+10%', '--text', self.parts[index]['text'], '--write-media', temporary]
        process.start(self.program, arguments)
        self.window.status_bar.showMessage('正在生成首段 MP3，完成后立即播放…')

    def _play(self, index):
        from PyQt5.QtMultimedia import QMediaContent
        if index >= len(self.parts) or not self.parts[index]['ready']:
            self.waiting_index = index
            self.window.status_bar.showMessage('正在准备下一段语音…')
            return
        self.waiting_index = None
        self.play_index = index
        self.player.setMedia(QMediaContent(QUrl.fromLocalFile(self.parts[index]['path'])))
        self.player.play()
        cached = all(part['ready'] for part in self.parts)
        suffix = '（已缓存）' if cached else '，后续语音正在后台生成'
        self.window.status_bar.showMessage('正在应用内朗读' + suffix + '；可点击“停止”')

    def _media_status_changed(self, status):
        from PyQt5.QtMultimedia import QMediaPlayer
        if status != QMediaPlayer.EndOfMedia or self.play_index < 0:
            return
        next_index = self.play_index + 1
        if next_index < len(self.parts):
            self._play(next_index)
        else:
            self.window.status_bar.showMessage('朗读完成')

    def speak_offline(self, text):
        self.stop()
        if importlib.util.find_spec('pyttsx3') is None:
            self.window.status_bar.showMessage('语音不可用：当前环境缺少 edge-tts 和 pyttsx3')
            return
        process = QProcess(self)
        self.process = process
        script = "import sys,pyttsx3; text=sys.stdin.buffer.read().decode('utf-8'); e=pyttsx3.init(); voices=e.getProperty('voices'); v=next((v for v in voices if 'chinese' in v.name.lower() or 'zh' in v.id.lower()),None); e.setProperty('voice',v.id) if v else None; e.setProperty('rate',160); e.say(text); e.runAndWait()"
        def started():
            process.write(text.encode('utf-8'))
            process.closeWriteChannel()
            self.window.status_bar.showMessage('正在离线朗读，可点击消息下方“停止”')
        def finished(code, _status):
            if self.process is not process:
                return
            self.process = None
            process.deleteLater()
            self.window.status_bar.showMessage('朗读完成' if code == 0 else '离线朗读失败，请检查 Windows 语音引擎与音频输出')
        process.started.connect(started)
        process.finished.connect(finished)
        process.errorOccurred.connect(lambda _: self.window.status_bar.showMessage('无法启动离线语音进程'))
        process.start(sys.executable, ['-c', script])
