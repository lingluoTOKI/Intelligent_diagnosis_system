"""Session-bound requests: worker snapshots never read live widgets."""
import threading
import re
from html import escape
from src.core.llm import DeepSeekAPI, AdviceResult
from src.core.classification import valid_prediction


def append_message(window,role,content):
    from PyQt5.QtGui import QTextCursor
    styles={
        'user':('您','#183E50','#83D8EA'),
        'assistant':('AI 助手','#263C38','#71DFC2'),
        'error':('请求失败','#48352B','#F1B761'),
    }
    label,background,accent=styles[role]
    text=escape(content)
    if role=='assistant':
        from .rich_text import render_text
        text=render_text(content)
    else:text=text.replace('\n','<br>')
    if not hasattr(window,'_message_texts'):window._message_texts={}
    import uuid
    identifier=uuid.uuid4().hex;window._message_texts[identifier]=content
    actions=''
    options=[('copy','复制')] if role=='user' else [('copy','复制'),('speak','朗读'),('stop','停止'),('export','导出文本')]
    for action,title in options:
        actions+=f'<a href="message:{action}:{identifier}" style="color:#83D8EA;">{title}</a>&nbsp;&nbsp;&nbsp;'
    text+='<p style="margin-top:10px;">'+actions+'</p>'
    # QTextDocument supports table cell backgrounds reliably, unlike CSS bubble layouts.
    html=(f'<table width="100%" cellspacing="0" cellpadding="12" bgcolor="{background}">'
          f'<tr><td><span style="color:{accent};"><b>{label}</b></span><br>'
          f'<span style="color:#EEF2F6;">{text}</span></td></tr></table>')
    cursor=QTextCursor(window.chat_display.document());cursor.movePosition(QTextCursor.End)
    cursor.insertHtml(html);cursor.movePosition(QTextCursor.End);cursor.insertBlock();cursor.insertBlock()
    window.chat_display.setTextCursor(cursor);window.chat_display.ensureCursorVisible()


def reset(window):
    window._chat_generation=getattr(window,"_chat_generation",0)+1
    window._chat_client=None
    window._chat_busy=False
    window.chat_history=[]
    window._message_texts={}
    if hasattr(window,"_speech_player"):window._speech_player.stop()
    window.chat_display.clear()
    window.send_chat_button.setEnabled(True)
    window.show_ai_progress(False)


def send(window, config):
    if getattr(window,"_chat_busy",False):return
    question=window.chat_input.toPlainText().strip()
    if not question:return
    if not window.use_api_checkbox.isChecked():
        window.show_message_box("在线问答未启用","请在系统设置中启用在线服务。");return
    key=window.api_key_input.text().strip()
    client=getattr(window,"_chat_client",None)
    if client is None or client.api_key!=key:
        client=DeepSeekAPI(key or None,config=config);window._chat_client=client
    generation=getattr(window,"_chat_generation",0)
    disease=getattr(window,"current_disease",None);confidence=getattr(window,"current_confidence",None)
    message=question
    if valid_prediction(disease,confidence):message+=f"\n[图像分类上下文，非临床诊断] {disease}，概率{confidence:.4f}"
    window._chat_busy=True;window.send_chat_button.setEnabled(False)
    window.chat_input.clear();window.show_ai_progress(True)
    window.ai_progress_bar.setRange(0,0)
    append_message(window,"user",question)
    def request():
        try:result=client.request_custom_advice(message)
        except Exception as error:result=AdviceResult(False,str(error),"error")
        try:window.chat_reply.emit((generation,question,result))
        except RuntimeError:pass
    threading.Thread(target=request,daemon=True).start()


def finish(window,payload):
    generation,question,result=payload
    if generation!=getattr(window,"_chat_generation",0) or getattr(window,"_closing",False):return
    window._chat_busy=False;window.send_chat_button.setEnabled(True)
    window.show_ai_progress(False);window.ai_progress_bar.setRange(0,100)
    append_message(window,"assistant" if result.success else "error",result.content)
    if result.success:
        window.chat_history.append({"question":question,"answer":result.content})
        if window.voice_chat_enabled.isChecked():
            # Retain the existing voice playback path without changing reply ownership.
            if hasattr(window,"speak_text"):window.speak_text(result.content)


from PyQt5.QtCore import QObject, QEvent, Qt

class ChatKeys(QObject):
    """Only the focused composer handles Enter; IME composition keeps its own keys."""
    def __init__(self,window):
        super().__init__(window.chat_input)
        self.window=window
        self.composing=False
    def eventFilter(self,obj,event):
        if event.type()==QEvent.InputMethod:
            self.composing=bool(event.preeditString())
        elif event.type()==QEvent.FocusOut:
            self.composing=False
        elif event.type()==QEvent.KeyPress and event.key() in (Qt.Key_Return,Qt.Key_Enter):
            if self.composing:return False
            if event.modifiers()==Qt.NoModifier:
                if not event.isAutoRepeat() and self.window.send_chat_button.isEnabled():
                    self.window.send_chat_button.click()
                return True
        return False


def install_chat_keys(window):
    window._message_actions=MessageActions(window)
    window.chat_display.viewport().installEventFilter(window._message_actions)
    window._chat_keys=ChatKeys(window)
    window.chat_input.installEventFilter(window._chat_keys)
    window.chat_input.setPlaceholderText('输入问题…  Enter 发送 · Shift+Enter 换行')
    window.send_chat_button.setToolTip('Enter 发送；Shift+Enter 换行')


class MessageActions(QObject):
    def __init__(self,window):super().__init__(window.chat_display);self.window=window
    def eventFilter(self,obj,event):
        if event.type()!=QEvent.MouseButtonRelease or event.button()!=Qt.LeftButton:return False
        link=self.window.chat_display.anchorAt(event.pos())
        if not link.startswith('message:'):return False
        from PyQt5.QtWidgets import QApplication,QFileDialog
        from pathlib import Path
        _,action,identifier=link.split(':',2)
        text=getattr(self.window,'_message_texts',{}).get(identifier)
        if text is None:return True
        if action=='copy':QApplication.clipboard().setText(text);self.window.status_bar.showMessage('消息已复制，可粘贴转发')
        elif action=='speak':self.window.speak_text(text)
        elif action=='stop':
            if hasattr(self.window,'_speech_player'):self.window._speech_player.stop()
        elif action=='export':
            path,_=QFileDialog.getSaveFileName(self.window,'导出消息','AI回复.txt','文本文件 (*.txt)')
            if path:
                try:Path(path).write_text(text,encoding='utf-8');self.window.status_bar.showMessage('消息已导出')
                except OSError as error:self.window.show_message_box('导出失败',str(error))
        return True
