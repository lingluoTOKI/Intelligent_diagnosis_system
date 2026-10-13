from PyQt5.QtCore import QByteArray
from .image_directory import read,save

def restore(window):
    preferences=read(window)
    window.reading_size_combo.setCurrentIndex(max(0,window.reading_size_combo.findData(preferences.get('reading_size',14))))
    window.voice_chat_enabled.setChecked(bool(preferences.get('auto_read',window.voice_chat_enabled.isChecked())))
    if preferences.get('window_geometry'):
        window.restoreGeometry(QByteArray.fromHex(preferences['window_geometry'].encode()))
    if preferences.get('splitter_state'):
        window.workbench_splitter.restoreState(QByteArray.fromHex(preferences['splitter_state'].encode()))
    window.reading_size_combo.currentIndexChanged.connect(lambda:_save_controls(window))
    window.voice_chat_enabled.toggled.connect(lambda:_save_controls(window))

def _save_controls(window):
    try:save(window,reading_size=window.reading_size_combo.currentData(),auto_read=window.voice_chat_enabled.isChecked())
    except OSError:window.status_bar.showMessage('偏好保存失败，请检查配置目录权限')

def save_window(window):
    try:save(window,window_geometry=bytes(window.saveGeometry().toHex()).decode(),splitter_state=bytes(window.workbench_splitter.saveState().toHex()).decode())
    except OSError:window.status_bar.showMessage('窗口布局保存失败')

def reset(window):
    save(window,font_family='Microsoft YaHei',font_size=14,reading_size=14,auto_read=False,window_geometry='',splitter_state='')
    window.font_family_combo.setCurrentIndex(max(0,window.font_family_combo.findData('Microsoft YaHei')))
    window.font_size_combo.setCurrentIndex(window.font_size_combo.findData(14))
    window.reading_size_combo.setCurrentIndex(window.reading_size_combo.findData(14))
    window.voice_chat_enabled.setChecked(False);window.workbench_splitter.setSizes([400,900]);window.showMaximized()
