"""Consistent graphite, blue and teal theme with persistent, live typography sizing."""
from PyQt5.QtGui import QFont, QPalette, QColor
from PyQt5.QtWidgets import QWidget, QTableWidget, QLineEdit, QTextEdit

STYLE = """
QLabel#notice { color: #F1B761; font-size: SMALLpt; padding-top: 6px; }
QMainWindow, QWidget { background: #171C24; color: #EEF2F6; font-family: 'Microsoft YaHei'; font-size: BODYpt; font-weight: normal; }
QLabel { background: transparent; }
QLabel#pageTitle { font-size: TITLEpt; font-weight: bold; color: #F4F7FA; }
QLabel#sectionTitle { font-size: SECTIONpt; font-weight: bold; color: #83D8EA; }
QLabel#muted, QLabel#probName { font-size: SMALLpt; color: #ADB8C7; }
QLabel#prediction { font-size: TITLEpt; font-weight: bold; }
QLabel#score { font-size: SCOREpt; font-weight: bold; color: #4DD6B4; }
QLabel#badge { color: #83D8EA; background: #123145; border: 1px solid #40536B; padding: 8px 14px; border-radius: 8px; font-size: BODYpt; }
QFrame#card { background: #242C37; border: 1px solid #394451; border-radius: 10px; }
QPushButton { background: #303B49; border: 1px solid #465362; border-radius: 7px; padding: 8px 12px; color: #EEF2F6; }
QPushButton:hover { border-color: #4DD6B4; background: #394958; }
QPushButton:pressed { background: #202A33; }
QPushButton:disabled { color: #8693A3; border-color: #394451; background: #252D37; }
QPushButton[primary='true'] { background: #4DD6B4; border-color: #4DD6B4; color: #102A24; font-weight: bold; }
QPushButton[primary='true']:hover { background: #72E5C8; }
QPushButton[primary='true']:disabled { background: #252D37; color: #8693A3; border-color: #394451; }
QTextEdit, QLineEdit, QComboBox { background: #1D242D; color: #EEF2F6; border: 1px solid #465362; border-radius: 7px; padding: 8px; selection-background-color: #226F98; selection-color: #FFFFFF; }
QLineEdit:disabled, QTextEdit:disabled { color: #A5B2C2; background: #252D37; }
QTextEdit:focus, QLineEdit:focus { border-color: #4DD6B4; }
QComboBox QAbstractItemView { background: #242C37; selection-background-color: #226F98; selection-color: #FFFFFF; }
QTabWidget::pane { border: 1px solid #394451; border-radius: 8px; background: #242C37; }
QTabBar::tab { padding: 10px 16px; color: #ADB8C7; border-bottom: 3px solid transparent; background: #171C24; }
QTabBar::tab:selected { color: #83D8EA; background: #242C37; border-bottom-color: #51C5E3; }
QTabBar::tab:hover { color: white; }
QGroupBox { border: 1px solid #465362; border-radius: 8px; margin-top: 30px; padding: 20px 14px 14px; font-weight: bold; }
QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; left: 16px; padding: 0 6px; color: #83D8EA; }
QCheckBox { spacing: 8px; }
QCheckBox::indicator { width: 18px; height: 18px; }
QProgressBar { background: #394451; border: none; border-radius: 3px; }
QProgressBar::chunk { background: #4DD6B4; border-radius: 3px; }
QSlider::groove:horizontal { height: 6px; background: #394451; }
QSlider::handle:horizontal { background: #51C5E3; width: 16px; margin: -6px 0; border-radius: 4px; }
QTableWidget { background: #242C37; alternate-background-color: #2A3440; gridline-color: #394451; selection-background-color: #226F98; selection-color: #FFFFFF; }
QHeaderView::section { background: #303B49; color: #EEF2F6; padding: 8px; border: none; }
QMenuBar, QMenu { background: #242C37; }
QMenuBar::item:selected, QMenu::item:selected { background: #35485E; }
QStatusBar { background: #242C37; border-top: 1px solid #394451; padding: 4px 10px; font-size: SMALLpt; }
QSplitter::handle { background: #171C24; }
QSplitter::handle:hover { background: #51C5E3; }
QScrollBar:vertical { background: #242C37; width: 10px; }
QScrollBar::handle:vertical { background: #556477; min-height: 24px; border-radius: 4px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: transparent; }
QScrollArea { border: none; }
QToolTip { background: #35485E; color: white; padding: 6px; border: 1px solid #556477; }
"""


def stylesheet(size, family="DengXian"):
    values={"BODY":size,"SMALL":max(12,size-2),"TITLE":size+6,"SECTION":size+1,"SCORE":size+10}
    css=STYLE.replace("Microsoft YaHei",family)
    for key,value in values.items():css=css.replace(key+"pt",str(value)+"pt")
    return css


def style_dialog(dialog,window):
    size=window.font_size_combo.currentData() or 14
    for child in dialog.findChildren(QWidget):child.setStyleSheet("")
    family=window.font_family_combo.currentData()
    dialog.setStyleSheet(stylesheet(size,family))
    apply_input_palette(dialog)
    for table in dialog.findChildren(QTableWidget):
        table.verticalHeader().setMinimumSectionSize(round(size*2.4))
        table.verticalHeader().setDefaultSectionSize(round(size*2.6))


def apply_workbench_theme(window):
    size=window.font_size_combo.currentData() or 14
    # Remove per-widget legacy overrides that caused clipped titles and mixed button sizes.
    for child in window.centralWidget().findChildren(QWidget):child.setStyleSheet("")
    family=window.font_family_combo.currentData()
    window.setStyleSheet(stylesheet(size,family))
    window.status_bar.setStyleSheet("")
    window.model_info_label.setStyleSheet("")
    window.network_info_label.setStyleSheet("")
    for name in ("detect_button","advice_button","send_chat_button"):
        button=getattr(window,name);button.setProperty("primary",True)
        button.style().unpolish(button);button.style().polish(button)
    window.workbench_splitter.setMinimumHeight(round((560 if window.probability_toggle.isChecked() else 400)*size/14))
    window.probability_details.setMinimumHeight(0)
    window.probability_details.layout().invalidate()
    window.original_image_label.setFont(QFont(family,size))
    for text in (window.advice_text,window.chat_display,window.chat_input):
        text.setFont(QFont(family,size))
        text.document().setDefaultFont(QFont(family,size))
    # Override old rich-text font runs too, so saved advice follows the preference.
    from PyQt5.QtGui import QTextCursor, QTextCharFormat
    for text in (window.advice_text,window.chat_display):
        cursor=QTextCursor(text.document());cursor.select(QTextCursor.Document)
        style=QTextCharFormat();style.setFontFamily(family)
        cursor.mergeCharFormat(style)
    for name,bar,score in window.probability_rows:
        score.setMinimumWidth(round(66*size/14))
    window.font_size_combo.setMinimumWidth(round(145*size/14))
    toolbar_height=window.font_size_combo.fontMetrics().height()+22
    for widget in (window.font_family_combo,window.font_size_combo,window.runtime_badge,window.font_size_label):
        widget.setFont(QFont(family,size))
        widget.setFixedHeight(toolbar_height)
    window.font_family_combo.setMinimumWidth(window.font_family_combo.fontMetrics().horizontalAdvance(window.font_family_combo.currentText())+46)

    for name in ("batch_page","history_page"):
        page=getattr(window,name,None)
        if page is not None:page.adapt_typography()

    window.probability_details.setStyleSheet("background: transparent;")

    apply_input_palette(window)
    if hasattr(window,"apply_reading_size"):window.apply_reading_size()


def apply_input_palette(root):
    """Qt 5 placeholder text needs an explicit palette role on dark themes."""
    for editor in root.findChildren(QLineEdit) + root.findChildren(QTextEdit):
        editor.ensurePolished()
        palette = editor.palette()
        for group in (QPalette.Active, QPalette.Inactive, QPalette.Disabled):
            palette.setColor(group, QPalette.PlaceholderText, QColor("#A5B2C2"))
            palette.setColor(group, QPalette.Text, QColor("#A5B2C2" if group == QPalette.Disabled else "#EEF2F6"))
        editor.setPalette(palette)
