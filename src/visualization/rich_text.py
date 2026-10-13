"""Compact, escaped Qt rich text shared by advice and assistant messages."""
import re
from html import escape

ACCENT = '#51C5E3'

def render_text(text):
    def inline(value):
        value=escape(value)
        return re.sub(r'\*\*(.+?)\*\*',r'<b style="color:'+ACCENT+r';">\1</b>',value)
    output=[]
    for raw in str(text).splitlines():
        line=raw.strip()
        if not line or re.fullmatch(r'[-*_]{2,}',line):continue
        heading=re.match(r'^(#{1,6})\s+(.+)',line)
        if heading:
            level=len(heading.group(1))
            size,color,spacing={1:('x-large','#EEF2F6',14),2:('large','#83D8EA',12)}.get(level,('medium','#71DFC2',8))
            title=escape(re.sub(r'\*\*(.+?)\*\*',r'\1',heading.group(2)))
            output.append(f'<p style="margin-top:{spacing}px;margin-bottom:6px;"><span style="font-size:{size};color:{color};font-weight:600;">{title}</span></p>')
            continue
        bullet=re.match(r'^[-*•]\s+(.+)',line)
        content=('• '+inline(bullet.group(1))) if bullet else inline(line)
        output.append('<p style="margin-top:0px;margin-bottom:6px;">'+content+'</p>')
    return ''.join(output)


def apply_reading_font(editor,size):
    """Override inherited/inline body sizes while retaining relative heading levels."""
    if getattr(editor,'_applying_reading_font',False):return
    from PyQt5.QtGui import QTextCursor,QTextCharFormat,QTextFormat,QFont
    editor._applying_reading_font=True
    try:
        document=editor.document()
        font=QFont(document.defaultFont());font.setPointSize(size);document.setDefaultFont(font)
        changes=[];block=document.begin()
        while block.isValid():
            iterator=block.begin()
            while not iterator.atEnd():
                fragment=iterator.fragment()
                if fragment.isValid():
                    fmt=fragment.charFormat()
                    adjustment=fmt.intProperty(QTextFormat.FontSizeAdjustment)
                    ratio={2:1.5,1:1.2}.get(adjustment,1.0)
                    target=size*ratio
                    if abs(fmt.fontPointSize()-target)>.01:changes.append((fragment.position(),fragment.length(),target))
                iterator+=1
            block=block.next()
        for position,length,target in changes:
            cursor=QTextCursor(document);cursor.setPosition(position);cursor.setPosition(position+length,QTextCursor.KeepAnchor)
            fmt=QTextCharFormat();fmt.setFontPointSize(target);cursor.mergeCharFormat(fmt)
    finally:editor._applying_reading_font=False
