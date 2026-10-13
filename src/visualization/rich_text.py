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


def advice_guide(classified=False):
    """Instructions shown only before a report has been generated."""
    if classified:
        text = """## 分类已完成，接下来可以
1. 点击 **生成辅助建议**，查看当前图像的结果说明。
2. 切换到 **多轮问答**，补充情况或继续提问。
3. 使用 **阅读字号** 调整文字，或点击 **展开阅读** 查看长内容。
在线建议需要在「系统设置」中配置 AI 服务。
"""
    else:
        text = """## 开始使用
1. **导入图像**：点击左侧「加载图像」，也可拖入图片或使用摄像头采集。
2. **开始分类**：确认模型已就绪，点击「开始分类」；未就绪时先点击「加载模型」。
3. **查看建议**：分类完成后，点击「生成辅助建议」。
4. **继续交流**：切换到「多轮问答」，输入问题并发送。
可用右上角「阅读字号」调整文字大小，点击「展开阅读」查看长内容。
"""
    return render_text(text)
