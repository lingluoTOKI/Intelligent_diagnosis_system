"""Windows user-bound credentials, with read-only legacy compatibility."""
import os,base64,ctypes
from ctypes import wintypes

class Blob(ctypes.Structure):
    _fields_=[('size',wintypes.DWORD),('data',ctypes.POINTER(ctypes.c_ubyte))]

def _crypt(data,encrypt):
    buffer=ctypes.create_string_buffer(data)
    source=Blob(len(data),ctypes.cast(buffer,ctypes.POINTER(ctypes.c_ubyte)));output=Blob()
    crypt=ctypes.WinDLL('crypt32',use_last_error=True)
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.LocalFree.argtypes=[ctypes.c_void_p];kernel.LocalFree.restype=ctypes.c_void_p
    if encrypt:ok=crypt.CryptProtectData(ctypes.byref(source),None,None,None,None,1,ctypes.byref(output))
    else:ok=crypt.CryptUnprotectData(ctypes.byref(source),None,None,None,None,1,ctypes.byref(output))
    if not ok:raise OSError('无法访问当前 Windows 用户的凭据保护服务')
    try:return ctypes.string_at(output.data,output.size)
    finally:kernel.LocalFree(output.data)

def encode_key(key):
    if os.name!='nt':raise OSError('此环境请通过 DEEPSEEK_API_KEY 提供密钥，不保存到文件')
    return 'dpapi:'+base64.b64encode(_crypt(key.encode('utf-8'),True)).decode('ascii')

def decode_key(value):
    if value.startswith('dpapi:'):
        if os.name!='nt':raise OSError('此密钥仅能由原 Windows 用户解密')
        return _crypt(base64.b64decode(value[6:]),False).decode('utf-8')
    try:return base64.b64decode(value,validate=True).decode('utf-8')
    except (ValueError,UnicodeError):return value
