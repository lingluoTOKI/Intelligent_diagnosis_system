
# 添加版本信息
"""
Version: 1.0.0
Date: 2023-10-01
Description: Initial version with basic functionality.
"""


class MainWindow(QMainWindow):
    """主窗口类"""

    def __init__(self):
        super().__init__()
        
        # 自动备份代码
        self.backup_code()

        # 设置窗口属性
        self.setWindowTitle("AI眼科疾病智诊系统 v2.0")
        self.setGeometry(100, 100, 1400, 900)
        self.setMinimumSize(1200, 800)
        

    def backup_code(self):
        """自动备份代码"""
        import shutil
        import datetime

        # 获取当前脚本的路径
        script_path = os.path.abspath(__file__)
        
        # 备份目录
        backup_dir = os.path.join(os.path.dirname(script_path), "backups")
        os.makedirs(backup_dir, exist_ok=True)
        
        # 备份文件名（包含时间戳）
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        backup_file = os.path.join(backup_dir, f"backup_{timestamp}.py")
        
        # 复制当前脚本到备份目录
        shutil.copy(script_path, backup_file)
        print(f"代码已备份至: {backup_file}")
