# 界面图形资源

本项目自行绘制眼睛与虹膜标识及线性操作图标，没有使用第三方图片或医学认证标识。

- `app.svg`：可缩放标识，供高 DPI 界面使用。
- `app.png`：256 像素预览。
- `app-windows.ico`：16、24、32、48、64、128、256 像素的 Windows DIB 图标，用于应用默认图标、窗口和启动快捷方式。
- `app.ico`：保留的早期 PNG 编码图标。

在项目目录运行 `powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/create_shortcut.ps1` 可重建带内嵌图标的单一启动程序。项目根目录的 `眼科图像分类辅助诊断系统.exe` 是启动入口。

应用图标在创建主窗口前设置，主窗口在首次显示前再次明确设置。修改后须退出旧窗口，再使用项目根目录的快捷方式启动。
