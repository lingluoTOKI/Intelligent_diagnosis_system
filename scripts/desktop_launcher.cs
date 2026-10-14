using System;
using System.Diagnostics;
using System.IO;
using System.Windows.Forms;
internal static class Launcher {
  [STAThread] static void Main() {
    string root=AppDomain.CurrentDomain.BaseDirectory;
    string script=Path.Combine(root,"scripts","launch_desktop.py");
    string python=Environment.GetEnvironmentVariable("EYE_PYTHON");
    if (String.IsNullOrWhiteSpace(python)) {
      string[] candidates={Path.Combine(root,@".venv\Scripts\python.exe"),
        Path.Combine(root,@"venv\Scripts\python.exe"), @"D:\Anaconda3\envs\yolov10\python.exe"};
      foreach(string candidate in candidates) if(File.Exists(candidate)) { python=candidate; break; }
      if(String.IsNullOrWhiteSpace(python)) {
        foreach(string directory in (Environment.GetEnvironmentVariable("PATH") ?? "").Split(Path.PathSeparator)) {
          if(String.IsNullOrWhiteSpace(directory)) continue;
          string candidate=Path.Combine(directory.Trim('"'),"python.exe");
          if(File.Exists(candidate)) { python=candidate; break; }
        }
      }
    }
    if (!File.Exists(script) || !File.Exists(python)) { MessageBox.Show("找不到启动脚本或 Python 环境，请检查安装路径。","眼科图像分类辅助诊断系统"); return; }
    try {
      Process.Start(new ProcessStartInfo {
        FileName=python,
        Arguments="\""+script+"\""+(Array.IndexOf(Environment.GetCommandLineArgs(),"--check")>=0 ? " --check" : ""),
        WorkingDirectory=root,
        UseShellExecute=false,
        CreateNoWindow=true,
        WindowStyle=ProcessWindowStyle.Hidden
      });
    } catch(Exception error) { MessageBox.Show(error.Message,"启动失败"); }
  }
}
