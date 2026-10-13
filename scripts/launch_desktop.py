"""Double-click desktop launcher with a fixed GUI environment and per-run logs."""
from pathlib import Path
import os,sys,subprocess,datetime
ROOT=Path(__file__).resolve().parents[1]
PYTHON=Path(os.environ.get('EYE_PYTHON',r'D:\Anaconda3\envs\yolov10\python.exe'))
def main():
    if not PYTHON.is_file():raise RuntimeError('GUI Python not found: '+str(PYTHON))
    env=dict(os.environ,PYTHONIOENCODING='utf-8',PYTHONUTF8='1')
    probe=subprocess.run([str(PYTHON),'-c','import PyQt5.QtWidgets,cv2,numpy,torch; print("GUI environment OK")'],cwd=ROOT,env=env,capture_output=True,text=True,encoding='utf-8')
    if probe.returncode:raise RuntimeError(probe.stderr)
    if '--check' in sys.argv:print(probe.stdout);return
    folder=ROOT/'runtime'/'startup_logs';folder.mkdir(parents=True,exist_ok=True)
    log=folder/(datetime.datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+str(os.getpid())+'.log')
    print('Starting desktop. Log:',log,flush=True)
    with log.open('w',encoding='utf-8') as stream:
        result=subprocess.run([str(PYTHON),'-u',str(ROOT/'visualization_test2.py')],cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
    if result.returncode:raise RuntimeError('Application exited with code '+str(result.returncode)+'. See '+str(log))
if __name__=='__main__':
    try:main()
    except Exception as error:print(str(error));sys.exit(1)
