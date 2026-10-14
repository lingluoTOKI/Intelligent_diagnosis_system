"""Double-click desktop launcher with a fixed GUI environment and per-run logs."""
from pathlib import Path
import os,sys,subprocess,datetime
ROOT=Path(__file__).resolve().parents[1]
def resolve_python():
    configured=os.environ.get('EYE_PYTHON')
    if configured:
        candidate=Path(configured).expanduser()
        if not candidate.is_file():raise RuntimeError('EYE_PYTHON points to a missing interpreter: '+str(candidate))
        return candidate
    for candidate in (ROOT/'.venv'/'Scripts'/'python.exe', ROOT/'venv'/'Scripts'/'python.exe',
                      ROOT/'.venv'/'bin'/'python', ROOT/'venv'/'bin'/'python',
                      Path(r'D:\Anaconda3\envs\yolov10\python.exe'), Path(sys.executable)):
        if candidate.is_file():return candidate
    raise RuntimeError('Python environment not found; install dependencies and set EYE_PYTHON')
def main():
    PYTHON=resolve_python()
    if not (ROOT/'visualization_test2.py').is_file():raise RuntimeError('Desktop entrypoint is missing: visualization_test2.py')
    env=dict(os.environ,PYTHONIOENCODING='utf-8',PYTHONUTF8='1')
    no_window = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
    probe=subprocess.run([str(PYTHON),'-c','import PyQt5.QtWidgets,cv2,numpy,torch; print("GUI environment OK")'],cwd=ROOT,env=env,capture_output=True,text=True,encoding='utf-8',creationflags=no_window)
    if probe.returncode:raise RuntimeError(probe.stderr)
    if '--check' in sys.argv or os.environ.get('EYE_LAUNCH_CHECK') == '1':print(probe.stdout);return
    folder=ROOT/'runtime'/'startup_logs';folder.mkdir(parents=True,exist_ok=True)
    log=folder/(datetime.datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+str(os.getpid())+'.log')
    print('Starting desktop. Log:',log,flush=True)
    with log.open('w',encoding='utf-8') as stream:
        result=subprocess.run([str(PYTHON),'-u',str(ROOT/'visualization_test2.py')],cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,creationflags=no_window)
    if result.returncode:raise RuntimeError('Application exited with code '+str(result.returncode)+'. See '+str(log))
if __name__=='__main__':
    try:main()
    except Exception as error:
        folder=ROOT/'runtime'/'startup_logs';folder.mkdir(parents=True,exist_ok=True)
        with (folder/'startup_errors.log').open('a',encoding='utf-8') as stream:
            stream.write(datetime.datetime.now().isoformat()+' '+str(error)+'\n')
        print(str(error));sys.exit(1)
