"""Own only this demo's processes; preserve append-only logs and all artifacts."""
import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / '.runtime'


def alive(entry):
    try:
        stat = Path(f"/proc/{entry['pid']}/stat").read_text().split()
        return stat[2] != 'Z' and stat[21] == entry['birth']
    except (OSError, KeyError):
        return False


def spawn(name, args):
    path = STATE / f'{name}.json'
    if path.exists() and alive(json.loads(path.read_text())):
        return
    with (STATE / f'{name}.log').open('ab') as log:
        child = subprocess.Popen(args, cwd=ROOT, stdout=log, stderr=log,
                                 stdin=subprocess.DEVNULL, start_new_session=True)
    birth = Path(f'/proc/{child.pid}/stat').read_text().split()[21]
    path.write_text(json.dumps({'pid': child.pid, 'birth': birth, 'args': args}))
    time.sleep(.3)
    if child.poll() is not None:
        raise RuntimeError(f'{name} exited; see {STATE / (name + ".log")}')


def stop(name):
    path = STATE / f'{name}.json'
    if not path.exists():
        return
    entry = json.loads(path.read_text())
    if alive(entry):
        os.killpg(entry['pid'], signal.SIGTERM)
        for _ in range(50):
            if not alive(entry):
                break
            time.sleep(.1)
        if alive(entry):
            os.killpg(entry['pid'], signal.SIGKILL)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['start', 'stop', 'status'])
    parser.add_argument('--dashboard', action='store_true')
    args = parser.parse_args()
    STATE.mkdir(exist_ok=True)
    # Serialize concurrent start/stop invocations on Linux.
    import fcntl
    with (STATE / 'control.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if args.action == 'stop':
            for name in ['dashboard', 'cam1', 'cam2', 'cam3', 'cam4', 'mediamtx']:
                stop(name)
            print('Demo processes stopped.')
        elif args.action == 'status':
            for path in sorted(STATE.glob('*.json')):
                print(path.stem, 'running' if alive(json.loads(path.read_text())) else 'stopped')
        else:
            media = Path(os.environ.get('MEDIAMTX_BIN', Path.home() / 'mediamtx/mediamtx'))
            for i in range(1, 5):
                if not (ROOT / f'sample_videos/cam{i}.mp4').is_file():
                    raise RuntimeError(f'Missing cam{i}.mp4')
            spawn('mediamtx', [str(media), str(ROOT / 'config/mediamtx-demo.yml')])
            for _ in range(30):
                try:
                    with socket.create_connection(('127.0.0.1', 8554), timeout=.2):
                        break
                except OSError:
                    time.sleep(.1)
            else:
                raise RuntimeError('MediaMTX did not open port 8554')
            for i in range(1, 5):
                name = f'cam{i}'
                spawn(name, ['ffmpeg', '-nostdin', '-hide_banner', '-loglevel', 'warning',
                             '-re', '-stream_loop', '-1', '-i', str(ROOT / f'sample_videos/{name}.mp4'),
                             '-an', '-c:v', 'libx264', '-preset', 'ultrafast', '-tune', 'zerolatency',
                             '-threads', '1', '-pix_fmt', 'yuv420p', '-g', '30', '-keyint_min', '30',
                             '-sc_threshold', '0', '-bf', '0', '-f', 'rtsp', '-rtsp_transport', 'tcp',
                             f'rtsp://127.0.0.1:8554/{name}'])
            if args.dashboard:
                spawn('dashboard', [sys.executable, '-m', 'streamlit', 'run', 'app.py',
                                    '--server.port', '8501', '--server.address', '0.0.0.0'])
            print('Four looping demo RTSP publishers started.' +
                  (' Dashboard: http://localhost:8501' if args.dashboard else ''))


if __name__ == '__main__':
    main()
