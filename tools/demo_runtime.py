"""Linux demo lifecycle: preflight first, track owned processes, verify RTSP paths."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / '.runtime'
NAMES = ('mediamtx', 'cam1', 'cam2', 'cam3', 'cam4', 'dashboard')


def read_entry(name):
    try:
        return json.loads((STATE / f'{name}.json').read_text())
    except (OSError, ValueError):
        return {}


def alive(entry):
    try:
        if entry.get('boot_id') and entry['boot_id'] != Path('/proc/sys/kernel/random/boot_id').read_text().strip():
            return False
        # comm can contain spaces; fields after the closing parenthesis start at field 3.
        tail = Path(f"/proc/{entry['pid']}/stat").read_text().rsplit(')', 1)[1].split()
        return tail[0] != 'Z' and tail[19] == entry['birth']
    except (OSError, KeyError, IndexError):
        return False


def spawn(name, args):
    entry = read_entry(name)
    if alive(entry):
        if entry.get('args') != args:
            raise RuntimeError(f'{name} already runs with different arguments. Stop this demo before changing configuration.')
        return False
    with (STATE / f'{name}.log').open('ab') as log:
        child = subprocess.Popen(args, cwd=ROOT, stdout=log, stderr=log,
                                 stdin=subprocess.DEVNULL, start_new_session=True)
    try:
        tail = Path(f'/proc/{child.pid}/stat').read_text().rsplit(')', 1)[1].split()
        entry = {'pid': child.pid, 'birth': tail[19], 'args': args,
                 'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip()}
        (STATE / f'{name}.json').write_text(json.dumps(entry))
    except OSError:
        # Child may have exited immediately; do not leave an untracked running child.
        if child.poll() is None:
            child.terminate()
        raise RuntimeError(f'{name} could not be tracked; check .runtime/{name}.log') from None
    time.sleep(.3)
    if child.poll() is not None:
        raise RuntimeError(f'{name} exited; check .runtime/{name}.log')
    return True


def stop(name):
    entry = read_entry(name)
    if alive(entry):
        try:
            os.killpg(entry['pid'], signal.SIGTERM)
            for _ in range(50):
                if not alive(entry):
                    break
                time.sleep(.1)
            if alive(entry):
                os.killpg(entry['pid'], signal.SIGKILL)
        except ProcessLookupError:
            pass


def port_open(port):
    try:
        with socket.create_connection(('127.0.0.1', port), timeout=.3):
            return True
    except OSError:
        return False


def require_executable(name, override=None):
    resolved = shutil.which(str(override)) if override else shutil.which(name)
    if not resolved:
        raise RuntimeError(f'{name} executable not found. Install it in WSL or set the documented binary override.')
    return resolved


def publisher_args(ffmpeg, video, camera, codec):
    args = [ffmpeg, '-nostdin', '-hide_banner', '-loglevel', 'warning',
            '-re', '-stream_loop', '-1', '-i', str(video), '-map', '0:v:0', '-an']
    if codec == 'h264':
        args += ['-c:v', 'copy']
    else:
        args += ['-c:v', 'libx264', '-preset', 'ultrafast', '-tune', 'zerolatency',
                 '-threads', '1', '-pix_fmt', 'yuv420p', '-g', '30', '-keyint_min', '30',
                 '-sc_threshold', '0', '-bf', '0']
    return args + ['-f', 'rtsp', '-rtsp_transport', 'tcp', f'rtsp://127.0.0.1:8554/{camera}']


def video_codec(ffprobe, source):
    try:
        result = subprocess.run([ffprobe, '-v', 'error', '-select_streams', 'v:0',
                                 '-show_entries', 'stream=codec_name', '-of', 'json', str(source)],
                                capture_output=True, text=True, timeout=5)
        streams = json.loads(result.stdout).get('streams', []) if not result.returncode else []
        if not streams:
            raise ValueError('no video')
        return streams[0]['codec_name']
    except (ValueError, KeyError, subprocess.TimeoutExpired):
        raise RuntimeError(f'{source.name}: no readable video track. Check the source with ffprobe.') from None


def validate_dashboard_sources():
    """The managed demo owns local paths; do not launch a dashboard pointing elsewhere."""
    config_path = Path(os.environ.get('VLM_EDGE_CAMERAS_CONFIG', ROOT / 'config/cameras.json'))
    try:
        cameras = json.loads(config_path.read_text(encoding='utf-8-sig'))['cameras']
        if not isinstance(cameras, dict) or set(cameras) != {f'cam{i}' for i in range(1, 5)}:
            raise ValueError('wrong camera set')
        for name, config in cameras.items():
            allowed = {f'rtsp://127.0.0.1:8554/{name}', f'rtsp://localhost:8554/{name}'}
            if config['url'] not in allowed or config.get('source_kind') != 'demo_rtsp':
                raise ValueError('not a managed demo endpoint')
    except (OSError, ValueError, KeyError, TypeError):
        raise RuntimeError('Dashboard camera configuration does not match this managed local demo. Use config/cameras.json, or start external sources and the dashboard separately.') from None


def preflight(dashboard=False):
    if sys.platform != 'linux':
        raise RuntimeError('Demo lifecycle requires WSL/Linux. Run bash start_demo.sh in the WSL project, not Windows Python.')
    media_override = os.environ.get('MEDIAMTX_BIN')
    if not media_override and (Path.home() / 'mediamtx/mediamtx').is_file():
        media_override = Path.home() / 'mediamtx/mediamtx'
    media = require_executable('mediamtx', media_override)
    ffmpeg = require_executable('ffmpeg', os.environ.get('FFMPEG_BIN'))
    ffprobe = require_executable('ffprobe', os.environ.get('FFPROBE_BIN'))
    if dashboard and importlib.util.find_spec('streamlit') is None:
        raise RuntimeError('Streamlit is missing from this Python environment. Activate the intended project environment.')
    if dashboard:
        validate_dashboard_sources()
    plans = {}
    for i in range(1, 5):
        name = f'cam{i}'
        video = ROOT / 'sample_videos' / f'{name}.mp4'
        if not video.is_file():
            raise RuntimeError(f'Missing sample_videos/{name}.mp4. Supply permitted local videos; no download is automatic.')
        plans[name] = publisher_args(ffmpeg, video, name, video_codec(ffprobe, video))
    # All checks happen before spawning. Never borrow an unrelated listener's readiness.
    if not alive(read_entry('mediamtx')) and port_open(8554):
        raise RuntimeError('Port 8554 is already used by an unowned process. Stop/configure it yourself or use external publishers; this helper will not kill it.')
    if dashboard and not alive(read_entry('dashboard')) and port_open(8501):
        raise RuntimeError('Port 8501 is already used by an unowned process. Stop it yourself before starting the demo dashboard.')
    return media, ffprobe, plans


def wait_rtsp(ffprobe, camera):
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        try:
            result = subprocess.run([ffprobe, '-v', 'error', '-rtsp_transport', 'tcp',
                                     '-i', f'rtsp://127.0.0.1:8554/{camera}',
                                     '-select_streams', 'v:0', '-show_entries', 'stream=codec_name',
                                     '-of', 'json'], capture_output=True, text=True, timeout=2)
            if not result.returncode and json.loads(result.stdout).get('streams'):
                return
        except (ValueError, subprocess.TimeoutExpired):
            pass
        if not alive(read_entry(camera)):
            break
        time.sleep(.3)
    raise RuntimeError(f'{camera}: publisher did not expose a video track. Check .runtime/{camera}.log and .runtime/mediamtx.log')


def start(dashboard=False):
    media, ffprobe, plans = preflight(dashboard)
    started = []
    try:
        if spawn('mediamtx', [media, str(ROOT / 'config/mediamtx-demo.yml')]):
            started.append('mediamtx')
        for _ in range(30):
            if alive(read_entry('mediamtx')) and port_open(8554):
                break
            time.sleep(.1)
        else:
            raise RuntimeError('MediaMTX did not open port 8554. Check .runtime/mediamtx.log and its configuration version.')
        for name, command in plans.items():
            if spawn(name, command):
                started.append(name)
            wait_rtsp(ffprobe, name)
        if dashboard:
            args = [sys.executable, '-m', 'streamlit', 'run', 'app.py',
                    '--server.port', '8501', '--server.address', '127.0.0.1']
            if spawn('dashboard', args):
                started.append('dashboard')
            for _ in range(40):
                try:
                    with urlopen('http://127.0.0.1:8501/_stcore/health', timeout=.5) as response:
                        if response.status == 200 and alive(read_entry('dashboard')):
                            break
                except OSError:
                    pass
                time.sleep(.2)
            else:
                raise RuntimeError('Dashboard did not become healthy. Check .runtime/dashboard.log')
    except Exception:
        # Preserve processes already running before this attempt.
        for name in reversed(started):
            stop(name)
        raise
    print('Four RTSP video tracks verified. Open Live Monitoring and click Start capture.')
    print('Run tools/verify_live_demo.py to verify changing decoded pixels; --ai additionally requires CUDA/cache.')
    if dashboard:
        print('Dashboard: http://localhost:8501')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['start', 'stop', 'status', 'check'])
    parser.add_argument('--dashboard', action='store_true')
    args = parser.parse_args()
    if sys.platform != 'linux':
        parser.exit(2, 'Use WSL/Linux for the demo lifecycle. Windows UI alone does not start RTSP publishers.\n')
    STATE.mkdir(exist_ok=True)
    import fcntl
    try:
        with (STATE / 'control.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if args.action == 'stop':
                for name in reversed(NAMES):
                    stop(name)
                print('Owned demo processes stopped. Other processes were not touched.')
            elif args.action == 'status':
                for name in NAMES:
                    print(name, 'running' if alive(read_entry(name)) else 'stopped/unmanaged')
            elif args.action == 'check':
                _, _, plans = preflight(args.dashboard)
                for name, command in plans.items():
                    print(name, 'source ready; codec mode:', command[command.index('-c:v') + 1])
                print('Preflight passed. No process was started.')
            else:
                start(args.dashboard)
    except (RuntimeError, OSError) as exc:
        parser.exit(1, f'Demo error: {exc}\n')


if __name__ == '__main__':
    main()
