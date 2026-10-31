"""Opt-in real-stream/GPU check. Writes a NEW timestamped evidence file."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.camera_capture import CameraCaptureManager
from src.live_camera import ask_latest, read_camera_config


def wait_valid(manager, camera, timeout=20):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        snapshot = manager.snapshot(camera)
        if snapshot['valid']:
            return snapshot
        time.sleep(.1)
    raise RuntimeError(f'{camera} did not deliver fresh frames: {snapshot["status"]}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ai', action='store_true')
    parser.add_argument('--reconnect', action='store_true')
    args = parser.parse_args()
    cameras, stale = read_camera_config()
    manager = CameraCaptureManager({k: v['url'] for k, v in cameras.items()},
                                   stale_after_seconds=stale)
    report = {'started_utc': datetime.now(timezone.utc).isoformat(), 'streams': {}}
    try:
        manager.start_many(manager.camera_ids)
        first = {c: wait_valid(manager, c) for c in manager.camera_ids}
        # Check decoded pixel changes, not merely increasing sequence numbers.
        digests = {c: {hashlib.sha256(s['frame'].tobytes()).hexdigest()} for c, s in first.items()}
        end = time.monotonic() + 6
        while time.monotonic() < end:
            time.sleep(.3)
            for c in manager.camera_ids:
                s = manager.latest_valid(c)
                digests[c].add(hashlib.sha256(s['frame'].tobytes()).hexdigest())
        for c in manager.camera_ids:
            s = manager.latest_valid(c)
            assert len(digests[c]) > 1, f'{c}: decoded pixels did not change'
            report['streams'][c] = {'url': cameras[c]['url'], 'distinct_decoded_frames': len(digests[c]),
                                    'frames_received': s['frames_received'], 'shape': list(s['frame'].shape),
                                    'age_seconds': s['age_seconds']}
        print('PASS: all four RTSP streams deliver changing decoded frames', flush=True)
        if args.reconnect:
            from tools.demo_runtime import stop, spawn
            c = 'cam1'
            previous = manager.latest_valid(c)['connection']
            stop(c)
            end = time.monotonic() + 15
            while manager.snapshot(c)['valid'] and time.monotonic() < end:
                time.sleep(.1)
            assert not manager.snapshot(c)['valid'], 'Disconnected frame remained valid'
            try:
                manager.latest_valid(c)
                raise AssertionError('Unavailable frame accepted')
            except RuntimeError:
                pass
            entry = json.loads((ROOT / '.runtime/cam1.json').read_text())
            spawn(c, entry['args'])
            recovered = wait_valid(manager, c)
            assert recovered['connection'] > previous
            report['reconnect'] = {'rejected_while_offline': True, 'connection': recovered['connection']}
            print('PASS: publisher disconnect rejected; capture reconnected', flush=True)
        if args.ai:
            result = ask_latest(manager, 'cam1', 'What is visible in this image?')
            metadata = result['frame_metadata']
            rgb = metadata.pop('submitted_rgb')
            from PIL import Image
            import torch
            stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
            image_path = ROOT / 'verification' / f'live-submitted-{stamp}.png'
            Image.fromarray(rgb).save(image_path)
            report['ai'] = {**result, 'question': 'What is visible in this image?',
                            'submitted_image': str(image_path.relative_to(ROOT)),
                            'gpu': torch.cuda.get_device_name(),
                            'allocated_mib': torch.cuda.memory_allocated() / 2**20,
                            'peak_allocated_mib': torch.cuda.max_memory_allocated() / 2**20}
            print(json.dumps(report['ai'], indent=2), flush=True)
        report['passed'] = True
    finally:
        manager.close()
        path = ROOT / 'verification' / ('live-demo-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '.json')
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps(report, indent=2))
        print(f'Evidence: {path}', flush=True)


if __name__ == '__main__':
    main()
