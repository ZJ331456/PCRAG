"""Preserve the local vLLM command while adjusting its memory allocation."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import urllib.request
from urllib.parse import urlsplit


def running(port):
    matches = []
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():
            continue
        try:
            argv = entry.joinpath('cmdline').read_bytes().decode().strip('\0').split('\0')
            if ('vllm.entrypoints.openai.api_server' in argv and '--port' in argv
                    and argv[argv.index('--port') + 1] == str(port)):
                matches.append((int(entry.name), argv))
        except (OSError, ValueError, IndexError, UnicodeError):
            continue
    if len(matches) > 1:
        raise RuntimeError('Multiple API servers match the requested port')
    return matches[0] if matches else None


def alive(pid):
    try:
        state = Path(f'/proc/{pid}/stat').read_text().split(') ', 1)[1].split()[0]
        return state != 'Z'
    except (OSError, IndexError):
        return False


def stop(pid):
    children = []
    for entry in Path('/proc').iterdir():
        if entry.name.isdigit():
            try:
                status = entry.joinpath('stat').read_text().split(') ', 1)[1].split()
                if int(status[1]) == pid:
                    children.append(int(entry.name))
            except (OSError, ValueError, IndexError):
                pass
    os.kill(pid, signal.SIGTERM)
    deadline = time.monotonic() + 20
    targets = [pid, *children]
    while any(alive(target) for target in targets) and time.monotonic() < deadline:
        time.sleep(.25)
    for target in targets:
        if alive(target):
            os.kill(target, signal.SIGKILL)
    deadline = time.monotonic() + 10
    while any(alive(target) for target in targets) and time.monotonic() < deadline:
        time.sleep(.25)
    if any(alive(target) for target in targets):
        raise RuntimeError('Previous vLLM service did not release its processes')


def run(args):
    if not 0 < args.utilization < 1:
        raise ValueError('Utilization must be between zero and one')
    url = urlsplit(args.base_url)
    if url.hostname not in ('127.0.0.1', 'localhost') or not url.port:
        raise ValueError('This helper only restarts a local service with an explicit port')
    out = Path(args.out_root).resolve()
    metadata = out / 'metadata/nv2_memory_tuning'
    metadata.mkdir(parents=True, exist_ok=True)
    original = metadata / 'original_vllm_command.json'
    current = running(url.port)
    if original.exists():
        argv = json.loads(original.read_text())['argv']
    elif current:
        argv = current[1]
        if '--gpu-memory-utilization' not in argv or '--max-model-len' not in argv:
            raise ValueError('Original service must expose its memory and context settings')
        original.write_text(json.dumps({'pid': current[0], 'argv': argv}, indent=2) + '\n')
    else:
        raise RuntimeError('No existing service command is available to preserve')
    env = os.environ.copy()
    # Preserve only runtime variables; never serialize process credentials.
    if current:
        for item in Path(f'/proc/{current[0]}/environ').read_bytes().split(b'\0'):
            key, separator, value = item.partition(b'=')
            name = key.decode('utf-8', 'replace')
            if separator and name.startswith(('VLLM_', 'CUDA_', 'NCCL_', 'TORCH_')):
                env[name] = value.decode('utf-8', 'replace')
        print(f'[service-stop] pid={current[0]} port={url.port}', flush=True)
        stop(current[0])
    argv = list(argv)
    argv[argv.index('--gpu-memory-utilization') + 1] = f'{args.utilization:.2f}'
    suffix = f'{int(round(args.utilization * 100)):03d}'
    logfile = out / 'logs' / f'vllm_gpu_{suffix}.log'
    logfile.parent.mkdir(parents=True, exist_ok=True)
    with logfile.open('ab') as stream:
        process = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=stream,
                                   stderr=subprocess.STDOUT, env=env,
                                   start_new_session=True, close_fds=True)
    info = {'pid': process.pid, 'argv': argv, 'gpu_memory_utilization': args.utilization,
            'log': str(logfile), 'base_url': args.base_url,
            'max_model_len': int(argv[argv.index('--max-model-len') + 1])}
    (metadata / 'active_service.json').write_text(json.dumps(info, indent=2) + '\n')
    print(f'[service-start] pid={process.pid} utilization={args.utilization:.2f} log={logfile}', flush=True)
    deadline = time.monotonic() + 360
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f'vLLM exited with {process.returncode}; inspect {logfile}')
        try:
            with urllib.request.urlopen(args.base_url.rstrip('/') + '/models', timeout=2) as response:
                if response.status == 200:
                    print(f'[service-ready] {json.dumps(info)}', flush=True)
                    return 0
        except (OSError, TimeoutError):
            pass
        time.sleep(1)
    raise RuntimeError(f'vLLM startup timed out; inspect {logfile}')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--utilization', type=float, required=True)
    parser.add_argument('--base-url', default='http://127.0.0.1:8035/v1')
    parser.add_argument('--out-root', default='/root/PathCondRAG/outputs/3multi_hop_datasets_results_with_nv2_10_8')
    return run(parser.parse_args(argv))
