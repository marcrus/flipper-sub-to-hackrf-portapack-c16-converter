import os
import argparse
import math
from typing import List, Tuple
import struct
import numpy as np


# Supported protocol types
SUPPORTED_PROTOCOLS = ['RAW']


def parse_Sub(file: str) -> dict:
    try:
        with open(file, 'r') as f:
            sub_data = f.read()
    except Exception as e:
        print(f'Cannot read input file: {e}')
        exit(-1)

    sub_chunks = [r.strip() for r in sub_data.split('\n') if r.strip()]
    
    # Safely parse header info
    info = {
        k.lower(): v.strip()
        for row in sub_chunks[:5] if ':' in row
        for k, v in [row.split(':', 1)]
    }

    print(f'Read info from file: {info}')

    if info.get('protocol') not in SUPPORTED_PROTOCOLS:
        print(f'Failed to parse {file}: Currently supported protocols are {", ".join(SUPPORTED_PROTOCOLS)} (found: {info.get("protocol")})')
        exit(-1)

    # Parse RAW_Data lines only
    info['chunks'] = [
        list(map(int, r.split(':', 1)[1].split()))
        for r in sub_chunks
        if r.startswith('RAW_Data:')
    ]

    return info


def write_HRF_file(file: str, buffer: bytes, frequency: str, sampling_rate: int) -> List[str]:
    base = os.path.splitext(file)[0]
    paths = [f'{base}.c16', f'{base}.txt']

    with open(paths[0], 'wb') as f:
        f.write(buffer)

    with open(paths[1], 'w') as f:
        f.write(generate_meta_string(frequency, sampling_rate))

    return paths


def generate_meta_string(frequency: str, sampling_rate: int) -> str:
    meta = [['sample_rate', sampling_rate], ['center_frequency', frequency]]
    return '\n'.join('='.join(map(str, r)) for r in meta)


def durations_to_bin_sequence(durations: List[List[int]], sampling_rate: int, intermediate_freq: int, amplitude: int) -> List[Tuple[int, int]]:
    sequence = []
    for chunk in durations:
        for duration in chunk:
            sequence.extend(us_to_sin(duration > 0, abs(duration), sampling_rate, intermediate_freq, amplitude))
    return sequence


def us_to_sin(level: bool, duration: int, sampling_rate: int, intermediate_freq: int, amplitude: int) -> List[Tuple[int, int]]:
    iterations = int(sampling_rate * duration / 1_000_000)
    if iterations == 0:
        return []

    step = 2 * math.pi * intermediate_freq / sampling_rate
    amp = (256 ** 2 - 1) * (amplitude / 100)

    return [
        (
            int(math.floor(math.cos(i * step) * (amp / 2))),
            int(math.floor(math.sin(i * step) * (amp / 2)))
        ) if level else (0, 0)
        for i in range(iterations)
    ]


def sequence_to_16LEBuffer(sequence: List[Tuple[int, int]]) -> bytes:
    return np.array(sequence).astype(np.int16).tobytes()


def parse_args() -> dict:
    parser = argparse.ArgumentParser(description="Convert Flipper SubGhz RAW to HackRF-compatible .c16")
    parser.add_argument('file', help="Input .sub file")
    parser.add_argument('-o', '--output', help="Output file name (without extension)")
    parser.add_argument('-sr', '--sampling_rate', type=int, default=500000, help="Sampling rate (default 500000)")
    parser.add_argument('-if', '--intermediate_freq', type=int, help="Intermediate frequency (defaults to sr / 100)")
    parser.add_argument('-a', '--amplitude', type=int, default=100, help="Amplitude percentage (default 100%)")
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    return vars(parser.parse_args())


if __name__ == '__main__':
    args = parse_args()

    file = args.get('file')
    output = args.get('output') or os.path.splitext(file)[0]
    sampling_rate = args.get('sampling_rate')
    intermediate_freq = args.get('intermediate_freq') or (sampling_rate // 100)
    amplitude = args.get('amplitude')
    verbose = args.get('verbose')

    info = parse_Sub(file)
    if verbose:
        print(f'Sub File information: {info}')

    chunks = info.get('chunks', [])
    if verbose:
        print(f'Found {len(chunks)} data chunks')

    iq_sequence = durations_to_bin_sequence(chunks, sampling_rate, intermediate_freq, amplitude)
    buffer = sequence_to_16LEBuffer(iq_sequence)

    out_files = write_HRF_file(output, buffer, info.get('frequency', '0'), sampling_rate)
    print(f'Written {round(len(buffer) / 1024)} KiB, {len(iq_sequence) / sampling_rate:.2f} seconds to: {", ".join(out_files)}')
