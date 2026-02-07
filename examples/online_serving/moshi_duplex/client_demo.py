#!/usr/bin/env python3
"""
Moshi Full-Duplex WebSocket Client Demo.

Connects to a running vLLM-Omni server, sends synthetic audio (sine wave),
and receives generated audio output. Works in both full-duplex and
half-duplex modes.

Usage:
    python client_demo.py                           # defaults
    python client_demo.py --duplex                  # full-duplex mode
    python client_demo.py --duration 30 --output out.pcm
    python client_demo.py --input recording.pcm     # send real audio

Requirements:
    pip install websockets numpy
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import struct
import sys
import time


def generate_sine_pcm(
    duration_s: float,
    sample_rate: int = 24000,
    frequency: float = 440.0,
    amplitude: float = 0.3,
) -> bytes:
    """Generate a PCM s16le sine wave tone."""
    num_samples = int(sample_rate * duration_s)
    samples = []
    for i in range(num_samples):
        t = i / sample_rate
        value = amplitude * math.sin(2 * math.pi * frequency * t)
        samples.append(int(value * 32767))
    return struct.pack(f"<{len(samples)}h", *samples)


def load_pcm_file(path: str) -> bytes:
    """Load raw PCM bytes from a file."""
    with open(path, "rb") as f:
        return f.read()


async def run_half_duplex(
    url: str,
    audio_bytes: bytes,
    sample_rate: int,
    temperature: float,
    top_k: int,
    max_duration_s: float,
    output_path: str | None,
) -> None:
    """Half-duplex: send all audio, then receive response."""
    import websockets

    print(f"Connecting to {url} (half-duplex)...")
    async with websockets.connect(url) as ws:
        # 1. session.start
        await ws.send(json.dumps({
            "type": "session.start",
            "model": "moshi",
            "sample_rate": sample_rate,
            "audio_format": "pcm_s16le",
            "output_format": "pcm_s16le",
            "temperature": temperature,
            "top_k": top_k,
            "max_duration_s": max_duration_s,
            "duplex": False,
        }))

        response = json.loads(await ws.recv())
        if response.get("type") == "error":
            print(f"Server error: {response['message']}")
            return
        session_id = response.get("session_id", "?")
        print(f"Session created: {session_id}")

        # 2. Send audio in chunks
        chunk_size = sample_rate * 2  # 1 second of pcm_s16le
        total = len(audio_bytes)
        sent = 0
        while sent < total:
            end = min(sent + chunk_size, total)
            await ws.send(audio_bytes[sent:end])
            sent = end
            pct = sent * 100 // total
            print(f"\r  Sending audio... {pct}%", end="", flush=True)
        print()

        # 3. Signal input complete
        await ws.send(json.dumps({"type": "audio.input.done"}))
        print("  Input complete. Waiting for response...")

        # 4. Receive output
        output_chunks = []
        chunk_count = 0
        t0 = time.monotonic()

        async for message in ws:
            if isinstance(message, bytes):
                output_chunks.append(message)
                chunk_count += 1
                elapsed = time.monotonic() - t0
                print(
                    f"\r  Receiving... {chunk_count} chunks ({elapsed:.1f}s)",
                    end="", flush=True,
                )
            else:
                data = json.loads(message)
                msg_type = data.get("type", "")
                if msg_type == "generation.done":
                    total_ms = data.get("total_duration_ms", 0)
                    print(
                        f"\n  Done: {data['total_chunks']} chunks, "
                        f"{total_ms:.0f} ms audio"
                    )
                    break
                elif msg_type == "error":
                    print(f"\n  Error: {data['message']}")
                    break

        # 5. Save output
        if output_chunks and output_path:
            with open(output_path, "wb") as f:
                for chunk in output_chunks:
                    f.write(chunk)
            total_bytes = sum(len(c) for c in output_chunks)
            duration = total_bytes / (sample_rate * 2)
            print(f"  Saved {total_bytes} bytes ({duration:.1f}s) to {output_path}")
        elif not output_chunks:
            print("  No audio output received.")


async def run_full_duplex(
    url: str,
    audio_bytes: bytes,
    sample_rate: int,
    temperature: float,
    top_k: int,
    max_duration_s: float,
    output_path: str | None,
) -> None:
    """Full-duplex: send and receive audio concurrently."""
    import websockets

    print(f"Connecting to {url} (full-duplex)...")
    async with websockets.connect(url) as ws:
        # 1. session.start with duplex=True
        await ws.send(json.dumps({
            "type": "session.start",
            "model": "moshi",
            "sample_rate": sample_rate,
            "audio_format": "pcm_s16le",
            "output_format": "pcm_s16le",
            "temperature": temperature,
            "top_k": top_k,
            "max_duration_s": max_duration_s,
            "duplex": True,
        }))

        response = json.loads(await ws.recv())
        if response.get("type") == "error":
            print(f"Server error: {response['message']}")
            return
        session_id = response.get("session_id", "?")
        is_duplex = response.get("duplex", False)
        print(f"Session created: {session_id} (duplex={is_duplex})")
        if not is_duplex:
            print("  WARNING: Server fell back to half-duplex mode.")

        output_chunks: list[bytes] = []
        stop_event = asyncio.Event()
        stats = {"sent_chunks": 0, "recv_chunks": 0, "text_tokens": []}

        async def send_audio():
            """Send audio in 80ms chunks (matching Moshi frame rate)."""
            chunk_samples = int(sample_rate * 0.08)  # 80ms
            chunk_bytes = chunk_samples * 2  # pcm_s16le
            total = len(audio_bytes)
            sent = 0

            while sent < total and not stop_event.is_set():
                end = min(sent + chunk_bytes, total)
                await ws.send(audio_bytes[sent:end])
                sent = end
                stats["sent_chunks"] += 1

                # Pace to real-time (80ms per chunk)
                await asyncio.sleep(0.08)

            # Signal end of input
            await ws.send(json.dumps({"type": "session.end"}))

        async def recv_audio():
            """Receive audio output and metadata."""
            t0 = time.monotonic()
            try:
                async for message in ws:
                    if stop_event.is_set():
                        break

                    if isinstance(message, bytes):
                        output_chunks.append(message)
                        stats["recv_chunks"] += 1
                        elapsed = time.monotonic() - t0
                        print(
                            f"\r  TX: {stats['sent_chunks']} chunks | "
                            f"RX: {stats['recv_chunks']} chunks | "
                            f"{elapsed:.1f}s",
                            end="", flush=True,
                        )
                    else:
                        data = json.loads(message)
                        msg_type = data.get("type", "")
                        if msg_type == "generation.done":
                            total_ms = data.get("total_duration_ms", 0)
                            print(
                                f"\n  Done: {data['total_chunks']} chunks, "
                                f"{total_ms:.0f} ms audio"
                            )
                            stop_event.set()
                            break
                        elif msg_type == "text.token":
                            text = data.get("text", "")
                            if text:
                                stats["text_tokens"].append(text)
                        elif msg_type == "error":
                            print(f"\n  Error: {data['message']}")
                            stop_event.set()
                            break
            except Exception as e:
                if not stop_event.is_set():
                    print(f"\n  Connection error: {e}")
                    stop_event.set()

        # Run both pipelines concurrently
        print("  Streaming (send + receive simultaneously)...")
        await asyncio.gather(send_audio(), recv_audio())

        # Print text tokens if any
        if stats["text_tokens"]:
            text = "".join(stats["text_tokens"])
            print(f"  Text output: {text[:200]}")

        # Save output
        if output_chunks and output_path:
            with open(output_path, "wb") as f:
                for chunk in output_chunks:
                    f.write(chunk)
            total_bytes = sum(len(c) for c in output_chunks)
            duration = total_bytes / (sample_rate * 2)
            print(f"  Saved {total_bytes} bytes ({duration:.1f}s) to {output_path}")
        elif not output_chunks:
            print("  No audio output received.")


def main():
    parser = argparse.ArgumentParser(
        description="Moshi Full-Duplex WebSocket Client Demo",
    )
    parser.add_argument(
        "--url", default="ws://localhost:8000/v1/audio/duplex",
        help="WebSocket URL (default: ws://localhost:8000/v1/audio/duplex)",
    )
    parser.add_argument(
        "--input", default=None,
        help="Path to input PCM file (24kHz, 16-bit, mono). "
             "If not specified, generates a sine wave.",
    )
    parser.add_argument(
        "--output", "-o", default="/tmp/moshi_output.pcm",
        help="Path to save output PCM (default: /tmp/moshi_output.pcm)",
    )
    parser.add_argument(
        "--duration", type=float, default=10.0,
        help="Duration of synthetic audio in seconds (default: 10)",
    )
    parser.add_argument(
        "--duplex", action="store_true",
        help="Use full-duplex mode (concurrent send/receive)",
    )
    parser.add_argument(
        "--sample-rate", type=int, default=24000,
        help="Audio sample rate in Hz (default: 24000)",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7,
        help="Sampling temperature (default: 0.7)",
    )
    parser.add_argument(
        "--top-k", type=int, default=25,
        help="Top-k sampling (default: 25)",
    )
    parser.add_argument(
        "--max-duration", type=float, default=30.0,
        help="Max output duration in seconds (default: 30)",
    )
    args = parser.parse_args()

    # Prepare audio input
    if args.input:
        print(f"Loading audio from {args.input}...")
        audio_bytes = load_pcm_file(args.input)
        duration = len(audio_bytes) / (args.sample_rate * 2)
        print(f"  {len(audio_bytes)} bytes, {duration:.1f}s")
    else:
        print(f"Generating {args.duration}s sine wave (440 Hz)...")
        audio_bytes = generate_sine_pcm(args.duration, args.sample_rate)
        print(f"  {len(audio_bytes)} bytes")

    # Run client
    if args.duplex:
        asyncio.run(run_full_duplex(
            url=args.url,
            audio_bytes=audio_bytes,
            sample_rate=args.sample_rate,
            temperature=args.temperature,
            top_k=args.top_k,
            max_duration_s=args.max_duration,
            output_path=args.output,
        ))
    else:
        asyncio.run(run_half_duplex(
            url=args.url,
            audio_bytes=audio_bytes,
            sample_rate=args.sample_rate,
            temperature=args.temperature,
            top_k=args.top_k,
            max_duration_s=args.max_duration,
            output_path=args.output,
        ))


if __name__ == "__main__":
    main()
