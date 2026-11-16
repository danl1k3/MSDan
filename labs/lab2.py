from __future__ import annotations

import hashlib
import sys
import time
from pathlib import Path
from typing import NamedTuple


class Token(NamedTuple):
    prefix: int
    symbol: int


class LZ78:
    def compress(self, data: bytes) -> bytes:
        if not data:
            return b""

        # 1. Генерація токенів LZ78
        tokens: list[Token] = []
        dictionary: dict[tuple[int, int], int] = {}
        current_node = 0
        dict_size = 1

        for byte in data:
            key = (current_node, byte)
            if key in dictionary:
                current_node = dictionary[key]
            else:
                tokens.append(Token(current_node, byte))
                dictionary[key] = dict_size
                dict_size += 1
                current_node = 0

        if current_node != 0:
            for (parent, symbol), idx in dictionary.items():
                if idx == current_node:
                    tokens.append(Token(parent, symbol))
                    break

        # 2. Бітове упакування токенів
        packed_stream = bytearray()
        acc = 0
        bits = 0

        for pos, token in enumerate(tokens):
            w = max(1, pos.bit_length())
            acc = (acc << (w + 8)) | (token.prefix << 8) | token.symbol
            bits += w + 8

            while bits >= 8:
                bits -= 8
                packed_stream.append((acc >> bits) & 0xFF)

            acc &= (1 << bits) - 1

        if bits > 0:
            packed_stream.append((acc << (8 - bits)) & 0xFF)

        # 3. Формування заголовка (20 байтів)
        header = (
            b"LZ78"
            + len(data).to_bytes(8, byteorder="big")
            + len(tokens).to_bytes(8, byteorder="big")
        )

        return header + bytes(packed_stream)

    def decompress(self, blob: bytes) -> bytes:
        if not blob:
            return b""
        if len(blob) < 20:
            raise ValueError("Розмір стиснених даних занадто малий для обробки заголовка.")

        magic = blob[:4]
        if magic != b"LZ78":
            raise ValueError("Некоректний формат файлу (хибна сигнатура).")

        original_size = int.from_bytes(blob[4:12], byteorder="big")
        token_count = int.from_bytes(blob[12:20], byteorder="big")
        payload = blob[20:]

        # 1. Бітове розпакування токенів
        tokens: list[Token] = []
        acc = 0
        bits = 0
        ptr = 0

        for pos in range(token_count):
            w = max(1, pos.bit_length())
            
            while bits < (w + 8):
                acc = (acc << 8) | payload[ptr]
                ptr += 1
                bits += 8

            bits -= w
            idx = (acc >> bits) & ((1 << w) - 1)
            acc &= (1 << bits) - 1

            bits -= 8
            sym = (acc >> bits) & 0xFF
            acc &= (1 << bits) - 1

            tokens.append(Token(idx, sym))

        # 2. Відновлення вихідних байтів за словником
        dictionary: list[bytes] = [b""]
        output = bytearray()

        for idx, sym in tokens:
            phrase = dictionary[idx] + bytes([sym])
            dictionary.append(phrase)
            output.extend(phrase)

        decompressed = bytes(output)
        if len(decompressed) != original_size:
            raise ValueError("Невідповідність розміру відновлених даних заголовку.")

        return decompressed


def main() -> None:
    codec = LZ78()

    folder = Path("lab2_files")
    default_files = [
        folder / "example.txt",
        folder / "dxcompiler.dll",
    ]

    
    file_paths = [Path(arg) for arg in sys.argv[1:]] or default_files

    for path in file_paths:
        if not path.is_file():
            print(f"[!] Файл '{path}' не знайдено.", file=sys.stderr)
            continue

        # --- КРОК 1: Зчитування оригінального файлу ---
        raw_bytes = path.read_bytes()
        orig_size = len(raw_bytes)
        sha_orig = hashlib.sha256(raw_bytes).hexdigest()

        print(f"Processing: {path.name}")
        print(f"  Original Size : {orig_size:,} bytes")
        print(f"  SHA-256 Orig  : {sha_orig}")

        # --- КРОК 2: Стиснення ---
        t0 = time.perf_counter()
        compressed_blob = codec.compress(raw_bytes)
        t_comp = time.perf_counter() - t0

        archive_path = path.with_suffix(path.suffix + ".lz78")
        archive_path.write_bytes(compressed_blob)

        comp_size = len(compressed_blob)
        tokens_count = int.from_bytes(compressed_blob[12:20], byteorder="big")
        saving = ((orig_size - comp_size) / orig_size * 100) if orig_size > 0 else 0.0
        ratio = (orig_size / comp_size) if comp_size > 0 else 0.0

        print(f"  [Compression]")
        print(f"    Packed Size   : {comp_size:,} bytes")
        print(f"    Tokens Count  : {tokens_count:,}")
        print(f"    Ratio / Saved : {ratio:.2f}x ({saving:.2f}%)")
        print(f"    Encoding Time : {t_comp * 1000:.2f} ms")

        # --- КРОК 3: Декомпресія та перевірка ---
        t1 = time.perf_counter()
        restored_bytes = codec.decompress(archive_path.read_bytes())
        t_decomp = time.perf_counter() - t1

        restored_path = path.with_name(f"{path.stem}.restored{path.suffix}")
        restored_path.write_bytes(restored_bytes)

        sha_rest = hashlib.sha256(restored_bytes).hexdigest()
        is_valid = (sha_orig == sha_rest)

        print(f"  [Decompression]")
        print(f"    Decoding Time : {t_decomp * 1000:.2f} ms")
        print(f"    SHA-256 Rest  : {sha_rest}")
        print(f"  Status        : {'OK' if is_valid else 'ERROR'}\n")


if __name__ == "__main__":
    main()