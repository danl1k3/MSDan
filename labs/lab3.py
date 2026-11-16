import bisect
from collections import Counter
import math
import os
import pickle
import sys
import tkinter as tk
from tkinter import ttk
from typing import Any

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


# =====================================================================
# 1. АЛГОРИТМ ХАФФМАНА
# =====================================================================

class Node:
    def __init__(self, symbol: Any, freq: int) -> None:
        self.symbol: Any = symbol
        self.freq: int = freq
        self.left: Node | None = None
        self.right: Node | None = None


class HuffmanTree:
    def __init__(self, sequence: list[Any] | None = None) -> None:
        self.root: Node | None = None
        self.codes: dict[Any, str] = {}
        self.frequencies: dict[Any, int] = {}
        if sequence:
            self.build(sequence)

    def build(self, sequence: list[Any]) -> None:
        if not sequence:
            self.root = None
            self.codes = {}
            self.frequencies = {}
            return

        self.frequencies = dict(Counter(sequence))
        self.build_from_frequencies(self.frequencies)

    def build_from_frequencies(self, frequencies: dict[Any, int]) -> None:
        self.frequencies = frequencies
        if not frequencies:
            self.root = None
            self.codes = {}
            return

        nodes: list[Node] = [Node(sym, freq) for sym, freq in frequencies.items()]
        nodes.sort(key=lambda x: x.freq)
        freqs: list[int] = [n.freq for n in nodes]

        while len(nodes) > 1:
            left = nodes.pop(0)
            freqs.pop(0)
            right = nodes.pop(0)
            freqs.pop(0)

            merged = Node(None, left.freq + right.freq)
            merged.left = left
            merged.right = right

            idx = bisect.bisect_left(freqs, merged.freq)
            freqs.insert(idx, merged.freq)
            nodes.insert(idx, merged)

        self.root = nodes[0]
        self.codes = {}
        self._generate_codes(self.root, "")

        if len(self.codes) == 1:
            only_sym = next(iter(frequencies.keys()))
            self.codes[only_sym] = "0"

    def _generate_codes(self, node: Node | None, current_code: str) -> None:
        if node is None:
            return
        if node.symbol is not None:
            self.codes[node.symbol] = current_code
            return
        self._generate_codes(node.left, current_code + "0")
        self._generate_codes(node.right, current_code + "1")

    def encode(self, sequence: list[Any]) -> str:
        if not sequence:
            return ""
        return "".join(self.codes[sym] for sym in sequence)

    def decode(self, encoded_bits: str, original_length: int) -> list[Any]:
        if not encoded_bits or self.root is None:
            return []

        if self.root.left is None and self.root.right is None:
            return [self.root.symbol] * original_length

        decoded_tokens: list[Any] = []
        current_node = self.root

        for bit in encoded_bits:
            if bit == "0":
                current_node = current_node.left if current_node.left else current_node
            else:
                current_node = current_node.right if current_node.right else current_node

            if current_node.symbol is not None:
                decoded_tokens.append(current_node.symbol)
                current_node = self.root
                if len(decoded_tokens) == original_length:
                    break

        return decoded_tokens


# =====================================================================
# 2. МАТЕМАТИКА JPEG
# =====================================================================

STD_Q_Y = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99],
], dtype=np.float64)

STD_Q_C = np.array([
    [17, 18, 24, 47, 99, 99, 99, 99],
    [18, 21, 26, 66, 99, 99, 99, 99],
    [24, 26, 56, 99, 99, 99, 99, 99],
    [47, 66, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
], dtype=np.float64)

ZIGZAG_ORDER = [
    (0, 0), (0, 1), (1, 0), (2, 0), (1, 1), (0, 2), (0, 3), (1, 2),
    (2, 1), (3, 0), (4, 0), (3, 1), (2, 2), (1, 3), (0, 4), (0, 5),
    (1, 4), (2, 3), (3, 2), (4, 1), (5, 0), (6, 0), (5, 1), (4, 2),
    (3, 3), (2, 4), (1, 5), (0, 6), (0, 7), (1, 6), (2, 5), (3, 4),
    (4, 3), (5, 2), (6, 1), (7, 0), (7, 1), (6, 2), (5, 3), (4, 4),
    (3, 5), (2, 6), (1, 7), (2, 7), (3, 6), (4, 5), (5, 4), (6, 3),
    (7, 2), (7, 3), (6, 4), (5, 5), (4, 6), (3, 7), (4, 7), (5, 6),
    (6, 5), (7, 4), (7, 5), (6, 6), (5, 7), (6, 7), (7, 6), (7, 7),
]


def build_dct_matrix(n: int = 8) -> np.ndarray:
    T = np.zeros((n, n), dtype=np.float64)
    for u in range(n):
        alpha = math.sqrt(1.0 / n) if u == 0 else math.sqrt(2.0 / n)
        for x in range(n):
            T[u, x] = alpha * math.cos(((2 * x + 1) * u * math.pi) / (2 * n))
    return T


DCT_T = build_dct_matrix(8)
DCT_T_INV = DCT_T.T


def get_quant_matrix(base_matrix: np.ndarray, quality: int) -> np.ndarray:
    q = max(1, min(100, quality))
    scale = 5000.0 / q if q < 50 else 200.0 - 2.0 * q
    q_mat = np.floor((base_matrix * scale + 50.0) / 100.0)
    q_mat[q_mat < 1] = 1
    return q_mat


def rgb_to_ycbcr(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r = rgb[:, :, 0].astype(np.float64)
    g = rgb[:, :, 1].astype(np.float64)
    b = rgb[:, :, 2].astype(np.float64)

    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 128.0
    cr = 0.5 * r - 0.418688 * g - 0.081312 * b + 128.0
    return y, cb, cr


def rgb_to_grayscale(rgb: np.ndarray) -> np.ndarray:
    r = rgb[:, :, 0].astype(np.float64)
    g = rgb[:, :, 1].astype(np.float64)
    b = rgb[:, :, 2].astype(np.float64)
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    return np.clip(np.round(gray), 0, 255).astype(np.uint8)


def ycbcr_to_rgb(y: np.ndarray, cb: np.ndarray, cr: np.ndarray) -> np.ndarray:
    cb_shift = cb - 128.0
    cr_shift = cr - 128.0

    r = y + 1.402 * cr_shift
    g = y - 0.344136 * cb_shift - 0.714136 * cr_shift
    b = y + 1.772 * cb_shift

    rgb = np.stack([r, g, b], axis=-1)
    return np.clip(np.round(rgb), 0, 255).astype(np.uint8)


def pad_to_8(channel: np.ndarray) -> tuple[np.ndarray, int, int]:
    h, w = channel.shape
    pad_h = (8 - h % 8) % 8
    pad_w = (8 - w % 8) % 8
    if pad_h == 0 and pad_w == 0:
        return channel, h, w
    padded = np.pad(channel, ((0, pad_h), (0, pad_w)), mode="edge")
    return padded, h, w


def subsample_420(channel: np.ndarray) -> np.ndarray:
    h, w = channel.shape
    pad_h = h % 2
    pad_w = w % 2
    if pad_h or pad_w:
        channel = np.pad(channel, ((0, pad_h), (0, pad_w)), mode="edge")
    return 0.25 * (
        channel[0::2, 0::2]
        + channel[1::2, 0::2]
        + channel[0::2, 1::2]
        + channel[1::2, 1::2]
    )


def upsample_420(sub: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    up = np.repeat(np.repeat(sub, 2, axis=0), 2, axis=1)
    return up[:target_h, :target_w]


def zigzag_scan(block: np.ndarray) -> list[int]:
    return [int(block[r, c]) for r, c in ZIGZAG_ORDER]


def inverse_zigzag(coeffs: list[int]) -> np.ndarray:
    block = np.zeros((8, 8), dtype=np.float64)
    for idx, (r, c) in enumerate(ZIGZAG_ORDER):
        block[r, c] = coeffs[idx]
    return block


def encode_channel_to_tokens(channel: np.ndarray, q_matrix: np.ndarray) -> tuple[list[tuple[int, int]], tuple[int, int]]:
    padded, orig_h, orig_w = pad_to_8(channel)
    shifted = padded - 128.0
    h, w = shifted.shape

    tokens: list[tuple[int, int]] = []
    prev_dc = 0

    for row in range(0, h, 8):
        for col in range(0, w, 8):
            block = shifted[row:row + 8, col:col + 8]
            dct_block = DCT_T @ block @ DCT_T_INV
            quant_block = np.round(dct_block / q_matrix).astype(np.int32)
            zz = zigzag_scan(quant_block)

            dc_diff = zz[0] - prev_dc
            prev_dc = zz[0]
            tokens.append((-1, dc_diff))

            zero_run = 0
            for ac in zz[1:]:
                if ac == 0:
                    zero_run += 1
                else:
                    while zero_run > 15:
                        tokens.append((15, 0))
                        zero_run -= 16
                    tokens.append((zero_run, ac))
                    zero_run = 0
            tokens.append((0, 0))

    return tokens, (orig_h, orig_w)


def decode_tokens_to_channel(
    tokens: list[tuple[int, int]],
    orig_shape: tuple[int, int],
    q_matrix: np.ndarray,
) -> np.ndarray:
    orig_h, orig_w = orig_shape
    pad_h = (8 - orig_h % 8) % 8
    pad_w = (8 - orig_w % 8) % 8
    h, w = orig_h + pad_h, orig_w + pad_w

    reconstructed = np.zeros((h, w), dtype=np.float64)
    idx = 0
    prev_dc = 0
    total_tokens = len(tokens)

    for row in range(0, h, 8):
        for col in range(0, w, 8):
            zz = [0] * 64
            if idx < total_tokens:
                _, dc_diff = tokens[idx]
                idx += 1
                dc_val = prev_dc + dc_diff
                prev_dc = dc_val
                zz[0] = dc_val

            pos = 1
            while idx < total_tokens and pos < 64:
                run, val = tokens[idx]
                idx += 1
                if (run, val) == (0, 0):
                    break
                if (run, val) == (15, 0):
                    pos += 16
                    continue
                pos += run
                if pos < 64:
                    zz[pos] = val
                    pos += 1

            quant_block = inverse_zigzag(zz)
            dequant_block = quant_block * q_matrix
            idct_block = DCT_T_INV @ dequant_block @ DCT_T
            reconstructed[row:row + 8, col:col + 8] = idct_block + 128.0

    return reconstructed[:orig_h, :orig_w]


def pack_bits_to_bytes(bitstring: str) -> bytes:
    padding = (8 - len(bitstring) % 8) % 8
    padded_bits = bitstring + ("0" * padding)
    byte_arr = bytearray()
    byte_arr.append(padding)
    for i in range(0, len(padded_bits), 8):
        byte_arr.append(int(padded_bits[i:i + 8], 2))
    return bytes(byte_arr)


def unpack_bytes_to_bits(data: bytes) -> str:
    padding = data[0]
    bits = "".join(f"{byte:08b}" for byte in data[1:])
    if padding > 0:
        bits = bits[:-padding]
    return bits


def compress_and_decompress_jpeg(img_array: np.ndarray, quality: int) -> dict[str, Any]:
    is_color = img_array.ndim == 3 and img_array.shape[2] == 3
    q_y = get_quant_matrix(STD_Q_Y, quality)
    q_c = get_quant_matrix(STD_Q_C, quality)

    if is_color:
        y, cb, cr = rgb_to_ycbcr(img_array)
        cb_sub = subsample_420(cb)
        cr_sub = subsample_420(cr)

        tokens_y, shape_y = encode_channel_to_tokens(y, q_y)
        tokens_cb, shape_cb = encode_channel_to_tokens(cb_sub, q_c)
        tokens_cr, shape_cr = encode_channel_to_tokens(cr_sub, q_c)

        all_tokens = tokens_y + tokens_cb + tokens_cr
        lengths = (len(tokens_y), len(tokens_cb), len(tokens_cr))
        shapes = (shape_y, shape_cb, shape_cr)
    else:
        gray = img_array.astype(np.float64)
        tokens_y, shape_y = encode_channel_to_tokens(gray, q_y)
        all_tokens = tokens_y
        lengths = (len(tokens_y),)
        shapes = (shape_y,)

    huff = HuffmanTree(all_tokens)
    bitstring = huff.encode(all_tokens)
    packed_stream = pack_bits_to_bytes(bitstring)

    container = {
        "is_color": is_color,
        "quality": quality,
        "shapes": shapes,
        "lengths": lengths,
        "frequencies": huff.frequencies,
        "stream": packed_stream,
    }
    compressed_bytes = pickle.dumps(container, protocol=pickle.HIGHEST_PROTOCOL)

    loaded = pickle.loads(compressed_bytes)
    dec_huff = HuffmanTree()
    dec_huff.build_from_frequencies(loaded["frequencies"])
    recovered_bits = unpack_bytes_to_bits(loaded["stream"])
    total_len = sum(loaded["lengths"])
    decoded_tokens = dec_huff.decode(recovered_bits, total_len)

    if loaded["is_color"]:
        ly, lcb, lcr = loaded["lengths"]
        sy, scb, scr = loaded["shapes"]
        rec_y = decode_tokens_to_channel(decoded_tokens[:ly], sy, q_y)
        rec_cb_sub = decode_tokens_to_channel(decoded_tokens[ly:ly + lcb], scb, q_c)
        rec_cr_sub = decode_tokens_to_channel(decoded_tokens[ly + lcb:], scr, q_c)

        rec_cb = upsample_420(rec_cb_sub, sy[0], sy[1])
        rec_cr = upsample_420(rec_cr_sub, sy[0], sy[1])
        reconstructed = ycbcr_to_rgb(rec_y, rec_cb, rec_cr)
    else:
        sy = loaded["shapes"][0]
        rec_gray = decode_tokens_to_channel(decoded_tokens, sy, q_y)
        reconstructed = np.clip(np.round(rec_gray), 0, 255).astype(np.uint8)

    raw_bytes = img_array.nbytes
    comp_size = len(compressed_bytes)
    k_ratio = raw_bytes / comp_size
    space_saving = (1.0 - comp_size / raw_bytes) * 100.0

    mse = float(np.mean((img_array.astype(np.float64) - reconstructed.astype(np.float64)) ** 2))
    psnr = 100.0 if mse == 0 else 10.0 * math.log10((255.0 ** 2) / mse)

    return {
        "original": img_array,
        "reconstructed": reconstructed,
        "raw_bytes": raw_bytes,
        "comp_bytes": comp_size,
        "K": k_ratio,
        "saving_pct": space_saving,
        "psnr": psnr,
        "is_color": is_color,
        "quality": quality,
    }


# =====================================================================
# 3. ВАЛІДАЦІЯ, ЗБЕРЕЖЕННЯ ТА ВІДОБРАЖЕННЯ
# =====================================================================

RESULTS_DIR = "lab3_results"


def load_raw_image(path: str) -> tuple[np.ndarray, bool]:
    if not path or not path.strip():
        raise ValueError(
            "Помилка: не вказано шлях до зображення. "
            "Запустіть програму з аргументом шляху: uv run labs/lab3.py path/to/image.bmp"
        )
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Помилка: файл за шляхом '{path}' не знайдено."
        )

    with Image.open(path) as im:
        if im.mode in ("L", "1", "I", "F"):
            return np.array(im.convert("L"), dtype=np.uint8), False
        return np.array(im.convert("RGB"), dtype=np.uint8), True


def prepare_rgb_and_gray_pair(primary_path: str | None, secondary_path: str | None = None) -> tuple[np.ndarray, np.ndarray, str]:
    if not primary_path:
        raise ValueError(
            "Помилка: не вказано шлях до фото. "
            "Передайте шлях до повнокольорового (RGB) зображення: uv run labs/lab3.py photo_rgb.bmp"
        )

    img1, is_color1 = load_raw_image(primary_path)

    if secondary_path:
        img2, is_color2 = load_raw_image(secondary_path)
        if is_color1 and not is_color2:
            return img1, img2, f"{os.path.basename(primary_path)} + {os.path.basename(secondary_path)}"
        if not is_color1 and is_color2:
            return img2, img1, f"{os.path.basename(secondary_path)} + {os.path.basename(primary_path)}"
        if not is_color1 and not is_color2:
            raise ValueError(
                "Помилка: обидва надані файли є чорно-білими. Надайте повнокольорове (RGB) зображення."
            )
        return img1, rgb_to_grayscale(img1), f"{os.path.basename(primary_path)} (RGB + авто-ЧБ)"

    if not is_color1:
        raise ValueError(
            f"Помилка: файл '{primary_path}' є чорно-білим, а для роботи потрібне також повнокольорове зображення. "
            "Надайте шлях до RGB-фото."
        )

    gray_from_rgb = rgb_to_grayscale(img1)
    return img1, gray_from_rgb, f"{os.path.basename(primary_path)} (RGB + авто-ЧБ)"


def ask_quality_in_console() -> int:
    raw = input("Введіть якість стиснення JPEG (1..100, Enter для 35): ").strip()
    if not raw:
        return 35
    if not raw.isdigit():
        raise ValueError(f"Некоректне значення якості '{raw}'. Потрібно ввести ціле число від 1 до 100.")
    quality = int(raw)
    if not (1 <= quality <= 100):
        raise ValueError(f"Значення якості {quality} поза межами діапазону 1..100.")
    return quality


def build_pair_visuals(res: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, int]:
    orig = res["original"]
    comp = res["reconstructed"]
    is_color = res["is_color"]

    diff = np.abs(orig.astype(np.float64) - comp.astype(np.float64))
    diff_map = np.mean(diff, axis=-1) * 10.0 if is_color else diff * 10.0
    diff_map = np.clip(diff_map, 0, 255).astype(np.uint8)

    h, w = orig.shape[:2]
    crop_size = min(64, h, w)
    cy, cx = max(0, h // 2 - crop_size // 2), max(0, w // 2 - crop_size // 2)
    orig_crop = orig[cy:cy + crop_size, cx:cx + crop_size]
    comp_crop = comp[cy:cy + crop_size, cx:cx + crop_size]
    zoom_pair = np.concatenate([orig_crop, comp_crop], axis=1)

    return diff_map, zoom_pair, crop_size


def create_pair_figure(
    left_img: np.ndarray,
    right_img: np.ndarray,
    left_title: str,
    right_title: str,
    suptitle: str,
    left_cmap: str | None = None,
    right_cmap: str | None = None,
    split_x: float | None = None,
) -> plt.Figure: # type: ignore
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.4), dpi=115)
    fig.suptitle(suptitle, fontsize=11.5, fontweight="bold")

    axes[0].imshow(left_img, cmap=left_cmap, vmin=0, vmax=255)
    axes[0].set_title(left_title, fontsize=10)
    axes[0].axis("off")

    interp = "nearest" if split_x is not None else None
    axes[1].imshow(right_img, cmap=right_cmap, vmin=0, vmax=255, interpolation=interp)
    if split_x is not None:
        axes[1].axvline(x=split_x, color="cyan", linestyle="--", linewidth=1.5)
    axes[1].set_title(right_title, fontsize=10)
    axes[1].axis("off")

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
    return fig


def save_and_show_tabbed_window(
    rgb_res: dict[str, Any],
    gray_res: dict[str, Any],
    source_desc: str,
    output_dir: str = RESULTS_DIR,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    quality = rgb_res["quality"]

    rgb_diff, rgb_zoom, rgb_crop = build_pair_visuals(rgb_res)
    fig_rgb_1 = create_pair_figure(
        left_img=rgb_res["original"],
        right_img=rgb_res["reconstructed"],
        left_title=f"Початкове RGB ({rgb_res['raw_bytes']:,} Б)",
        right_title=f"Стиснене JPEG ({rgb_res['comp_bytes']:,} Б) | K = {rgb_res['K']:.2f} ({rgb_res['saving_pct']:.1f}%)",
        suptitle=f"RGB: Початкове та стиснене | Якість = {quality}/100 | Коефіцієнт стиснення K = {rgb_res['K']:.2f}",
    )
    fig_rgb_1.savefig(os.path.join(output_dir, "rgb_1_orig_vs_compressed.png"), bbox_inches="tight")

    fig_rgb_2 = create_pair_figure(
        left_img=rgb_diff,
        right_img=rgb_zoom,
        left_title="Карта втрати якості (різниця x10)",
        right_title="Фрагмент 8x8 (Зліва: оригінал | Справа: JPEG)",
        suptitle=f"RGB: Оцінка втрат | K = {rgb_res['K']:.2f} | PSNR = {rgb_res['psnr']:.2f} дБ",
        left_cmap="hot",
        split_x=float(rgb_crop) - 0.5,
    )
    fig_rgb_2.savefig(os.path.join(output_dir, "rgb_2_quality_comparison.png"), bbox_inches="tight")

    gray_diff, gray_zoom, gray_crop = build_pair_visuals(gray_res)
    fig_gray_1 = create_pair_figure(
        left_img=gray_res["original"],
        right_img=gray_res["reconstructed"],
        left_title=f"Початкове ЧБ ({gray_res['raw_bytes']:,} Б)",
        right_title=f"Стиснене JPEG ({gray_res['comp_bytes']:,} Б) | K = {gray_res['K']:.2f} ({gray_res['saving_pct']:.1f}%)",
        suptitle=f"ЧБ (Grayscale): Початкове та стиснене | Якість = {quality}/100 | Коефіцієнт стиснення K = {gray_res['K']:.2f}",
        left_cmap="gray",
        right_cmap="gray",
    )
    fig_gray_1.savefig(os.path.join(output_dir, "gray_1_orig_vs_compressed.png"), bbox_inches="tight")

    fig_gray_2 = create_pair_figure(
        left_img=gray_diff,
        right_img=gray_zoom,
        left_title="Карта втрати якості ЧБ (різниця x10)",
        right_title="Фрагмент 8x8 ЧБ (Зліва: оригінал | Справа: JPEG)",
        suptitle=f"ЧБ (Grayscale): Оцінка втрат | K = {gray_res['K']:.2f} | PSNR = {gray_res['psnr']:.2f} дБ",
        left_cmap="hot",
        right_cmap="gray",
        split_x=float(gray_crop) - 0.5,
    )
    fig_gray_2.savefig(os.path.join(output_dir, "gray_2_quality_comparison.png"), bbox_inches="tight")

    metrics_text = (
        f"Джерело: {source_desc}\n"
        f"Рівень якості JPEG: {quality}/100\n\n"
        f"1. Повнокольорове зображення (RGB):\n"
        f"   Початковий розмір: {rgb_res['raw_bytes']:,} байт\n"
        f"   Стиснений розмір:  {rgb_res['comp_bytes']:,} байт\n"
        f"   Коефіцієнт стиснення K: {rgb_res['K']:.2f} ({rgb_res['saving_pct']:.1f}%)\n"
        f"   PSNR: {rgb_res['psnr']:.2f} дБ\n\n"
        f"2. Напівтонове зображення (ЧБ / Grayscale):\n"
        f"   Початковий розмір: {gray_res['raw_bytes']:,} байт\n"
        f"   Стиснений розмір:  {gray_res['comp_bytes']:,} байт\n"
        f"   Коефіцієнт стиснення K: {gray_res['K']:.2f} ({gray_res['saving_pct']:.1f}%)\n"
        f"   PSNR: {gray_res['psnr']:.2f} дБ\n"
    )

    metrics_path = os.path.join(output_dir, "report_metrics.txt")
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write(metrics_text)

    print("\n" + "=" * 60)
    print(metrics_text.strip())
    print("=" * 60)
    print(f"Усі файли збережено у: {os.path.abspath(output_dir)}")

    # Одне вікно з вкладками ("сторінками") для перегляду результатів
    root = tk.Tk()
    root.title(f"ЛР №3 — JPEG + Хаффман ({source_desc}, Якість = {quality})")
    root.geometry("1160x660")

    notebook = ttk.Notebook(root)
    notebook.pack(fill=tk.BOTH, expand=True)

    pages = [
        ("1. RGB: Оригінал vs Стиснене", fig_rgb_1),
        ("2. RGB: Втрата якості та артефакти", fig_rgb_2),
        ("3. ЧБ: Оригінал vs Стиснене", fig_gray_1),
        ("4. ЧБ: Втрата якості та артефакти", fig_gray_2),
    ]

    for tab_title, fig in pages:
        tab_frame = ttk.Frame(notebook)
        notebook.add(tab_frame, text=tab_title)
        canvas = FigureCanvasTkAgg(fig, master=tab_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    root.protocol("WM_DELETE_WINDOW", lambda: (plt.close("all"), root.destroy()))
    root.mainloop()


if __name__ == "__main__":
    arg1 = sys.argv[1] if len(sys.argv) > 1 else None
    arg2 = sys.argv[2] if len(sys.argv) > 2 else None

    rgb_image, gray_image, description = prepare_rgb_and_gray_pair(arg1, arg2)
    chosen_quality = ask_quality_in_console()

    print("Виконується стиснення та декодування...")
    rgb_result = compress_and_decompress_jpeg(rgb_image, chosen_quality)
    gray_result = compress_and_decompress_jpeg(gray_image, chosen_quality)

    save_and_show_tabbed_window(rgb_result, gray_result, description)