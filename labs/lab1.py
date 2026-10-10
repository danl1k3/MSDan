from collections import Counter
import math
from decimal import Decimal, getcontext


# --- Haffman Algo ---

class Node:

  def __init__(self, char: str | None, freq: int) -> None:
    self.char: str | None = char
    self.freq: int = freq
    self.left: Node | None = None
    self.right: Node | None = None


class HuffmanTree:

  def __init__(self, text: str = "") -> None:
    self.root: Node | None = None
    self.codes: dict[str, str] = {}
    if text:
      self.build(text)

  def build(self, text: str) -> None:
    if not text:
      self.root = None
      self.codes = {}
      return

    frequency = Counter(text)
    nodes: list[Node] = [Node(char, freq) for char, freq in frequency.items()]

    while len(nodes) > 1:
      nodes.sort(key=lambda x: x.freq, reverse=True)
      left = nodes.pop()
      right = nodes.pop()

      merged = Node(None, left.freq + right.freq)
      merged.left = left
      merged.right = right
      nodes.append(merged)

    self.root = nodes[0]
    self.codes = {}
    self._generate_codes(self.root, "")

    if len(self.codes) == 1:
      self.codes[text[0]] = "0"

  def _generate_codes(self, node: Node | None, current_code: str) -> None:
    if node is None:
      return

    if node.char is not None:
      self.codes[node.char] = current_code
      return

    self._generate_codes(node.left, current_code + "0")
    self._generate_codes(node.right, current_code + "1")

  def encode(self, text: str) -> str:
    if not text:
      return ""
    return "".join(self.codes[char] for char in text)

  def decode(self, encoded_text: str, original_length: int) -> str:
    if not encoded_text or self.root is None:
      return ""

    if self.root.left is None and self.root.right is None:
      return (self.root.char or "") * original_length

    decoded_chars: list[str] = []
    current_node = self.root

    for bit in encoded_text:
      if bit == "0":
        current_node = current_node.left if current_node.left else current_node
      else:
        current_node = (
            current_node.right if current_node.right else current_node
        )

      if current_node.char is not None:
        decoded_chars.append(current_node.char)
        current_node = self.root

    return "".join(decoded_chars)

# --- Arithmetic Algo

def get_arithmetic_ranges(
    text: str,
) -> dict[str, tuple[Decimal, Decimal]]:
  total_length = Decimal(len(text))
  counts = Counter(text)

  ranges: dict[str, tuple[Decimal, Decimal]] = {}
  current_low = Decimal("0.0")

  for char, count in counts.items():
    prob = Decimal(count) / total_length
    ranges[char] = (current_low, current_low + prob)
    current_low += prob

  return ranges


def arithmetic_compress(
    text: str,
) -> tuple[Decimal, dict[str, tuple[Decimal, Decimal]]]:
  if not text:
    return Decimal("0.0"), {}

  ranges = get_arithmetic_ranges(text)
  low = Decimal("0.0")
  high = Decimal("1.0")

  for char in text:
    symbol_low, symbol_high = ranges[char]
    current_range = high - low
    high = low + current_range * symbol_high
    low = low + current_range * symbol_low

  encoded_value = (low + high) / Decimal("2")
  return encoded_value, ranges


def arithmetic_decompress(
    encoded_value: Decimal,
    ranges: dict[str, tuple[Decimal, Decimal]],
    length: int,
) -> str:
  if length == 0:
    return ""

  decoded_chars = []
  value = encoded_value

  for _ in range(length):
    for char, (low, high) in ranges.items():
      if low <= value < high:
        decoded_chars.append(char)
        value = (value - low) / (high - low)
        break

  return "".join(decoded_chars)


if __name__ == "__main__":
  text = "ABRACADABRA SIMSALABIM HUFFMAN CODING EXAMPLE"

  print(f"\nТекст ({len(text)} символів): {text}")
  orig_bits = len(text.encode("utf-8")) * 8
  print(f"Початковий розмір (UTF-8): {orig_bits} біт [{orig_bits / 8:.1f} байт]\n")

  # --- Run Huff ---
  huffman_tree = HuffmanTree(text)
  huff_encoded = huffman_tree.encode(text)
  huff_decoded = huffman_tree.decode(huff_encoded, len(text))

  huff_bits = len(huff_encoded)
  huff_ratio = (1 - (huff_bits / orig_bits)) * 100

  print("--- 1. Алгоритм Хаффмана ---")
  print(f"  Декодований текст:  {huff_decoded}")
  print(f"  Розмір даних:       {huff_bits} біт")
  print(f"  Стиснення:          {huff_ratio:.2f}%")

 
  getcontext().prec = 150

  # --- Run Arithmetic ---
  arith_value, arith_ranges = arithmetic_compress(text)
  arith_decoded = arithmetic_decompress(arith_value, arith_ranges, len(text))

  low = Decimal("0.0")
  high = Decimal("1.0")
  for char in text:
    s_low, s_high = arith_ranges[char]
    s_low, s_high = Decimal(str(s_low)), Decimal(str(s_high))
    curr_range = high - low
    high = low + curr_range * s_high
    low = low + curr_range * s_low

  final_range = high - low

  if final_range > Decimal("0"):
    ln_2 = Decimal("2").ln()
    arith_bits = math.ceil(-(final_range.ln() / ln_2))
  else:
    arith_bits = 0

  arith_ratio = (1 - (arith_bits / orig_bits)) * 100

  print("\n--- 2. Арифметичне кодування ---")
  print(f"  Декодований текст: {arith_decoded}")
  print(f"  Розмір даних:       ~{arith_bits} біт")
  print(f"  Стиснення:          {arith_ratio:.2f}%")