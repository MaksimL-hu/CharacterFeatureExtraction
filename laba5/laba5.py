from pathlib import Path
import csv
import unicodedata
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


# -----------------------------
# НАСТРОЙКИ
# -----------------------------
OUTPUT_DIR = Path("lab5_output")
SYMBOLS_DIR = OUTPUT_DIR / "symbols"
PROFILES_DIR = OUTPUT_DIR / "profiles"
CSV_PATH = OUTPUT_DIR / "features.csv"
ALPHABET_PATH = OUTPUT_DIR / "alphabet.txt"

FONT_SIZE = 52

FONT_PATH = r"C:\Users\m9164\AppData\Local\Microsoft\Windows\Fonts\PonomarUnicode.ttf"

# Холст для рендеринга символов
CANVAS_SIZE = (240, 240)

# Порог бинаризации после рендеринга
BINARIZE_THRESHOLD = 200

# Дополнительный отступ после обрезки белых полей
CROP_PADDING = 0

# Набор символов: только те, что указаны в строке "Кириллица заглавные"
ALPHABET = [
    "А", "Б", "В", "Г", "Д", "Є", "Ж", "Ѕ", "З", "И", "І",
    "К", "Л", "М", "Н", "О", "П", "Р", "С", "Т", "Ѹ",
    "Ф", "Х", "Ц", "Ч", "Ш", "Щ", "Ъ", "Ы", "Ь", "Ѣ",
    "Ю", "Ѵ", "Ѯ", "Ѱ", "Ѡ", "Ѧ", "Ѩ"
]

for folder in [OUTPUT_DIR, SYMBOLS_DIR, PROFILES_DIR]:
    folder.mkdir(parents=True, exist_ok=True)


# -----------------------------
# ПОИСК ШРИФТА
# -----------------------------
def find_ponomar_font() -> str:
    """
    Ищет установленный шрифт Ponomar Unicode.
    """
    if FONT_PATH is not None:
        path = Path(FONT_PATH)
        if path.exists():
            return str(path)
        raise FileNotFoundError(f"Шрифт не найден: {FONT_PATH}")

    candidates = [
        r"C:\Windows\Fonts\PonomarUnicode.ttf",
        r"C:\Windows\Fonts\PonomarUnicode.otf",
        r"C:\Windows\Fonts\Ponomar Unicode TT.ttf",
        r"C:\Windows\Fonts\Ponomar Unicode.ttf",
    ]

    for candidate in candidates:
        if Path(candidate).exists():
            return candidate

    windows_fonts = Path(r"C:\Windows\Fonts")
    if windows_fonts.exists():
        for path in windows_fonts.glob("*Ponomar*.*"):
            if path.suffix.lower() in [".ttf", ".otf"]:
                return str(path)

    raise FileNotFoundError(
        "Не найден шрифт Ponomar Unicode. "
        "Укажите путь к нему в переменной FONT_PATH."
    )


# -----------------------------
# СОХРАНЕНИЕ АЛФАВИТА
# -----------------------------
def save_alphabet_txt(alphabet: list[str], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(" ".join(alphabet))


# -----------------------------
# РЕНДЕРИНГ СИМВОЛА
# -----------------------------
def render_symbol(symbol: str, font: ImageFont.FreeTypeFont) -> np.ndarray:
    """
    Рендерит символ в grayscale-изображение.
    Чёрный символ на белом фоне.
    """
    image = Image.new("L", CANVAS_SIZE, 255)
    draw = ImageDraw.Draw(image)

    bbox = draw.textbbox((0, 0), symbol, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    x = (CANVAS_SIZE[0] - text_w) // 2 - bbox[0]
    y = (CANVAS_SIZE[1] - text_h) // 2 - bbox[1]

    draw.text((x, y), symbol, font=font, fill=0)

    return np.array(image, dtype=np.uint8)


def grayscale_to_binary_manual(gray: np.ndarray, threshold: int = 200) -> np.ndarray:
    """
    Чёрный = 1, белый = 0
    """
    return np.where(gray < threshold, 1, 0).astype(np.uint8)


def crop_binary_image(binary: np.ndarray, padding: int = 0) -> np.ndarray:
    """
    Обрезает белые поля вокруг символа.
    """
    ys, xs = np.where(binary == 1)

    if len(xs) == 0 or len(ys) == 0:
        return binary.copy()

    x_min = max(0, xs.min() - padding)
    x_max = min(binary.shape[1], xs.max() + 1 + padding)
    y_min = max(0, ys.min() - padding)
    y_max = min(binary.shape[0], ys.max() + 1 + padding)

    return binary[y_min:y_max, x_min:x_max]


def save_binary_symbol_image(binary: np.ndarray, path: Path) -> None:
    """
    Сохраняет бинарное изображение как чёрный символ на белом фоне.
    """
    img = np.where(binary == 1, 0, 255).astype(np.uint8)
    Image.fromarray(img, mode="L").save(path)


# -----------------------------
# ПРИЗНАКИ
# -----------------------------
def split_into_quarters(binary: np.ndarray):
    h, w = binary.shape
    mid_y = h // 2
    mid_x = w // 2

    q1 = binary[:mid_y, :mid_x]   # верх-лево
    q2 = binary[:mid_y, mid_x:]   # верх-право
    q3 = binary[mid_y:, :mid_x]   # низ-лево
    q4 = binary[mid_y:, mid_x:]   # низ-право

    return q1, q2, q3, q4


def quarter_weight(q: np.ndarray) -> int:
    return int(q.sum())


def quarter_relative_weight(q: np.ndarray) -> float:
    area = q.shape[0] * q.shape[1]
    if area == 0:
        return 0.0
    return float(q.sum()) / float(area)


def center_of_mass(binary: np.ndarray) -> tuple[float, float]:
    """
    Координаты центра тяжести чёрных пикселей.
    x — по горизонтали, y — по вертикали.
    """
    h, w = binary.shape
    weight = binary.sum()

    if weight == 0:
        return 0.0, 0.0

    y_indices, x_indices = np.indices((h, w))
    x_c = float((x_indices * binary).sum()) / float(weight)
    y_c = float((y_indices * binary).sum()) / float(weight)

    return x_c, y_c


def normalized_center_of_mass(binary: np.ndarray, x_c: float, y_c: float) -> tuple[float, float]:
    h, w = binary.shape

    x_rel = x_c / (w - 1) if w > 1 else 0.0
    y_rel = y_c / (h - 1) if h > 1 else 0.0

    return x_rel, y_rel


def axial_moments(binary: np.ndarray, x_c: float, y_c: float) -> tuple[float, float]:
    """
    Осевые моменты инерции:
    Ix — относительно горизонтальной оси,
    Iy — относительно вертикальной оси.
    """
    h, w = binary.shape
    y_indices, x_indices = np.indices((h, w))

    ix = float((((y_indices - y_c) ** 2) * binary).sum())
    iy = float((((x_indices - x_c) ** 2) * binary).sum())

    return ix, iy


def normalized_axial_moments(binary: np.ndarray, ix: float, iy: float) -> tuple[float, float]:
    """
    Нормированные осевые моменты.
    """
    h, w = binary.shape
    denom = float((w * w) * (h * h))

    if denom == 0:
        return 0.0, 0.0

    return ix / denom, iy / denom


def profiles(binary: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Профили:
    X — сумма по столбцам,
    Y — сумма по строкам.
    """
    profile_x = binary.sum(axis=0).astype(int)
    profile_y = binary.sum(axis=1).astype(int)
    return profile_x, profile_y


# -----------------------------
# СОХРАНЕНИЕ ПРОФИЛЕЙ
# -----------------------------
def save_profiles_png(symbol: str, profile_x: np.ndarray, profile_y: np.ndarray, out_path: Path) -> None:
    """
    Сохраняет профили X и Y в одном PNG:
    - X — обычная столбчатая диаграмма;
    - Y — горизонтальная столбчатая диаграмма
      с правильной ориентацией по оси Y.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Профиль X
    axes[0].bar(np.arange(len(profile_x)), profile_x)
    axes[0].set_title(f"Профиль X: {symbol}")
    axes[0].set_xlabel("X")
    axes[0].set_ylabel("Сумма чёрных пикселей")
    axes[0].xaxis.set_major_locator(MaxNLocator(integer=True))
    axes[0].yaxis.set_major_locator(MaxNLocator(integer=True))

    # Профиль Y
    axes[1].barh(np.arange(len(profile_y)), profile_y)
    axes[1].set_title(f"Профиль Y: {symbol}")
    axes[1].set_xlabel("Сумма чёрных пикселей")
    axes[1].set_ylabel("Y")
    axes[1].xaxis.set_major_locator(MaxNLocator(integer=True))
    axes[1].yaxis.set_major_locator(MaxNLocator(integer=True))
    axes[1].invert_yaxis()

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# -----------------------------
# ВСПОМОГАТЕЛЬНОЕ
# -----------------------------
def fmt_float(x: float) -> str:
    return f"{x:.6f}"


def make_safe_name(symbol: str) -> str:
    return f"U+{ord(symbol):04X}_{symbol}"


# -----------------------------
# ОБРАБОТКА ОДНОГО СИМВОЛА
# -----------------------------
def process_symbol(symbol: str, font: ImageFont.FreeTypeFont) -> dict:
    gray = render_symbol(symbol, font)
    binary = grayscale_to_binary_manual(gray, threshold=BINARIZE_THRESHOLD)
    binary = crop_binary_image(binary, padding=CROP_PADDING)

    safe_name = make_safe_name(symbol)

    symbol_path = SYMBOLS_DIR / f"{safe_name}.png"
    save_binary_symbol_image(binary, symbol_path)

    q1, q2, q3, q4 = split_into_quarters(binary)

    q1_w = quarter_weight(q1)
    q2_w = quarter_weight(q2)
    q3_w = quarter_weight(q3)
    q4_w = quarter_weight(q4)

    q1_rel = quarter_relative_weight(q1)
    q2_rel = quarter_relative_weight(q2)
    q3_rel = quarter_relative_weight(q3)
    q4_rel = quarter_relative_weight(q4)

    total_weight = int(binary.sum())
    h, w = binary.shape

    x_c, y_c = center_of_mass(binary)
    x_rel, y_rel = normalized_center_of_mass(binary, x_c, y_c)

    ix, iy = axial_moments(binary, x_c, y_c)
    ix_rel, iy_rel = normalized_axial_moments(binary, ix, iy)

    profile_x, profile_y = profiles(binary)
    profile_path = PROFILES_DIR / f"{safe_name}_profiles.png"
    save_profiles_png(symbol, profile_x, profile_y, profile_path)

    return {
        "symbol": symbol,
        "unicode": f"U+{ord(symbol):04X}",
        "unicode_name": unicodedata.name(symbol, ""),
        "width": w,
        "height": h,
        "total_weight": total_weight,

        "q1_weight": q1_w,
        "q2_weight": q2_w,
        "q3_weight": q3_w,
        "q4_weight": q4_w,

        "q1_relative_weight": q1_rel,
        "q2_relative_weight": q2_rel,
        "q3_relative_weight": q3_rel,
        "q4_relative_weight": q4_rel,

        "center_x": x_c,
        "center_y": y_c,
        "center_x_norm": x_rel,
        "center_y_norm": y_rel,

        "Ix": ix,
        "Iy": iy,
        "Ix_norm": ix_rel,
        "Iy_norm": iy_rel,

        "symbol_image": str(symbol_path),
        "profiles_image": str(profile_path),
    }


# -----------------------------
# CSV
# -----------------------------
def save_features_csv(rows: list[dict], csv_path: Path) -> None:
    fieldnames = [
        "symbol",
        "unicode",
        "unicode_name",
        "width",
        "height",
        "total_weight",

        "q1_weight",
        "q2_weight",
        "q3_weight",
        "q4_weight",

        "q1_relative_weight",
        "q2_relative_weight",
        "q3_relative_weight",
        "q4_relative_weight",

        "center_x",
        "center_y",
        "center_x_norm",
        "center_y_norm",

        "Ix",
        "Iy",
        "Ix_norm",
        "Iy_norm",

        "symbol_image",
        "profiles_image",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(fieldnames)

        for row in rows:
            writer.writerow([
                row["symbol"],
                row["unicode"],
                row["unicode_name"],
                row["width"],
                row["height"],
                row["total_weight"],

                row["q1_weight"],
                row["q2_weight"],
                row["q3_weight"],
                row["q4_weight"],

                fmt_float(row["q1_relative_weight"]),
                fmt_float(row["q2_relative_weight"]),
                fmt_float(row["q3_relative_weight"]),
                fmt_float(row["q4_relative_weight"]),

                fmt_float(row["center_x"]),
                fmt_float(row["center_y"]),
                fmt_float(row["center_x_norm"]),
                fmt_float(row["center_y_norm"]),

                fmt_float(row["Ix"]),
                fmt_float(row["Iy"]),
                fmt_float(row["Ix_norm"]),
                fmt_float(row["Iy_norm"]),

                row["symbol_image"],
                row["profiles_image"],
            ])


# -----------------------------
# MAIN
# -----------------------------
def main():
    print("=== Лабораторная работа №5 ===")
    print("Выделение признаков символов")
    print("Алфавит: Кириллица заглавные")
    print(f"Кегль: {FONT_SIZE}")
    print()

    font_path = find_ponomar_font()
    print(f"Используемый шрифт: {font_path}")

    save_alphabet_txt(ALPHABET, ALPHABET_PATH)

    print(f"Количество символов: {len(ALPHABET)}")
    print("Алфавит:")
    print(" ".join(ALPHABET))
    print()

    font = ImageFont.truetype(font_path, FONT_SIZE)

    rows = []
    for symbol in ALPHABET:
        row = process_symbol(symbol, font)
        rows.append(row)
        print(f"[OK] {symbol} ({unicodedata.name(symbol, '')})")

    save_features_csv(rows, CSV_PATH)

    print()
    print("Готово.")
    print(f"Символы:   {SYMBOLS_DIR.resolve()}")
    print(f"Профили:   {PROFILES_DIR.resolve()}")
    print(f"CSV:       {CSV_PATH.resolve()}")
    print(f"Алфавит:   {ALPHABET_PATH.resolve()}")


if __name__ == "__main__":
    main()