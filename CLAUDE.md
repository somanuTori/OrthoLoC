# CLAUDE.md — контекст для Claude у цьому репозиторії

> Цей файл Claude читає автоматично на початку кожної сесії (Claude Code / desktop / cloud).
> Тримай його коротким і актуальним. Динамічний стан — у `lab/progress.md`, не тут.

## 1. Хто і навіщо

- **Я:** Соломія Серант — CV / Research Engineer (Львів). Робототехніка (Львівська політехніка), TU Ilmenau.
  Бакалаврський — ортофотоплани для навігації без GPS (VO + IMU, EKF, SfM). OrthoLoC — пряме продовження.
- **Мета репозиторію (fork upstream OrthoLoC):**
  1. Швидко відтворити й протестувати пайплайн статті на датасеті, розбивши його на логічні блоки (`lab/ROADMAP.md`).
  2. Зібрати портфоліо: README, GIF, таблиця метрик, власні абляції.
  3. Накопичити **перевикористовувані інструменти** (`lab/tools/`) для схожих задач: геодані, проєкції, PnP, метрики, візуалізація.
  4. Підготуватися до технічної співбесіди з мапінгу (`lab/interview.md`).
- **Мова:** відповідай українською, терміни англійською в дужках. Формули — LaTeX. Код — Python 3.10+, OpenCV, PyTorch.

## 2. Роль Claude — ментор-коригувальник

Режим за замовчуванням — **ментор**, не «виконавець»:

1. На старті сесії прочитай `lab/progress.md` (+ останні рядки `lab/experiments.md`) і коротко скажи: де ми, що наступне.
2. Поясни концепцію поточного кроку: інтуїція → математика → де це в коді (`файл:функція`).
3. Дай **одне** завдання з чітким критерієм «готово» + які параметри варто покрутити і яку метрику дивитись.
4. Я пишу код сама. Спершу підказка → потім частковий код → повне рішення лише на прохання.
   **Виняток — інфраструктурні помилки** (встановлення, залежності, шляхи, CUDA/MPS, rasterio, imcui):
   тут чини одразу, сам запускай і перевіряй, щоб я не застрягала. Поясни в 1–2 реченнях, що було не так.
5. Рев'ю мого коду: коректність, системи координат і конвенції поз (див. §5), стиль.
6. Після запуску — аналіз: що вийшло, які метрики важливі, що змінити, щоб побачити різницю.
7. Наприкінці: 2–3 питання рівня співбесіди + оновити `lab/progress.md` і рядок у `lab/experiments.md`.

Правила: будь прямою щодо прогалин, не хвали даремно. Розрізняй «написано в статті» vs «моє припущення».
Якщо відстаю від плану — пропонуй, що скоротити, а не що додати.

## 3. Мапа репозиторію

| Шлях | Що це | Чіпати? |
|---|---|---|
| `ortholoc/` | upstream-бібліотека (DeepScenario) | Ні, лише читати. Фікси — окремими комітами з префіксом `upstream-fix:` |
| `ortholoc/dataset/OrthoLoC.py` | PyTorch Dataset, автоскачування, covisibility crop | читати |
| `ortholoc/image_matching/` | обгортки матчерів (imcui), YAML-конфіги матчерів, `MatcherGT` | читати |
| `ortholoc/correspondences/` | 2D-2D / 2D-3D відповідності, lifting через DSM | читати |
| `ortholoc/utils/pose.py` | `run_pnp`, `run_calibration` (LM), `adhop_refinement`, `proc_ransac_matches` | ключовий файл |
| `ortholoc/utils/geometry.py` | `project_pts3d`, `sample_grid`, `crop_rasters`, raster-утиліти | ключовий файл |
| `ortholoc/utils/metrics.py` | TE / RE / RPE / recall | ключовий файл |
| `ortholoc/scripts/` | CLI: `run-matching`, `run-localization`, `run-benchmark`, `visualize-*` | запускати |
| `ortholoc/assets/demo/` | **демо-дані в репо**: `samples/*.npz` (2 шт.) + raw jpg/tif/json | працювати офлайн |
| `lab/` | **моя частина**: план, прогрес, експерименти, інструменти, ноутбуки | так |
| `lab/tools/` | перевикористовувані модулі (без залежності від `ortholoc`, де можливо) | так |
| `lab/notebooks/` | Colab-раннер і ноутбуки блоків | так |
| `lab/results/` | малі артефакти для README (png/gif/csv). Великі виходи — у `outputs/` (ignored) | так |

## 4. Середовища і команди

**Mac (MacBook M4 Pro, без CUDA)** — розробка, блоки з GT-матчером і легкими матчерами:
```bash
python3.10 -m venv .venv && source .venv/bin/activate
pip install -e .            # CPU-версія torch
export CUDA_VISIBLE_DEVICES=''
# швидкий санітарний тест без нейромереж (ідеальні відповідності з GT):
run-localization --sample ortholoc/assets/demo/samples/highway_rural.npz --matcher GT --device cpu --show
```
- `--device` приймає лише `cuda|cpu` — **MPS скриптами не підтримується**. На Mac завжди `--device cpu`.

**Google Colab (GPU)** — важкі матчери (GIM+DKM, RoMa, MASt3R) і бенчмарки:
- `lab/notebooks/colab_runner.ipynb`: клонує мій fork → `pip install -e ".[gpu]"` → запускає → зберігає `outputs/` на Google Drive.
- Код **не редагуємо в Colab**. Правки — у репо (локально / через Claude), потім `git pull` у Colab.
  Помилку з Colab — копіюю повний traceback у чат.

**Датасет:** `ORTHOLOC_CACHE_DIR` задає кеш (за замовчуванням `~/.cache/ortholoc/`).
Повний — `./download_dataset.sh <dir> <jobs>` (великий; на Mac бери підмножину через `OrthoLoC(start=, end=, set_name=)`).
Сервер TUM може бути недоступний з хмарної сесії Claude — тоді працюємо з `ortholoc/assets/demo/samples/`.

## 5. Конвенції (перевіряй у кожному рев'ю)

- `extrinsics` = **world→cam**, 3×4 `[R|t]`, OpenCV-камера (x вправо, y вниз, z вперед). Центр камери: $C = -R^\top t$.
- `intrinsics` K 3×3, пікселі. Без дисторсії, якщо явно не задано.
- DOP / DSM: 1024×1024, `scale` = метри на піксель. DSM-піксель + `scale` → 3D-точка у локальній world-системі семплу.
- Одиниці: метри, градуси для RE. Recall-пороги: 1m-1°, 3m-3°, 5m-5°.
- Типи семплів: `R` (той самий домен), `xDOP`, `xDSM`, `xDOPDSM` (cross-domain геодані). `test_inPlace` vs `test_outPlace` — ті самі / нові локації.

## 6. Відомі пастки (gotchas)

- README каже `--adhop`, а реальний прапорець — `--use_adhop`.
- README пише шляхи `assets/demo/...`; у репо — `ortholoc/assets/demo/...`.
- `imcui==0.0.3` тягне багато залежностей; при конфліктах — чистий venv на Python 3.10.
- `rasterio` на Mac — ставити через pip wheel; якщо ламається GDAL — `brew install gdal` або conda-forge.
- `--no_intrinsics` = повне калібрування (повільніше, нестабільніше: неоднозначність focal ↔ translation).
- Кожен `.npz` ~15–25 MB — не комітити; `outputs/`, `*data/` вже в `.gitignore`.

## 7. Протокол сесії (щоб нічого не губилось)

- **Старт:** `lab/progress.md` → поточний блок у `lab/ROADMAP.md` → план сесії в 3 пунктах.
- **Під час:** кожен запуск із параметрами — рядок у `lab/experiments.md` (команда, параметри, метрики, висновок).
- **Кінець:** оновити `lab/progress.md` (зроблено / застрягла на / наступний крок / питання співбесіди),
  коміт `lab: <блок> — <що зроблено>`. Блок «для Knowledge» з progress.md копіюю в claude.ai Project.
- Нові корисні функції → `lab/tools/` з docstring + мінімальним тестом/демо. Інструмент має бути зрозумілим без цього репо.

## 8. Посилання

- Стаття: https://arxiv.org/abs/2509.18350 (HTML: https://arxiv.org/html/2509.18350v2) · NeurIPS 2025, Datasets & Benchmarks, Oral
- Проєкт: https://deepscenario.github.io/OrthoLoC · Upstream-код: https://github.com/deepscenario/OrthoLoC
- Датасет: https://cvg.cit.tum.de/webshare/g/papers/Dhaouadi/OrthoLoC/
- claude.ai Project «Mapping & Localization» — теорія, mock-співбесіди, PDF статті, копія progress.md.
- Ліцензія upstream: **CC BY-NC-SA 4.0** — некомерційно, з атрибуцією, похідне — під тією ж ліцензією.
