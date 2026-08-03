# Дополнительные материалы к статье

Архив `article-supplementary-materials.zip` содержит исходный текст
статьи, программную реализацию трёх методов, автоматические тесты,
воспроизводимые результаты эксперимента и исходные теоретические
материалы. Структура архива совпадает со структурой корня проекта:
после распаковки команды ниже выполняются без переноса файлов и
исправления путей.

## Состав

- `article.tex`, `mesh_methods.py`, `mesh_gui.py`,
  `mesh_gui_model.py` — статья и программная реализация;
- `test_mesh_methods.py`, `test_mesh_gui.py` — 30 автоматических тестов;
- `requirements*.txt`, `build_exe.ps1`,
  `mesh_grid_studio_version.txt` — зависимости и сборка приложения;
- `output/generated/` — CSV, JSON, таблица и рисунки основной серии,
  а также отдельный контрольный расчёт при `mu=0`;
- `output/pdf/article.pdf` — собранная статья;
- `output/app/ui-preview-fast-exe.png`, `docs/ui-preview.png` —
  изображения интерфейса, используемые статьёй и README;
- `tishkin_grid.tex` — сохранившийся фрагмент материалов,
  атрибутированный В. Ф. Тишкину в исходной работе;
- `README.md`, `AUDIT.md`, `SUPPLEMENTARY.md` — описание проекта,
  технический аудит и настоящая инструкция.

## Воспроизведение из распакованного архива

Откройте PowerShell в корне распакованного архива, где находится
`mesh_methods.py`, и выполните:

```powershell
python -m pip install -r requirements.txt
python -m unittest -v test_mesh_methods.py test_mesh_gui.py
python mesh_methods.py --output-dir output/generated
python mesh_methods.py --adaptive-mu0-control --output-dir output/generated
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=output/pdf article.tex
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=output/pdf article.tex
```

Для PDF нужен pdfLaTeX с пакетами из преамбулы и масштабируемыми
кириллическими шрифтами `cm-super`. Команда `pdffonts
output/pdf/article.pdf` не должна показывать шрифты Type 3.

Для сборки приложения:

```powershell
python -m pip install -r requirements-build.txt
powershell -ExecutionPolicy Bypass -File build_exe.ps1
output/app/MeshGridStudio/MeshGridStudio.exe --solver-self-test
output/app/MeshGridStudio/MeshGridStudio.exe --smoke-test
```

## Повторная упаковка

В рабочем проекте архив создаётся из той же корневой структуры:

```powershell
powershell -ExecutionPolicy Bypass -File build_supplement.ps1
```
