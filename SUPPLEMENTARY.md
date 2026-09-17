# Дополнительные материалы к статье

Архив `article-supplementary-materials.zip` содержит исходный текст
статьи, программную реализацию трёх методов, автоматические тесты,
воспроизводимые результаты эксперимента и исходные теоретические
материалы.

## Состав

- `article/` — `article.tex` и собранный `article.pdf`;
- `source/` — вычислительное ядро (2D и 3D), модель интерфейса и
  Tk-приложение;
- `tests/` — 43 автоматических теста (15 двумерных, 15 трёхмерных и
  13 для модели интерфейса);
- `generated/` — CSV, JSON, таблица и рисунки основной серии (2D и 3D),
  а также отдельный контрольный расчёт при `mu=0`;
- `original-theory/` — неизменённый исходный текст исследования
  `main.tex` и рукопись В. Ф. Тишкина `tishkin_grid.tex`;
- `build/` — зависимости и сценарий сборки Windows-приложения.

## Воспроизведение

```powershell
python -m pip install -r requirements.txt
python -m unittest -v test_mesh_methods.py test_mesh_methods_3d.py test_mesh_gui.py
python mesh_methods.py --output-dir output/generated
python mesh_methods.py --adaptive-mu0-control --output-dir output/generated
python mesh_methods_3d.py --output-dir output/generated
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=output/pdf article.tex
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=output/pdf article.tex
```

Для сборки приложения:

```powershell
python -m pip install -r requirements-build.txt
powershell -ExecutionPolicy Bypass -File build_exe.ps1
```
