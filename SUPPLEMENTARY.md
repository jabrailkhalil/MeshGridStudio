# Дополнительные материалы к статье

Архив `article-supplementary-materials.zip` содержит исходный текст
статьи, программную реализацию трёх методов, автоматические тесты,
воспроизводимые результаты эксперимента и исходные теоретические
материалы.

## Состав

- `article.tex`, `mesh_methods.py`, `mesh_methods_3d.py`, `mesh_gui.py` и
  `mesh_gui_model.py` — статья и программная реализация 2D- и 3D-методов;
- `test_mesh_methods.py`, `test_mesh_methods_3d.py` и `test_mesh_gui.py` —
  43 автоматических теста;
- `requirements*.txt`, `build_exe.ps1`, `build_installer.ps1` и
  `mesh_grid_studio_version.txt` — зависимости и сборка приложения;
- `verify_generated.py` — сравнение повторного расчёта с опубликованными
  научными показателями по явно заданному допуску;
- `output/generated/` — CSV, JSON, таблицы и рисунки основной серии (2D и
  3D), а также отдельный контрольный расчёт при `mu=0`;
- `output/pdf/article.pdf` — собранная статья;
- `docs/ui-preview.png` и `docs/ui-preview-3d.png` — изображения интерфейса;
- `main.tex` и `tishkin_grid.tex` — сохранённые исходные теоретические
  материалы;
- `README.md`, `AUDIT.md` и `SUPPLEMENTARY.md` — описание проекта, технический
  аудит и настоящая инструкция.

## Воспроизведение

```powershell
python -m pip install -r requirements.txt
python -m unittest -v test_mesh_methods.py test_mesh_methods_3d.py test_mesh_gui.py
python mesh_methods.py --output-dir output/reproduced
python mesh_methods.py --adaptive-mu0-control --output-dir output/reproduced
python mesh_methods_3d.py --output-dir output/reproduced
python verify_generated.py output/generated output/reproduced
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=output/pdf article.tex
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=output/pdf article.tex
```

Для сборки приложения:

```powershell
python -m pip install -r requirements-build.txt
powershell -ExecutionPolicy Bypass -File build_exe.ps1
```
