# SdfTask
Task 1 (Signed Distance Fields). Part of Forward and Inverse Rendering (FAIR) course, CMC MSU, Spring 2025.

## Build
Install SDF (Ubuntu 22)

    sudo apt-get install libsdl2-2.0-0

- For installations on other platforms see https://wiki.libsdl.org/

Build the executable:

    cmake -B build && cmake --build build

## Execute

Primitives rendering (A1, A4, A5):

    ./render render primitives

Naive mesh rendering (A2):

    ./render render mesh_naive "input.obj"

    ./render render mesh_naive cube.obj

SDFGrid rendering (B1):

    ./render render grid "input.grid"

    ./render render grid example_grid.grid

## Rendering Camera Movement
- WASD - move around
- QE - go up/down
- MOUSE - look around

## Tasks Done:
1. A1. Рендер плоскости, заданной уравнением
2. A2. Рендер треугольного меша (наивный)
3. A4. Тени
4. A5. Зеркальные отражения
5. B1. Рендер модели, представленной SDFGrid
6. B2. Построение регулярной сетки по мешу