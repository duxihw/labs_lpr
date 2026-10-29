#!/usr/bin/env bash
# Сборка отчётов лабораторных работ.
#
#   .github/scripts/build_reports.sh [ПАПКА ...]
#
# Без аргументов собирает все папки вида x.x.x с report/main.tex.
# Для каждой работы:
#   1) если есть scripts/process_data.py -- запускает его
#      (пересчитывает data/processed/ из data/raw/);
#   2) компилирует report/main.tex и кладёт результат в report/<номер>.pdf
#      (например, 3.4.5 -> 3.4.5/report/345.pdf).
# Ошибка в одной работе не останавливает остальные; код возврата != 0,
# если хоть одна работа не собралась.

set -uo pipefail
cd "$(git rev-parse --show-toplevel)"

if [ "$#" -gt 0 ]; then
    labs=("$@")
else
    mapfile -t labs < <(ls -d [0-9]*.[0-9]*.[0-9]* 2>/dev/null)
fi

failed=()
for lab in "${labs[@]}"; do
    lab="${lab%/}"
    [ -f "$lab/report/main.tex" ] || { echo "-- $lab: нет report/main.tex, пропуск"; continue; }
    echo "::group::$lab"
    ok=1

    if [ -f "$lab/scripts/process_data.py" ]; then
        echo "-- $lab: обработка данных"
        python "$lab/scripts/process_data.py" || { echo "!! $lab: process_data.py упал"; ok=0; }
    fi

    if [ "$ok" = 1 ]; then
        echo "-- $lab: сборка LaTeX"
        pdf="${lab//./}.pdf"
        (
            cd "$lab/report" &&
            latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex >/dev/null &&
            cp main.pdf "$pdf"
        ) || {
            echo "!! $lab: LaTeX не собрался, ошибки из лога:"
            grep -A5 "^!" "$lab/report/main.log" 2>/dev/null | head -30
            ok=0
        }
        # main.pdf -- промежуточный файл; удаляем, если он не хранится в git
        (cd "$lab/report" && latexmk -c >/dev/null 2>&1
         git ls-files --error-unmatch main.pdf >/dev/null 2>&1 || rm -f main.pdf)
    fi

    [ "$ok" = 1 ] && echo "-- $lab: OK" || failed+=("$lab")
    echo "::endgroup::"
done

if [ "${#failed[@]}" -gt 0 ]; then
    echo "Не собрались: ${failed[*]}"
    exit 1
fi
echo "Все отчёты собраны."
