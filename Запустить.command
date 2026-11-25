#!/bin/zsh
set -e
cd "$(dirname "$0")"
export MPLCONFIGDIR="$PWD/.cache/matplotlib"
export XDG_CACHE_HOME="$PWD/.cache"
if [[ ! -x .venv/bin/python ]]; then
  echo 'Первый запуск: подготовка Python и библиотек. Понадобится интернет.'
  python3 -m venv .venv
  .venv/bin/python -m pip install -e .
fi
echo 'Открывается программа. Для остановки нажмите Ctrl+C в этом окне.'
.venv/bin/python main.py
