#!/usr/bin/env bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
echo "Env ready. TODO: run export and bench scripts."