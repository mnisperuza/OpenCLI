@echo off
REM Device Guard can block Python's generated .exe console launchers.
REM Run the installed local package through the Python interpreter instead.
python -m fenrir_agent.cli %*
