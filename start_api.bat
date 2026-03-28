@echo off
echo Starting Fed-Vis Doctor's Cockpit API...
echo Loading federated_global_best.pth Model...
echo.

set PYTHONPATH=src
python -m fedvis.api.app --checkpoint outputs/federated_global_best.pth

pause
