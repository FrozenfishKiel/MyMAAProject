@echo off
chcp 65001 > /dev/null
title [RL] MaAutomaton Training
cd /d D:\BiShe\MaAutomaton-main\MaAutomaton-main

echo ==========================================
echo Starting MaAutomaton RL Training Script
echo ==========================================
echo.
echo Activating anaconda environment...
call D:\Anaconda3\Scripts\activate.bat D:\Anaconda3\envs\maa-auto

echo Starting YOLO Monitor...
start "YOLO_Monitor" cmd /c "title YOLO_Monitor && python tests\test_device_and_vision.py --monitor-only"

echo Running training...
python src\rl-training\train.py

echo.
echo Training process finished or exited.
echo Closing YOLO Monitor...
taskkill /FI "WINDOWTITLE eq YOLO_Monitor*" /T /F >nul 2>&1

pause
