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

echo Running training...
python src\rl-training\train.py

echo.
echo Training process finished or exited.
pause
