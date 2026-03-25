@echo off
title [RL] Arknights End-to-End PPO Training

echo ===================================================
echo   Starting maa-auto virtual environment and training...
echo ===================================================
echo.

D:\Anaconda3\envs\maa-auto\python.exe D:\BiShe\MaAutomaton-main\MaAutomaton-main\src\rl-training\train.py

echo.
echo ===================================================
echo   Training finished or interrupted by user (Ctrl+C).
echo ===================================================
pause
