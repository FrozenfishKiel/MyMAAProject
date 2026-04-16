@echo off
title [RL] TensorBoard Dashboard

echo ===================================================
echo   Starting TensorBoard in maa-auto environment...
echo ===================================================
echo.

start "" http://localhost:6006/
D:\Anaconda3\envs\maa-auto\python.exe -m tensorboard.main --logdir=D:\BiShe\MaAutomaton-main\MaAutomaton-main\tensorboard_logs

echo.
echo ===================================================
echo   TensorBoard server stopped (Ctrl+C).
echo ===================================================
pause
