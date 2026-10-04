@echo off
rem sauerbraten with the path tracer. settings/saves go to "My Games\Sauerbraten" like normal sauer
cd /d "%~dp0"
start "" "%~dp0bin64\sauerbraten.exe" "-q$HOME\My Games\Sauerbraten" -glog.txt %*
