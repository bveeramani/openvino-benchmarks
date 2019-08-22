@ECHO OFF

SET root=%~dp0..\..

FOR /R %root%\models\fp16 %%M IN (*.xml) DO (
    python %root%\scripts\python\profile.py --model %%M -f .\profiles.csv
)

FOR /R %root%\models\fp32 %%M IN (*.xml) DO (
    python %root%\scripts\python\profile.py --model %%M -f .\profiles.csv
)
