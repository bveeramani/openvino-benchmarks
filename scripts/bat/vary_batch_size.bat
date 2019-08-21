@ECHO OFF

SET root=%~dp0..\..

IF "%1"=="MYRIAD" (
    SET precision=fp16
) ELSE (
    IF "%1"=="CPU" (
        SET precision=fp32
    ) ELSE (
        ECHO Expected CPU or Myriad as an argument.
        EXIT /b 9
    )
)

echo Using %1 device.

FOR /R %root%\models\%precision% %%M IN (*.xml) DO (
    FOR %%B IN (1, 2, 4, 8, 16, 32, 64) DO (
      python %root%\scripts\python\benchmark.py --model %%M --device %1^
      --batch_size %%B -f .\benchmarks.csv --num_infer_requests 2 --api async
    )
)
