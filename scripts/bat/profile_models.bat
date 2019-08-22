SET root=%~dp0..\..

FOR /R %root%\models\%precision% %%M IN (*.xml) DO (
    FOR %%A IN (sync, async) DO (
      python %root%\scripts\python\benchmark.py --model %%M --device %1^
      --batch_size 32 -f .\benchmarks.csv --num_infer_requests 2 --api %%A
    )
)
