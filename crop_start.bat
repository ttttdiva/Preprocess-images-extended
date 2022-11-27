set cur_dir=%~dp0

set /p src_dir="INPUT:"

C:
copy %src_dir% %cur_dir%tmp\
timeout /t 0.5
cd %cur_dir%tmp\

setlocal enabledelayedexpansion

set count=0
for %%f in ( * ) do (
  set /a count=!count!+1
  ren %%f !count!.png
  )

endlocal

cd ../
call venv\Scripts\activate.bat
call activate venv
python inference.py --net isnet_is --ckpt isnetis.ckpt --data tmp --out tmp2 --img-size 1024 --only-matted

python WhiteBG.py --src_dir tmp2/ --out_dir input_data/

python face-body-splitter.py input_data/ output_data/

move %cur_dir%input_data\*.png %cur_dir%output_data\

del /q %cur_dir%tmp\
del /q %cur_dir%tmp2\
del /q %cur_dir%input_data\

start %cur_dir%output_data\

echo "-----------完了-----------"
