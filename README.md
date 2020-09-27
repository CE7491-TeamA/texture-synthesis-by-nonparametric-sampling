
# installation

this program requires python3.

```sh
python3 -m venv myenv
source myenv/bin/activate
pip install -r requirements.txt
```

# run
```sh
python main.py -imgpath ./textures_simple/T1.gif -winsize 31 -saveprogress='T1' -savepath T1_new.png
python main.py -imgpath ./textures_simple/T2.gif -winsize 31 -saveprogress='T2' -savepath T2_new.png
python main.py -imgpath ./textures_simple/T3.gif -winsize 31 -saveprogress='T3' -savepath T3_new.png
python main.py -imgpath ./textures_simple/T4.gif -winsize 31 -saveprogress='T4' -savepath T4_new.png
python main.py -imgpath ./textures_simple/T5.gif -winsize 31 -saveprogress='T5' -savepath T5_new.png
python main.py -imgpath ./textures/1.1.01_small.tiff -winsize 31 -saveprogress='1.1.01_small__31' -savepath 1.1.01_new.png
python main.py -imgpath ./textures/1.1.10_small.tiff -winsize 31 -saveprogress='1.1.10_small__31' -savepath 1.1.10_new.png
python main.py -imgpath ./textures/1.4.05_small.tiff -winsize 31 -saveprogress='1.4.05_small__31' -savepath 1.4.05_new.png
python main.py -imgpath ./textures/1.4.07_small.tiff -winsize 31 -saveprogress='1.4.07_small__31' -savepath 1.4.07_new.png
