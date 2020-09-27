
# installation

this program requires python3.

```sh
python3 -m venv myenv
source myenv/bin/activate
pip install -r requirements.txt
```

# run
```sh
winsize=(
	5
	9
	13
	21
	31
	51
	)
for iimg in $(seq 1 5) ; do
	for ws in "${winsize[@]}" ; do
		python main.py -imgpath ./textures_simple/T${iimg}.gif -winsize "$ws" -saveprogress="T${iimg}_win${ws}" -savepath "T${iimg}_win${ws}.png"
	done
done

# or
python main.py -imgpath ./textures_simple/T1.gif -winsize 31 -saveprogress='T1' -savepath T1_new.png
python main.py -imgpath ./textures_simple/T2.gif -winsize 31 -saveprogress='T2' -savepath T2_new.png
python main.py -imgpath ./textures_simple/T3.gif -winsize 31 -saveprogress='T3' -savepath T3_new.png
python main.py -imgpath ./textures_simple/T4.gif -winsize 31 -saveprogress='T4' -savepath T4_new.png
python main.py -imgpath ./textures_simple/T5.gif -winsize 31 -saveprogress='T5' -savepath T5_new.png

# other examples
python main.py -imgpath ./textures/1.1.01_small.tiff -winsize 31 -saveprogress='1.1.01_small__31' -savepath 1.1.01_new.png
python main.py -imgpath ./textures/1.1.10_small.tiff -winsize 31 -saveprogress='1.1.10_small__31' -savepath 1.1.10_new.png
python main.py -imgpath ./textures/1.4.05_small.tiff -winsize 31 -saveprogress='1.4.05_small__31' -savepath 1.4.05_new.png
python main.py -imgpath ./textures/1.4.07_small.tiff -winsize 31 -saveprogress='1.4.07_small__31' -savepath 1.4.07_new.png
