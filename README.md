# ss_image_enhancement
1. Enter the dist directory, and then execute the command in the terminal.     
```
pip install ContrastEnhancement-0.0.1-py2.py3-none-any.whl
```


2. If you wanna test the algorithm, you can enter this command as follows:      
```
python test.py -name *.jpg -gamma 0.00
```

3. The original images are shown on the left, and the enhanced image on the right.    

|      Raw Image           |        Enhanced Image   |
| :-----------------: | :--------------: |
| ![](data/1.jpg) | ![](data/1_secedct.jpg)|
| ![](data/4.jpg) | ![](data/4_secedct.jpg)|
| ![](data/7.jpg) | ![](data/7_secedct.jpg)|

4. QRCM: This measure considers both the level of relative contrast enhancement between input and output images and distortions resulting from the enhancement process. The measure produces a number in the range [−1, 1] where -1 and 1 refer to full level of contrast degradation and improvement, respectively.

5. References  
[1] Spatial Entropy-Based Global and Local Image Contrast Enhancement         
[2] Spatial Mutual Information and PageRank-Based Contrast Enhancement and Quality-Aware Relative Contrast Measure    










