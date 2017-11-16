csc411proj
===  
Crowd Monitoring: Abnormal Behaviour Detection (In Progress)  
&emsp;This project applied computer vision and pattern recognition methods aimed to detect abnormal behaved object in crowd, such as runner (fast motion) in a crowd of walking (slow motion) people.  
&emsp;&emsp;- Based on: Python, OpenCV, Scikit-Learn and Keras(Tensorflow Backend)  
&emsp;&emsp;- Feature extraction: Morphological filter, Normalisation, Optical Flow  
&emsp;&emsp;- Classification Model: Fisher, Clustering, CNN  
<br>
---
Please keep the file in such structure  
|  
|--original_pics  
| &emsp;&emsp;&emsp;     |--001.tif  
| &emsp;&emsp;&emsp;     |--002.tif  
| &emsp;&emsp;&emsp;     .  
| &emsp;&emsp;&emsp;     .  
| &emsp;&emsp;&emsp;     .  
| &emsp;&emsp;&emsp;     |--200.tif  
|  
|--code  
| &emsp;&emsp;  |--plot_hough_lines.py  
| &emsp;&emsp;  |--weical.py      
|  
|--ref_data  
|  &emsp;&emsp;&emsp; |--hough_lines_only  
|  &emsp;&emsp;&emsp; |--pics_with_hough_lines  
|  &emsp;&emsp;&emsp;&emsp;&emsp;&emsp; |--h_lines_only_001.tif  
|  &emsp;&emsp;&emsp;&emsp;&emsp;&emsp; |--h_lines_only_002.tif  
| &emsp;&emsp;&emsp;  &emsp;&emsp;&emsp;   .  
| &emsp;&emsp;&emsp;  &emsp;&emsp;&emsp;   .  
| &emsp;&emsp;&emsp;  &emsp;&emsp;&emsp;   .  
|  &emsp;&emsp;&emsp;&emsp;&emsp;&emsp; |--h_lines_only_200.tif  
|  
|  
|--README.md  
