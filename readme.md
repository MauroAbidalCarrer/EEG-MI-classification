## Motor imagery school project
This small school project is an introduction to scikit learn, MNE (python package to process neuroimaging data) and statiscis.  
In this project, we are tasked to implement a dimentionality reduction algorithm.   
I spent a lot of time trying to implement the most commonly used [CSP algorithm](https://en.wikipedia.org/wiki/Common_spatial_pattern).   
While I couldn't implement it al the way from the ground up, I was able to implement the PCA algorithm and transform CSP implementation I found online to make it fit in the scikit-learn pipeline.   
Take a look at the [notebook](source_code/decoding_csp_eeg.ipynb), the [CSP](source_code/my_CSP.py) and [PCA](source_code/my_PCA.py) implementations.   
