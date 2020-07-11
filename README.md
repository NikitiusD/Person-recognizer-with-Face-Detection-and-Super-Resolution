# **Person verification with Super-Resolution**

# Introduction
Let's assume that there are 2 photos of people in front of us and we need to determine if the same person is depicted in the photos or not. Also suppose that we have an algorithm for verifying the person which answers this question.
# Problem
We want to find out if it is possible to improve the quality of Person Verification with a preliminary application of Super-Resolution.  

Thus, **we want to confirm the hypothesis of the possibility of improving the quality of Person Verification with a preliminary application of Super-Resolution**.

Since we don't know which metric is the best to use and this is a binary classification problem (1 - the same person, 0 - different persons), we take `Accuracy`, `Precision`, `Recall`, `F1` and entire `Confusion Matrix` metrics just in case.

# The main part
We took [Labeled Faces in the Wild](http://vis-www.cs.umass.edu/lfw/) dataset for our experiments. Then we prepared 834 pairs of faces of the same person, and 834 pairs of different persons. A total of 1 668 pairs - we thought that would be enough for a primary research with our computational resources constrains.

Then, we tested the verification algorithm on the obtained data, saved the results, and began to search through all of the available pre-trained Super-Resolution algorithms. We managed to reproduce and run some of them but many failed for various reasons.

After obtaining upscaled images, we applied the verification algorithm and inserted the metrics' values into the benchmark.

Since we did not know which metric is the most important, we introduced a new one - `Rank`, which shows `Accuracy`, `Precision` and `Recall` together and suitable for our purpose. It equals the sum of indices of `Accuracy`, `Precision` and `Recall` when sorting by them.

# Result
Following benchmark clearly confirms the hypothesis:  
![Drag Racing](benchmark.jpg)

In fact, the metrics values are very interesting for further research.  
For a more complete research, a larger dataset and more models trained specifically on people's faces are needed.  

*There are no resulting images in the repository, if necessary, you can restore most of the results using [this](https://drive.google.com/drive/folders/1dgNBfAv1VsdX-TtKKnOD3jkf5EZuMXAu?usp=sharing), and then notebook's (`Whole pipeline.ipynb` and `Files fixes.ipynb`) from the repository.*

# Papers and repositories used in research (only those that gave the result)
- https://github.com/ipazc/mtcnn <- https://arxiv.org/abs/1604.02878  
- https://github.com/jiny2001/dcscn-super-resolution <- https://arxiv.org/abs/1707.05425  
- https://github.com/thstkdgus35/EDSR-PyTorch <- https://arxiv.org/abs/1707.02921  
- https://github.com/alterzero/DBPN-Pytorch <- https://arxiv.org/abs/1803.02735  
- https://github.com/titu1994/Image-Super-Resolution <- https://arxiv.org/abs/1501.00092  
- https://github.com/fperazzi/proSR <- https://arxiv.org/abs/1804.02900  
- https://github.com/Paper99/SRFBN_CVPR19 <- https://arxiv.org/abs/1903.09814  
- https://github.com/xinntao/ESRGAN

> Thanks to [papers with code](https://paperswithcode.com/task/image-super-resolution).
# Team members
> [Nikita Detkov](https://github.com/NikitiusD), [Ilya Liyasov](https://github.com/Literman), Tanya Vasilyeva.
