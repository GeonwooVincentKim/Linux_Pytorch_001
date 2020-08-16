# Linux_Pytorch_001
Using Linux to make Pytorch Project with Jupyter Notebook.

# How to execute this file without using GPU
1. When you trying to input in "device" Parameter..
   - 1. torch.tensor([], device="cuda:0" if torch.cuda.is_available() else "cpu")
   - 2. torch.tensor([[1, 2], [3, 4.]], device="cuda:0" if torch.cuda.is_available() else "cpu")

2. When you trying to input in "to" Parameter..
   - 3. to("cuda:0" if torch.cuda.is_available() else "cpu")
   - 4. torch.zeros(100, 10).to("cuda:0" if torch.cuda.is_available() else "cpu")
  
# Prepare LFW Data size as 250 by 250.
## How are the Image-sizes configures and replaced with Batch-size?
1. How are the Image-sizes configues? (There are some comments contained batch-size.)
   - 1. The meaning of batch-size are use for divide Image data 
        how many data you want to use.

   - 2. For example, There is a Image that sizes are 784 
        consists width and height as 28. 

   - 3. Therefore, when you multiple width and height which have 
        same values both of them, the result will became 784.

2. Replacing Image-sizes with Batch-Size.
   - 1. Batch-Size is applied to Image-Data-Division to classify
        images and training images easier.

   - 2. For example, There is a Image that sizes are 784
        consists width and height as 28.

   - 3. Therefore, we can set Batch-Size at least bigger 
        than one-lines, which means the Batch-Size should 
        become two-lines (56).    
