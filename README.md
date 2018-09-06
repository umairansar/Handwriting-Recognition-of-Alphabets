# Handwriting-Recognition-of-Alphabets
Using the famous NOT-MNIST dataset, this model reached the accuracy of 89.2%

### Methodology:
The model was built using the Google's Tensorflow framework. The approach can be broken down as follows:
        
        1- Extraction of images.
        2- Preparing Labels from image folders.
        2- Preprocessing of image data, such as flattening and converting to greyscale.
        3- Preprocessing of label data to encode them using 'Local Binarizer.'
        4- Setting placeholders and function approximator variables.
        5- Running a session to train the data.
        6- Testing accuracy on a single image by getting prediction by model.
        
I have uploaded the code with this repository and you will see the same flow there as described above.

### Challenges:

1) During tweaking with the shapes of the arrays, the shapes were irregular even after reshaping the images numpy array.
2) One Hot Encoding or Label Encoding does not work with 'String' or 'Char' labels.

### Resolution and Soltion:
1) Delete the folders named 'A' and 'F' since they were corrupted and were affecting the shape of train and test arrays of images. (This could be a user specific error or actual error in the dataset by the provider.)
2) Use Local Binarizer instead. This is built specifically to encode string labels.

### Final Outcome:
The model was successfully trained on 14980 images from MNIST dataset, returining an accuracy of 89.2%

### Future Work:
This single object image classification can be revamped to recognize multiple numbers within an image, and this technique is already being used in text recognition softwares.

### Note:
Adding hidden layers and CNN can greatly increase the accuracy of the model. This model is was created while taking simplicity into account.

#### This repository can also be accesed at umuzworld.com
