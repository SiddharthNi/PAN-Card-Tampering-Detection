# PAN-Card-Tampering-Detection

An image processing tampering detection model using OpenCV to detect whether the given id is valid or not. For this project, the Parmanent Account Number has been taken but it can be used in different organizations for the verification of their ids.
The aim of this project is to utilize the computer vision applications in the fake identity detection.

## Purpose
The purpose of this project is to detect tampering/fraud of PAN cards using computer vision. This project will help the different organizations in detecting whether the Id i.e. the PAN card provided to them by their employees or customers or anyone is original or not.
For this project we will calculate the structural similarity of the original PAN card and the PAN card uploaded by the user – This is the soul of this project we will discuss it thoroughly later in this blog.
Similarly in this project with the help of image processing involving the techniques of computer vision we are going to detect that whether the given image of the PAN card is original or tampered (fake) PAN card.



## What is Computer Vision?

Computer vision is a field of artificial intelligence that trains computers to interpret and understand the visual world. Using digital images from cameras and videos and deep learning models, machines can accurately identify and classify objects — and then react to what they “see”.

Computer Vision in Image processing is mainly focused on processing the raw input images to enhance them or preparing them to do other tasks. Computer vision is focused on extracting information from the input images or videos to have a proper understanding of them to predict the visual input like the human brain

## Usage
 The Jupyter notebook Pan_Card_Tampering_Detection.ipynb contains the code for loading and preprocessing the dataset, as well as implementing and evaluating the tampered as well as original image. To run the notebook, simply open it in Jupyter or Google Colab and run each cell in order.
## Results
Structural Similarity Index (SSIM) between the tampered and original image is 31.67%.
## Authors

- [@SiddharthNi](https://github.com/SiddharthNi)


## The steps involved in this project are as follows :-
1. Import necessarylibraries
2. Scraping the tampered and original pan card from the website
3. Scaling down the shape of the tampered image as the original image
4. Read original and tampered image
5. Converting an image into a grayscale image
6. Applying Structural Similarity Index (SSIM) technique between the two images
7. Calculate Threshold and contours and
8. Experience real-time contours and threshold on images

## meaning of library :-
1.	Skimage: Scikit-image, or ski-mage, is an open-source Python package, in this project most of the image processing techniques will be used via scikit-image
2.	imutils: Imutils are a series of convenience functions to make basic image processing functions such as translation, rotation, resizing, and displaying images easier with OpenCV.
3.	cv2: OpenCV (Open Source Computer Vision Library) is a library of programming functions. Here in this project major reading and writing of the image are done via cv2.
4.	PIL: PIL (Python Imaging Library) is a free and open-source additional library for the Python programming language that adds support for opening, manipulating, and saving many different image file formats.
### Make folders and sub-folders for storing images, you may create it manually it’s completely up to you (Optional).
 
 !mkdir pan_card_tampering
  
  !mkdir pan_card_tampering/image

Scraping original and tampered PAN card from different sources

    original = Image.open(requests.get('https://www.thestatesman.com/wp-content/uploads/2019/07/pan-card.jpg', stream=True).raw)

    tampered = Image.open(requests.get('https://assets1.cleartax-cdn.com/s/img/20170526124335/Pan4.png', stream=True).raw)

In the above code snippet, we are web scarping the images from different sources using the requests library.
### Loading original and user-provided images:-
As you can see in the above output, The original size of the original image and the original size of tampered image are different which will result in unwanted/false results while doing image processing, that’s why scaling down both the image to equal shape is prominently needed.

## Scraping original and tampered PAN card from different sources :-

    original = Image.open(requests.get('https://www.thestatesman.com/wp-content/uploads/2019/07/pan-card.jpg', stream=True).raw)

    tampered = Image.open(requests.get('https://assets1.cleartax-cdn.com/s/img/20170526124335/Pan4.png', stream=True).raw)

In the above code snippet, we are web scarping the images from different sources using the requests library.
### Loading original and user-provided images :-
As you can see in the above output, The original size of the original image and the original size of tampered image are different which will result in unwanted/false results while doing image processing, that’s why scaling down both the image to equal shape is prominently needed.
Converting the format of a tampered image similar to the original image.
### Resize Image
    original = original.resize((250, 160))

    print(original.size)

    original.save('pan_card_tampering/image/original.png')#Save image

    tampered = tampered.resize((250,160))

    print(tampered.size)

    tampered.save('pan_card_tampering/image/tampered.png')#Saves image

Output :
(250, 160)

(250, 160)

Now, if you will see the output the shape of both the images (Original image and tampered image) is scaled down to equal shape i.e. (250,160). Now the image processing will be smoother and more accurate than it was before.
We can change the format of the image (png or jpg) if needed.
### Change image type if required from png to jpg
    tampered = Image.open('pan_card_tampering/image/tampered.png')

    tampered.save('pan_card_tampering/image/tampered.png')#can do png to jpg

Display original PAN card image which will be used for comparison.


 
## Reading images using OpenCV.

### load the two input images
original = cv2.imread('pan_card_tampering/image/original.png')

tampered = cv2.imread('pan_card_tampering/image/tampered.png')

Now in the above code, we are reading both the images (Original and Tampered) using cv2’s imread() function.

## Convert the images to grayscale
    original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

    tampered_gray = cv2.cvtColor(tampered, cv2.COLOR_BGR2GRAY)

In the above code, we have converted the original images (Original pan card and user given Pan card) to gray-scale images using cv2’s function cvtColor() which have parameter as cv2.COLOR_BGR2GRAY.

### But why we need to convert them into grayscale? 
Here’s the reason why :

Converting images into grayscale is very much beneficial inaccuracy of image processing because in image processing many applications don’t help us in identifying the importance, edges of the colored images also colored images are a bit complex to understand by machine because they have 3 channel while grayscale has only 1 channel.
Applying Structural Similarity Index (SSIM) technique between the two images

## Hold on ! First we need to understand what is SSIM !

###  What is SSIM?
The Structural Similarity Index (SSIM) is a perceptual metric that quantifies the image quality degradation that is caused by processing such as data compression or by losses in data transmission.
### How SSIM perform its function?
This metric is basically a full reference that requires 2 images from the same shot, this means 2 graphically identical images to the human eye. The second image generally is compressed or has a different quality, which is the goal of this index.
### What is the real-world use of SSIM?
SSIM is usually used in the video industry but has as well a strong application in photography.

Become a Full Stack Data Scientist

Transform into an expert and significantly impact the world of data science.
Download Brochure

### How SSIM helps in detection?
SSIM actually measures the perceptual difference between two similar images. It cannot judge which of the two is better: that must be inferred from knowing which is the original one and which has been exposed to additional processing such as compression or filters.

###  Compute the Structural Similarity Index (SSIM) between the two images, 

### ensuring that the difference image is returned

    (score, diff) = structural_similarity(original_gray, tampered_gray, full=True)
    diff = (diff * 255).astype("uint8")

    print("SSIM Score is : {}".format(score*100))

    if score >= 80:

    print ("The given pan card is original")

    else:

    print("The given pan card is tampered")

    Output :

    SSIM Score is : 31.678790332739425

The given pan card is tampered

### Let’s break down what just happened in the above code!
•	Structural similarity index helps us to determine exactly where in terms of x,y coordinates location, the image differences are. Here, we are trying to find similarities between the original and tampered image.

•	The lower the SSIM score lower is the similarity, i.e SSIM score is directly proportional to the similarity between two images

•	We have given one threshold value of “45” i.e if any score is >= 80 it will be regarded as the original pan card else tampered with one.

•	Generally SSIM values 0.97, 0.98, 0.99 for good quallty recontruction techniques.
Experience real-time threshold and contours on images
Contours detection is a process that can be explained simply as a curve joining all the continuous points (along with the boundary), having the same color or intensity. The algorithm does indeed find edges of images but also puts them in a hierarchy.

### Calculating threshold and contours 
    thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cnts = imutils.grab_contours(cnts)

Here we are using the threshold function of computer vision which applies an adaptive threshold to the image which is stored in the form array. This function transforms the grayscale image into a binary image using a mathematical formula. 

Find contours works on binary image and retrieve the contours. These contours are a useful tool for shape analysis and recognition. Grab contours grabs the appropriate value of the contours.





    
## Screenshots
### Orignal 
            ![orignal](https://github.com/SiddharthNi/PAN-Card-Tampering-Detection/assets/116881073/a231fd7e-6713-499d-aa1c-d4586e25469d) 

### Tampering
             ![tempared](https://github.com/SiddharthNi/PAN-Card-Tampering-Detection/assets/116881073/e04b43e9-a927-4eac-b0b6-aa562f386b83)


### Display original image
original

Output :![op1](https://github.com/SiddharthNi/PAN-Card-Tampering-Detection/assets/116881073/75ed59bd-209b-407e-8c58-1b20b11058d8)

 
Display user-provided image which will be compared with PAN card.
### Display user given image
tampered

Output :![0p2](https://github.com/SiddharthNi/PAN-Card-Tampering-Detection/assets/116881073/3d6dccd9-fbc6-4904-a9b1-073c2ff6a3af)


### Difference bw the 2 Images
Difference bw the 2 Images

Output :![op3](https://github.com/SiddharthNi/PAN-Card-Tampering-Detection/assets/116881073/eea912a2-df12-46da-a5c3-0b8d62de2633)


### Threshold Image
Threshold Image

Output :![t4](https://github.com/SiddharthNi/PAN-Card-Tampering-Detection/assets/116881073/58c91c5c-68ae-453f-99b5-b5e77af82954)


## Lessons Learned


1 ) Computer vision:-

 Computer vision is a field of artificial intelligence that trains computers to interpret and understand the visual world. Using digital images from cameras and videos and deep learning models, machines can accurately identify and classify objects — and then react to what they “see”.
Computer Vision in Image processing is mainly focused on processing the raw input images to enhance them or preparing them to do other tasks. Computer vision is focused on extracting information from the input images or videos to have a proper understanding of them to predict the visual input like the human brain

2 ) SSSIM:-

The Structural Similarity Index (SSIM) is a perceptual metric that quantifies the image quality degradation that is caused by processing such as data compression or by losses in data transmission.
This metric is basically a full reference that requires 2 images from the same shot, this means 2 graphically identical images to the human eye. The second image generally is compressed or has a different quality, which is the goal of this index.SSIM is usually used in the video industry but has as well a strong application in photography.

3 ) SSIM function in detection:-

SSIM actually measures the perceptual difference between two similar images. It cannot judge which of the two is better: that must be inferred from knowing which is the original one and which has been exposed to additional processing such as compression or filters.
## Conclusion

In this project, we understood the application of Structural Similarity Index in computer vision for detecting fake id's and shown the contours and threshold on the tampered and original image, also verfied the images with gray scaling. This project could be further applied to tackle the id's large industry,transport sector, educatinal system  dataset for making a tool to verify the pan id.


