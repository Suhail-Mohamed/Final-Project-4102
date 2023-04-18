This project looks to explore the idea of performing facemask removal given an image of a person with and without a mask on.  The program uses facial landmark detection, homography, and warp perspective techniques to align the maskless and masked images. Next, it uses a convex hull to extract the lower portion of the face from both the masked and maskless image. Lastly, Delaunay triangulation is performed to swap the mask and maskless images, resulting in an output image with the facemask removed. Overall, the project works well to achieve the main goal, however, further improvements could be made for colour correction and working with images that have extreme differences in perspectives. 

Video Demo showing the program running:  https://youtu.be/FicHZMOU6gw
# How to Run: (Please refer to the video demo if the code doesn't run for you) #
    1: Navigate to the /src directory.
    2. Ensure that the following pip dependancies have been installed (if not use pip install <package name from below>):
        - numpy
        - face_alignment
        - io
        - opencv-python
    3. Ensure that Visual Studio is installed on the computer that you are using (Windows).
    4. Run the code using: python main.py
        - There are two command line args that could be used
            - First Command line arg
                - A number between 1-4 that changes the input images that the program runs on
            - Second Command line arg
                - A number that is used for thresholding for finding the artificial chin point (Manually defaulted to 20)

Examples of commands to run the code:
python main.py 1
python main.py 2 10
python main.py 3
python main.py 4 5
