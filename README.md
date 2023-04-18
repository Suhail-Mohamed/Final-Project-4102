This project looks to explore the idea of performing a facemask removal from an image of a person by taking a maskless image of the person as the second input. The program uses the techniques of finding facial landmarks on a person's face, homography and warp perspective to align the maskless and masked images, then uses a convex hull to “cut out” the maskless face portion that is covered by the mask, and then uses Delaunay triangulation to perform the face swap of the images. In the end, the program produces a result image that removes the facemask from the masked image. Overall, the project works well to achieve the main goal, however, further improvements could be made for colour correction and working with images that have extreme differences in perspectives. 

How to Run:
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

