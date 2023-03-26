import os
import cv2


def saveVideoFrames():
    # Open the video file
    cap = cv2.VideoCapture('data/video/Non_Accident/na_2.mp4')

    # Create a counter variable
    frame_count = 0

    # Create the directory to save the frames
    output_dir = 'data/na/images2'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Loop through the video frames
    while cap.isOpened():
        # Read a frame
        ret, frame = cap.read()

        # Check if the frame was successfully read
        if not ret:
            break

        # Save the frame as an image in the output directory
        filename = os.path.join(output_dir, f"{frame_count}.jpg")
        cv2.imwrite(filename, frame)

        # Increment the counter
        frame_count += 1

    # Release the video capture object and close all windows
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    saveVideoFrames()
