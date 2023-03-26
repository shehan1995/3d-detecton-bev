
import cv2

# Replace 'video.mp4' with the path to your video file
cap = cv2.VideoCapture('../data/video/accident/007.mp4')

# Open a text file for writing
with open('../data/video/accident/007.txt', 'w') as f:
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            # Get the current frame number and time in milliseconds
            frame_num = cap.get(cv2.CAP_PROP_POS_FRAMES)
            time_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            # Write the frame number and time to the text file
            f.write(f'Frame: {frame_num}, Time (ms): {time_ms}\n')
            print('Frame:', frame_num, 'Time (ms):', time_ms)

            cv2.imshow('frame', frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break

# Release the video capture and close the text file
cap.release()
cv2.destroyAllWindows()







