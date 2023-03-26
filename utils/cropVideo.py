import cv2

# Read input video file
cap = cv2.VideoCapture('../data/video/001.mp4')

# Define cropping parameters
top_crop = 150
bottom_crop = 150
left_crop = 0
right_crop = 0

# Get input video dimensions
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Calculate output video dimensions after cropping
out_width = width - left_crop - right_crop
out_height = height - top_crop - bottom_crop

# Create output video writer object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('../data/video/cropped_1.mp4', fourcc, 30, (out_width, out_height))

# Process each frame of the input video
while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        # Crop the frame using the defined parameters
        cropped_frame = frame[top_crop:height-bottom_crop, left_crop:width-right_crop]
        # Write the cropped frame to the output video file
        out.write(cropped_frame)
    else:
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
