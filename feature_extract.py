import cv2
import numpy

# Used only for GUI
import cv

def get_image(image_path):
	return cv2.imread(image_path, cv2.CV_LOAD_IMAGE_GRAYSCALE)

def get_image_features(image):
	# Workadound for missing interfaces
	surf = cv2.FeatureDetector_create("SURF")
	surf.setInt("hessianThreshold", 60)
	surf_extractor = cv2.DescriptorExtractor_create("SURF")

	# Detect keypoints on the image
	keypoints = surf.detect(image, None)

	# Compute descriptors for passed keypoints
	keypoints, descriptors = surf_extractor.compute(image, keypoints)
	return keypoints, numpy.array(descriptors)

if __name__ == "__main__":
	print "OpenCV Demo, OpenCV version " + cv2.__version__
	image = get_image("tst2.jpg")
	image = cv2.resize(image, (600, 318))

	keypoints, descriptors = get_image_features(image)

	# Draw keypoints on image
	for keypoint in keypoints:
		cv2.circle(image, (int(keypoint.pt[0]), int(keypoint.pt[1])), 2, (255, 0, 0))

	cv.NamedWindow("Car")
	cv.ShowImage("Car", cv.fromarray(image))
	cv.WaitKey(0)