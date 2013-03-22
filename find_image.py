import cv2
import numpy
import os
import collections
import operator

# Used for timing
import time

files = []
matcher = None

def get_image(image_path):
	return cv2.imread(image_path, cv2.CV_LOAD_IMAGE_GRAYSCALE)

def get_image_features(image):
	# Workadound for missing interfaces
	surf = cv2.FeatureDetector_create("SURF")
	surf.setInt("hessianThreshold", 100)
	surf_extractor = cv2.DescriptorExtractor_create("SURF")
	# Get keypoints from image
	keypoints = surf.detect(image, None)
	# Get keypoint descriptors for found keypoints
	keypoints, descriptors = surf_extractor.compute(image, keypoints)
	return keypoints, numpy.array(descriptors)

def train_index():
	# Prepare FLANN matcher
	flann_params = dict(algorithm = 1, trees = 4)
	matcher = cv2.FlannBasedMatcher(flann_params, {})

	# Train FLANN matcher with descriptors of all images
	for f in os.listdir("img/"):
		print "Processing " + f
		image = get_image("./img/%s" % (f,))
		keypoints, descriptors = get_image_features(image)
		matcher.add([descriptors])
		files.append(f)

	print "Training FLANN."
	matcher.train()
	print "Done."
	return matcher

def match_image(index, image):
	# Get image descriptors
	image = get_image(image)
	keypoints, descriptors = get_image_features(image)

	# Find 2 closest matches for each descriptor in image
	matches = index.knnMatch(descriptors, 2)
	
	# Cound matcher for each image in training set
	print "Counting matches..."
	count_dict = collections.defaultdict(int)
	for match in matches:
		# Only count as "match" if the two closest matches have big enough distance
		if match[0].distance / match[1].distance < 0.3:
			continue

		image_idx = match[0].imgIdx
		count_dict[files[image_idx]] += 1

	# Get image with largest count
	matched_image = max(count_dict.iteritems(), key=operator.itemgetter(1))[0]

	# Show results
	print "Images", files
	print "Counts: ", count_dict
	print "==========="
	print "Hit: ", matched_image
	print "==========="

	return matched_image

if __name__ == "__main__":
	print "OpenCV Demo, OpenCV version " + cv2.__version__
	
	start_time = time.time()
	flann_matcher = train_index()
	print "\nIndex generation took ", (time.time() - start_time), "s.\n"
	# ======================== Training done, image matching here ===============
	
	start_time = time.time()
	match_image(flann_matcher, "tst2.jpg")
	print "Matching took", (time.time() - start_time), "s."
