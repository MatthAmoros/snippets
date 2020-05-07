import sys
import cv2
import re
import numpy as np
import math
import json
from json import JSONEncoder
import datetime
import glob
import pytesseract


""" Sample job.json
{
	"1":{
			"anchor": {"size": "216", "x":"86", "y":"930", "o":"0", "box":"[[326, 940], [68, 940], [68, 672], [326, 672]]"},
			"roi_box": "[[326, 310], [200, 310], [200, 252], [326, 252]]",
			"type":"OCR", "value":"([:0-9.]{5,})", "label":"CSG", "expected_results":"108124, 93872"
		}
}

"""

""" Requieres : sudo apt-get install libdmtx0a"""
from pylibdmtx.pylibdmtx import decode

""" Thanks to StackOverFlow for this one """
class NumpyArrayEncoder(JSONEncoder):
	def default(self, obj):
		if isinstance(obj, np.ndarray):
				return obj.tolist()
		return JSONEncoder.default(self, obj)

def pre_process_for_DM_heavy(img):
	"""
	Resize and binarize image
	@param img = Grayscale image
	@return : 600x300 binarized image
	"""
	dm_possible = True
	filtered = prepare_image(img)
	filtered = filtered.astype(np.uint8)
	_, bw = cv2.threshold(filtered, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

	return bw, dm_possible

def pre_process_for_OCR(img):
	"""
	Resize and binarize image
	@param img = Grayscale image
	@return : binarized image
	"""

	## Threshold
	blurred = cv2.GaussianBlur(img,(3,3),0.3)
	_, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

	return thresh

def prepare_image(img):
	"""
	Prepare Image for DM detection
	Using algorithm from :
	"Efficient 1D and 2D barcode detection using
	mathematical morphology"
	Melinda Katona and L´aszl´o G. Ny´ul

	@param img = Raw image
	@return Image masked with possible DM postions
	"""
	src = img.copy()

	possible_matches = 0
	""" Cancel noise """
	blurred = cv2.GaussianBlur(src,(3,3),0.3)

	""" Bottom hat filter """
	""" SE size should be 2 x module size to detect silence zone """
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(50,50))
	open =  cv2.morphologyEx(blurred, cv2.MORPH_BLACKHAT, kernel)

	mask, possible_matches = get_mask_for_DM(open, kernel)

	""" Try again with SE rotated """
	angle_step = 30
	current_angle = angle_step
	while possible_matches == 0 and current_angle < 180:
		print("DM Mask not found, rotating SE")
		kernel_rotated = rotate_element(kernel, current_angle)
		open =  cv2.morphologyEx(blurred, cv2.MORPH_BLACKHAT, kernel_rotated)
		mask, possible_matches = get_mask_for_DM(open, kernel_rotated)
		current_angle = current_angle + angle_step

	""" Apply to original image """
	return mask * img

def get_mask_for_DM(img, kernel):
	"""
	Try to detect possible ROI for datamatrix
	Using algorithm from :
	"Efficient 1D and 2D barcode detection using
	mathematical morphology"
	Melinda Katona and L´aszl´o G. Ny´ul

	@param img = Blurred / Opened image
	@return Binary mask for datamatrix DM ROI
	"""

	possible_matches = 0
	h, w = img.shape[:2]

	""" Binarize """
	_, bw = cv2.threshold(img, 0, 1, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

	""" DIST_L2 = Eculidean distance, 3 = approx """
	distance = cv2.distanceTransform(bw, cv2.DIST_L2, 3)

	""" Compute average for every row """
	row_average = np.average(distance, axis=1)

	""" Production parameters regarding average and minimum area """
	tolerance = 1
	""" 1% of total area """
	min_area = 0.01 * h * w
	""" 25% of total area """
	max_area = 0.25 * h * w
	convex_defect_limit = 10

	index = 0
	for row in distance:
		average = row_average[index]
		""" Keep pixel with distance < average """
		row = (row < average * tolerance).astype(int)
		distance[index] = row
		index = index + 1

	""" Erode and dilate """
	distance = cv2.GaussianBlur(distance,(5,5),0.3)

	""" Convert to uint 8 matrix """
	distance = distance.astype(np.uint8)
	""" Initialize mask """
	mask = distance * 0
	_, cnts, _ = cv2.findContours(distance, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	""" Get contours surfaces """
	contours = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]
	for c in cnts:
		""" Get approx polynome """
		approx = cv2.approxPolyDP(c,0.01*cv2.arcLength(c,True),True)
		""" Calculate convex hull """
		hull = cv2.convexHull(approx,returnPoints = False)
		""" Get defects """
		defects = cv2.convexityDefects(approx,hull)
		sum_of_defects = 0

		if defects is not None:
			for i in range(defects.shape[0]):
				s,e,f,d = defects[i,0]
				sum_of_defects=sum_of_defects+d

		""" Try to filter out "not square-like" contours based on hull defects """
		if sum_of_defects <= convex_defect_limit:
			""" Filter out area """
			area = cv2.contourArea(c)
			if area > min_area and area < max_area:
				cv2.drawContours(mask, [c], -1, (1, 1, 1), -1)
				possible_matches = possible_matches + 1

	return mask, possible_matches

def rotate_element(element, angle):
	(h,w) = element.shape[:2]
	(cx,cy) = (w/2,h/2)

	M = cv2.getRotationMatrix2D((cx,cy),-angle,1.0)

	cos = np.abs(M[0,0])
	sin = np.abs(M[0,1])

	nW = int((h*sin)+(w*cos))
	nH = int((h*cos)+(w*sin))

	M[0,2] += (nW/2) - cx
	M[1,2] += (nH/2) - cy

	return cv2.warpAffine(element,M,(nW,nH))

def rotate_image(image, angle):
	image_center = tuple(np.array(image.shape[1::-1]) / 2)
	rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
	result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_AREA)
	return result

def is_point_in_rectangle(point, rectangle):
	"""
	Return True if point is contained in rectangle
	@param point = (x,y)
	@param rectangle = aray[4][2]
	@return boolean
	"""
	x = point[0]
	y = point[1]

	rect_1 = rectangle[1]
	rect_2 = rectangle[3]

	return ((x > rect_1[0] and x < rect_2[0] and y > rect_1[1] and y < rect_2[1]) or (rect_1[0] == 0 or rect_2[0] == 300 or rect_1[1] == 0 or rect_2[1] == 300))

def execute_from_file(image_path):
	"""
	Load label picture file and process it according to jobs
	@param image_path = Path to label image file
	"""
	img = cv2.imread(image_path, 0)
	""" Standard sizing """
	img = cv2.resize(img, (2000, 1000))
	shrinked = cv2.resize(img, (600, 300))

	""" Binarize """
	pre_for_DM, dm_possible = pre_process_for_DM_heavy(img)

	""" Filter small element (avoid noise), DM should be big enough to be detected """
	anchor, orientation = get_anchor_properties_from_DM(pre_for_DM, dm_possible)

	detection = np.copy(img)

	""" Iterate over jobs and process them """
	for filename in glob.glob('./jobs/*.json'):
		with open(filename) as job_file:
			job = json.load(job_file)
			detection = process_job(anchor, orientation, detection, job, filename)

	cv2.imwrite('./output/detection.png', detection)

def get_tesseract_result(job_roi, img, pattern, deviation, label='', expected_results=None):
	"""
	Call tesseract to get string results
	@param img = image
	@return filtered image, and flag indicating if DM presence is possible
	"""
	""" Call Tessaract to get strings """
	a = datetime.datetime.now()

	""" Max iterrations """
	MAX_ITERATIONS = 2

	""" Shifting step """
	MOVE_STEP = 20
	INITIAL_SHIFT = 40

	""" Prepare image for OCR """
	OCR_fitted = np.copy(img)
	OCR_fitted = pre_process_for_OCR(OCR_fitted)

	""" OEM: 3 - Default /  PSM: 7- Single line  """
	custom_config = r'--oem 3 --psm 7'

	""" Initialize iteration """
	iteration = 0

	new_roi = job_roi
	roi_img = None
	string = ""
	upper_bound_reached = False
	result = None
	axis = 'x'
	revert = False
	cycles = 0
	""" Result not matching regex """
	""" Result not in expected results """
	""" MAX_ITERATIONS reached """
	while result is None \
	or (len(expected_results) > 1 and string not in expected_results):
		""" Not found or multiple found """
		""" Let's shift ROI """
		""" Extract ROI  """
		""" Horizontal shift """
		if cycles >= 2:
			break

		if iteration * MOVE_STEP == INITIAL_SHIFT:
			revert = axis == 'y'
			if revert:
				cycles = cycles + 1
				""" Revert """
				axis = 'x'
				iteration = 1
				new_roi[:,1] = new_roi[:,1] - INITIAL_SHIFT

				INITIAL_SHIFT = -INITIAL_SHIFT
				MOVE_STEP = -MOVE_STEP
			else:
				""" Change axis """
				axis = 'y'
				iteration = 1
				new_roi[:,0] = new_roi[:,0] - INITIAL_SHIFT

		if axis == 'x':
			if iteration == 1:
				new_roi[:,0] = new_roi[:,0] + INITIAL_SHIFT
			if iteration > 0:
				new_roi[:,0] = new_roi[:,0] - iteration * MOVE_STEP

		if axis == 'y':
			if iteration == 1:
				new_roi[:,1] = new_roi[:,1] + INITIAL_SHIFT
			if iteration > 0:
				new_roi[:,1] = new_roi[:,1] - iteration * MOVE_STEP

		print("Shifting " + axis + ": " + str(iteration * MOVE_STEP))

		""" Bound ROI and extract points """
		new_roi = bound_np_array(new_roi, 2000, 1000)

		roi_img = crop_roi(new_roi, OCR_fitted)

		iteration = iteration + 1

		string = pytesseract.image_to_string(roi_img, config=custom_config)
		result = re.match(pattern, string)

	""" Found <> MAX ITERATION """
	found = iteration < MAX_ITERATIONS

	if found:
		print("ROI found in " + str(iteration - deviation) + " iterations.")
		""" Save deviation """
		deviation = iteration
	else:
		print("ROI not found")

	if roi_img is not None:
		cv2.imwrite("./output/pre_OCR_ " + label + ".png", roi_img)

	b = datetime.datetime.now()
	print("OCR done in: " + str(b - a))

	return string, new_roi, found, deviation

def crop_roi(roi, img):
	rect = cv2.minAreaRect(roi)

	box = cv2.boxPoints(rect)
	box = np.int0(box)

	width = int(rect[1][0])
	height = int(rect[1][1])

	src_pts = box.astype("float32")

	dst_pts = np.array([
						[height-1, width-1],
						[0, width-1],
	                    [0, 0],
	                    [height-1, 0]
	                    ], dtype="float32")

	# the perspective transformation matrix
	M = cv2.getPerspectiveTransform(src_pts, dst_pts)

	# directly warp the rotated rectangle to get the straightened rectangle
	warped = cv2.warpPerspective(img, M, (height, width))

	return warped

def get_anchor_properties_from_DM(img, dm_possible):
	if dm_possible:
		a = datetime.datetime.now()
		""" Get datamatrix """
		decoded = decode(img)
		assert decoded
		b = datetime.datetime.now()
		print("DM detection done in: " + str(b - a))
		height, width = img.shape[:2]
		dm_rect = decoded[0].rect

		value = decoded[0].data.decode('ASCII')

		""" Build  DM contour """
		""" DM will be our anchor """
		""" Here we try to find it's size and if it's tilted in some way """
		""" We assume that DM is always square and paralele to label lower bounds """
		_, cnts, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		hull = None
		""" Get contours surfaces """
		for c in cnts:
			""" Get approx polynome """
			approx = cv2.approxPolyDP(c,0.01*cv2.arcLength(c,True),True)
			""" Calculate convex hull """
			hull = cv2.convexHull(approx,returnPoints = True)

		""" Get angle from contour and minAreaRect, should returns angle from origin, store box points """
		DM_area = cv2.minAreaRect(hull)

		""" Get box points """
		box = cv2.boxPoints(DM_area)
		box = np.int0(box)

		orientation = abs(DM_area[2])
	else:
		""" No DM detected, return origin as anchor """
		x1, y1, orientation, size, box = (0,0,60,50, None)

	return box, orientation

def process_job(detected_anchor, orientation, img, job, file_path):
	"""
	Process job file on current picture
	@param anchor = Current scene anchor
	@param img = Current picture
	@param job = Job object
	@param file_path = Job file path
	@return Image with detection
	"""
	print("Processing job " + file_path)
	detection = np.copy(img)
	deviation = 0
	h, w = img.shape[:2]

	for item_i in job:
		item = job[item_i]
		if 'type' in item:
			if item['type'] == 'OCR':
				job_roi = extract_roi_from_job(item, detected_anchor, orientation)
				label = item['label']
				pattern_for_re = item['value'].replace('\\\\', '\\')
				expected_results = item['expected_results'].split(',')

				""" Bounding / Casting to int """
				job_roi = bound_np_array(job_roi, h, w)

				result, job_roi, found, deviation = get_tesseract_result(job_roi, img, pattern_for_re, deviation, label=label, expected_results=expected_results)
				rect = get_rect_from_np_coords(job_roi)

				cv2.rectangle(detection, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (128,128,128), 3)

				print( label + " " + result)
	return detection

def bound_np_array(np_array, h, w):
	np_array[np_array < 0] = 0
	bound_h = np_array[:,0]
	bound_w = np_array[:,1]

	bound_h[bound_h > h] = h
	bound_w[bound_w > w] = w

	np_array[:,0] = bound_h
	np_array[:,1] = bound_w
	np_array = np.round(np_array)
	np_array = np_array.astype(np.uint0)

	return np_array

def get_rect_from_np_coords(coords):
	""" Get top left corner """
	x, y, h, w = np.amin(coords[:,0]), \
	 			np.amin(coords[:,1]), \
				np.amax(coords[:,0]) - np.amin(coords[:,0]), \
				np.amax(coords[:,1]) - np.amin(coords[:,1])

	return (x, y, h, w)

def extract_roi_from_job(item, detected_anchor, orientation):
	if 'roi_box' in item and 'anchor' in item:
		""" Get item context """
		item_box = json.loads(item['anchor']['box'])
		np_item_box = np.asarray(item_box)

		current_anchor_box = detected_anchor

		h, status = cv2.findHomography(np_item_box, current_anchor_box)

		""" Build counter rotation matrix (we already rotated image) """
		cancel_rotation_angle = 90 - orientation
		theta = np.radians(cancel_rotation_angle)
		c, s = np.cos(theta), np.sin(theta)

		R = np.array((
				(c, s, 0),
				(-s, c, 0),
				(0, 0, 1)
		))

		"""
		h = np.array([
							[1, 0, 0],
							[0, 1, 0],
							[0, 0, 1]
							])
		"""
		""" Get item ROI """
		item_roi_box = json.loads(item['roi_box'])
		np_item_roi_box = np.asarray(item_roi_box)

		""" Create destination matrix """
		homo_coords = np.zeros((4,2))
		i = 0

		for x,y in np_item_roi_box:
			""" Compute transformation """
			res = R @ h @ (x,y,0)
			homo_coords[i] = res[0:2]
			i = i + 1

	return abs(homo_coords)

def filter_out_negatives_and_out_of_bound(array, bounds):
	""" Replace negatives with 0 """
	index = 0
	array = list(array)
	for coord in array:
		if coord < 0:
			array[index] = 0
		index = index + 1

	if array[0] > bounds[0]:
		array[0] = bounds[0]
	if array[1] > bounds[1]:
		array[1] = bounds[1]
	if len(array) == 3:
		if array[2] > bounds[0]:
			array[2] = bounds[0]

	if len(array) == 4:
		if array[3] > bounds[1]:
			array[3] = bounds[1]
	array = tuple(array)

	return array

def load_template_from_file(file_path):
	with open(file_path) as f:
		template = json.load(f)

	return template

if __name__ == "__main__":
	""" Load from file """
	a = datetime.datetime.now()

	execute_from_file('./samples/100x50prod_crisp_rotated.png')

	b = datetime.datetime.now()
	print("Full process done in: " + str(b - a))

	sys.exit(0)
