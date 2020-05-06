import sys
import cv2
import re
import numpy as np
import math
import json
import datetime
import glob
import pytesseract


""" Requieres : sudo apt-get install libdmtx0a"""
from pylibdmtx.pylibdmtx import decode

def crop_image_square(img, crop_size=3):
	"""
	Crop image into smaller windows
	@param crop_size = window size
	@return an array of cropped windows
	"""
	crop_collection = []

	for r in range(0, img.shape[0]):
		for c in range(0, img.shape[1]):
			window = img[r:r+crop_size, c:c+crop_size]
			crop_collection.append(window)

	return crop_collection

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
	@return : 600x300 binarized image
	"""

	## Threshold
	_, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
	kernel = np.ones((1,1),np.uint8)
	erode = cv2.erode(thresh, kernel, iterations = 1)
	return erode

def filter_out_big_elements(img, count_biggest = 10):
	"""
	Filter out small elements
	@param img = Thinned image
	@param size = Element size
	@return filtered image, and flag indicating if DM presence is possible
	"""
	dm_possible = False
	_, contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	""" Get contours surfaces """
	areas = [cv2.contourArea(c) for c in contours]
	""" If there is only one countour, there is no DM detected """
	areas.sort(reverse=True)
	for cnt in range(len(areas)):
		if(cnt < count_biggest):
			cv2.drawContours(img, [contours[cnt]], 0, (0,0,0), -1)

	return img

def filter_out_small_elements(img, size=200):
	"""
	Filter out small elements
	@param img = Thinned image
	@param size = Element size
	@return filtered image, and flag indicating if DM presence is possible
	"""
	dm_possible = False
	_, contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	""" Get contours surfaces """
	areas = [cv2.contourArea(c) for c in contours]
	""" If there is only one countour, there is no DM detected """

	if len(areas) > 1:
		for cnt in range(len(areas)):
			if(areas[cnt] < size):
				cv2.drawContours(img, [contours[cnt]], 0, (0,255,0), -1)

		dm_possible = True

	return img, dm_possible

def get_external_contour(img, margin=1):
	"""
	Get external object borders
	@param img = Thinned image
	@param margin = Applied margin
	@return bounding box
	"""
	""" Finding bounding box """
	""" Blur to smooth contours """
	blurred = cv2.blur(img,(20,20))
	_, contours, hierarchy = cv2.findContours(blurred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	""" Get biggest contours """
	areas = [cv2.contourArea(c) for c in contours]
	max_index = np.argmax(areas)
	cnt=contours[max_index]

	""" x, y, w, h of bounding rectangle """
	bounding_box = cv2.minAreaRect(cnt)
	bounding_box = cv2.boxPoints(bounding_box)
	bounding_box = np.int0(bounding_box)

	""" Apply margin """
	bounding_box = bounding_box * margin
	bounding_box = np.int0(bounding_box)

	return bounding_box

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
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(25,25))
	open =  cv2.morphologyEx(blurred, cv2.MORPH_BLACKHAT, kernel)


	""" Try again with SE rotated """
	if possible_matches == 0:
		print("DM Mask not found, rotating SE")
		kernel_rotated = rotate_element(kernel, 90)
		open =  cv2.morphologyEx(blurred, cv2.MORPH_BLACKHAT, kernel_rotated)
		mask, possible_matches = get_mask_for_DM(open)

	cv2.imwrite('./output/pre_DM.png', mask * img)

	""" Apply to original image """
	return mask * img

def get_mask_for_DM(img):
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
	min_area =0.01* h * w
	""" 25% of total area """
	max_area =0.25* h * w
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
	cv2.imwrite('./output/pre_DM.png', distance * 255)
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
	print(M.shape)
	print(M)

	cos = np.abs(M[0,0])
	sin = np.abs(M[0,1])

	nW = int((h*sin)+(w*cos))
	nH = int((h*cos)+(w*sin))

	M[0,2] += (nW/2) - cx
	M[1,2] += (nH/2) - cy

	return cv2.warpAffine(element,M,(nW,nH))

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
	""" Prepare for OCR and DM detection """
	#prepared = prepare_image(img)

	shrinked = cv2.resize(img, (600, 300))

	""" Binarize """
	pre_for_DM, dm_possible = pre_process_for_DM_heavy(shrinked)

	""" Get scaling rate """
	height, width = pre_for_DM.shape[:2]
	original_h, original_w = img.shape[:2]

	y_scaling = original_h / height
	x_scaling = original_w / width

	""" Filter small element (avoid noise), DM should be big enough to be detected """
	anchor = get_anchor_properties_from_DM(pre_for_DM, dm_possible)

	""" Scale according to original image """
	anchor = (int(anchor[0] * x_scaling), int(anchor[1] * y_scaling), anchor[2], anchor[3] * x_scaling)

	""" Show anchor """
	cv2.circle(img, (anchor[0], anchor[1]), 5, (128,128,128), -1)

	envelop = get_dm_envelop(anchor, img)
	anchor = (anchor[0], anchor[1], anchor[2], anchor[3], envelop)

	print("Current anchor: " + str(anchor))
	detection = np.copy(img)
	""" Iterate over jobs and process them """
	for filename in glob.glob('./jobs/*.json'):
		with open(filename) as job_file:
			job = json.load(job_file)
			detection = process_job(anchor, detection, job, filename)

	cv2.imwrite('./output/detection.png', detection)

def get_dm_envelop(anchor, img):
	""" Anchor is bottom left point """
	""" (x1, y1, orientation, size) """
	blurred = cv2.blur(img,(5,5))

	box = None

	margin_x, margin_y = 7,7
	width, height = 1, 1
	x_a, y_a, _, _ = anchor
	env = None

	while blurred[y_a - margin_y - height][x_a + margin_x + width] <= 100:
		width = width + 1

	env = (width + margin_x, 0)

	width, height = 1, 1
	while blurred[y_a - margin_y - height][x_a + margin_x + width] <= 100:
		height = height + 1

	env = (env[0], height - margin_y)

	return env

def get_tesseract_result(job_roi, img, pattern, deviation):
	"""
	Call tesseract to get string results
	@param img = image
	@return filtered image, and flag indicating if DM presence is possible
	"""
	""" Call Tessaract to get strings """
	a = datetime.datetime.now()
	""" Max 10 iterrations """
	MAX_ITERATIONS = 5
	""" Shifting step """
	MOVE_STEP = 7
	OCR_fitted = np.copy(img)
	OCR_fitted = pre_process_for_OCR(OCR_fitted)
	cv2.imwrite("./output/pre_OCR.png", OCR_fitted)

	new_roi = job_roi

	""" Extract ROI, revert if needed  """
	if job_roi[0] > job_roi[2]:
		new_roi = (job_roi[2], new_roi[1], job_roi[0], new_roi[3])

	if job_roi[1] > job_roi[3]:
		new_roi = (new_roi[0], job_roi[3], new_roi[2], job_roi[1])

	roi_img = OCR_fitted[new_roi[0]:new_roi[2]+1, new_roi[1]:new_roi[3]+1]

	""" OEM: 3 - Default /  PSM: 7- Single line  """
	custom_config = r'--oem 3 --psm 7'

	string = pytesseract.image_to_string(roi_img, config=custom_config)
	""" Initialize iteration """
	if deviation > 0:
		iteration = deviation
	else:
		iteration = 1

	new_roi = job_roi
	upper_bound_reached = False
	while re.match(pattern, string) is None and iteration < MAX_ITERATIONS:
		""" Not found or multiple found """
		""" Let's shift ROI """
		""" Extract ROI  """
		if iteration > MAX_ITERATIONS / 2 and not upper_bound_reached:
			new_roi = (
						job_roi[0] - iteration * MOVE_STEP,
						job_roi[1] - iteration * MOVE_STEP,
						job_roi[2] - iteration * MOVE_STEP,
						job_roi[3] - iteration * MOVE_STEP
						)
			""" Replace negatives with 0 and out of bound """
			new_roi = filter_out_negatives_and_out_of_bound(new_roi, (1000, 2000))
		else:
			new_roi = (
						job_roi[0] + iteration * MOVE_STEP,
						job_roi[1] + iteration * MOVE_STEP,
						job_roi[2] + iteration * MOVE_STEP,
						job_roi[3] + iteration * MOVE_STEP
						)
			new_roi = filter_out_negatives_and_out_of_bound(new_roi, (1000, 2000))

		roi_img = img[new_roi[1]:new_roi[3], new_roi[0]:new_roi[2]]
		iteration = iteration + 1
		string = pytesseract.image_to_string(roi_img, config=custom_config)

	""" Found <> MAX ITERATION """
	found = iteration == MAX_ITERATIONS

	if found:
		print("ROI not found.")
	else:
		print("ROI found in " + str(iteration) + " iterations.")
		""" Save deviation """
		deviation = iteration
	b = datetime.datetime.now()
	print("OCR done in: " + str(b - a))

	return string, new_roi, found, deviation

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
		x1, y1, x2, y2, x3, y3, x4, y4 = (dm_rect.left,
							height - dm_rect.top,

							dm_rect.left,
							height - dm_rect.top - dm_rect.height,

							dm_rect.left + dm_rect.width,
							height - dm_rect.top,

							dm_rect.left + dm_rect.width,
							height - dm_rect.top - dm_rect.height,
							)
		cont = np.array([
						[x1,  y1],
						[x2,  y2],
						[x3,  y3],
						[x4,  y4]
						])
		""" Squared DM """
		size = x2 - x1
		""" Get angle from contour and minAreaRect, should returns angle from origin """
		DM_area = cv2.minAreaRect(cont)

		orientation = abs(DM_area[2]) % 45
	else:
		""" No DM detected, return origin as anchor """
		x1, y1, orientation, size, box = (0,0,60,50, None)

	return (x1, y1, orientation, size)

def process_job(detected_anchor, img, job, file_path):
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

	for item_i in job:
		item = job[item_i]
		if 'type' in item:
			if item['type'] == 'OCR':
				job_roi = extract_roi_from_job(item, detected_anchor)

				pattern_for_re = item['value'].replace('\\\\', '\\')
				result, job_roi, found, deviation = get_tesseract_result(job_roi, img, pattern_for_re, deviation)

				cv2.rectangle(detection, (job_roi[0],job_roi[1]), (job_roi[2],job_roi[3]), (0,255,0), 3)
				print(item['label'] + " " + result)

	return detection

def extract_roi_from_job(item, detected_anchor):
	x1, y1, x2, y2 = (0, 0, 0, 0)

	""" Unpack detected anchor """
	a_x, a_y, a_width, a_height = (detected_anchor[0], detected_anchor[1], detected_anchor[4][0], detected_anchor[4][1])
	rotation = detected_anchor[2]
	anchor_size = int(detected_anchor[3])

	if 'anchor' in item:
		""" Apply size rate """
		size_rate = anchor_size / int(item['anchor']['size'])
		"""" Item anchor pos """
		i_a_x, i_a_y = int(item['anchor']['x']), int(item['anchor']['y'])

	if 'rect' in item:
		""" Unpack rectangle """
		""" (r_x1, r_y1) top-left corner """
		""" (r_x2, r_y2) bottom right corner """
		r_x1, r_y1, r_x2, r_y2 = (int(item['rect']['x1']), int(item['rect']['y1']), int(item['rect']['x2']), int(item['rect']['y2']))

		""" As np arrays """
		rect_1 = np.array([r_x1, r_y1, 1])
		rect_2 = np.array([r_x2, r_y2, 1])

		pts_src = np.array([
							[i_a_x, i_a_y],
							[i_a_x + int(item['anchor']['size']), i_a_y],
							[i_a_x, i_a_y - int(item['anchor']['size'])],
							[i_a_x + int(item['anchor']['size']), i_a_y - int(item['anchor']['size'])]
							])

		pts_dst = np.array([
							[a_x, a_y],
							[a_x + a_width, a_y],
							[a_x, a_y - a_height],
							[a_x + a_width, a_y - a_height]
							])

		h, status = cv2.findHomography(pts_src, pts_dst)


		"""
		h = np.array([
							[1, 0, 0],
							[0, 1, 0],
							[0, 0, 1]
							])
		print(h)
		"""
		final_1 = h @ rect_1
		final_2 = h @ rect_2

		final_1 = filter_out_negatives_and_out_of_bound(final_1, (1000, 2000))
		final_2 = filter_out_negatives_and_out_of_bound(final_2, (1000, 2000))

		x1, y1, x2, y2 = final_1[0], final_1[1], final_2[0], final_2[1]

		#print("From " + str((r_x1, r_y1, r_x2, r_y2)))
		#print("To " + str((int(x1), int(y1), int(x2), int(y2))))

	return (int(x1), int(y1), int(x2), int(y2))

def rotate_point(point, center, angle):
	x, y = point
	cx, cy = center
	theta = np.radians(angle)

	""" Translate to origen """
	trans_X = x - cx;
	trans_Y = y - cy;

	""" Rotate """
	rot_X = trans_X*np.cos(theta) - trans_Y*np.sin(theta);
	rot_Y = trans_X*np.sin(theta) + trans_Y*np.cos(theta);

	""" Back to position """
	x = rot_X + cx;
	y = rot_Y + cy;

	return (x , y)

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
