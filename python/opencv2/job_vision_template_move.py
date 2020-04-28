#!/usr/bin/env python3
import sys
import cv2
import numpy as np
import math
import json
import datetime
import glob

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

def pre_process(img):
	"""
	Resize and binarize image
	@param img = Grayscale image
	@return : 600x300 binarized image
	"""

	## Threshold
	_, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

	# Remove small details with opening
	kernel = np.ones((2,2),np.uint8)
	opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)

	#Smooth DM square to be rotation proof
	dilate = cv2.blur(opening,(3,3))

	"""
	# Combine noise
	kernel = np.ones((1,1),np.uint8)
	dilate = cv2.dilate(opening,kernel,iterations=2)
	"""
	return dilate

def filter_out_small_elements(img, size=1):
	"""
	Filter out small elements
	@param img = Thinned image
	@param size = Element size
	@return filtered image
	"""
	_, contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	""" Get contours surfaces """
	areas = [cv2.contourArea(c) for c in contours]

	for cnt in range(len(areas)):
		if(areas[cnt] < size):
			cv2.drawContours(img, [contours[cnt]], 0, (0,255,0), -1)

	return img

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

def draw_defects(img, minutiae_pos):
	for i in range(len(minutiae_pos)):
		#NOTE : color filled circle of 2px at minutiae center
		cv2.circle(img, (minutiae_pos[i][0], minutiae_pos[i][1]), 2, minutiae_pos[i][2], -1)

	return img

def save_to_json_file(template_name, template):
	"""
	Save template to JSON file for later use
	"""
	with open('./templates/' + template_name + '.tmplt', 'w') as outfile:
		""" Numpy arrays need to be converted to list to be dumped to JSON """
		listify = template
		if isinstance(template, list) == False:
			listify = template.tolist()
		json.dump(listify, outfile)

def execute_from_file(image_path):
	"""
	Load label picture file and process it according to jobs
	@param image_path = Path to label image file
	"""
	img = cv2.imread(image_path, 0)
	""" To gray scale and resize """
	colorized_output = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
	colorized_output = cv2.resize(img, (600, 300))
	""" Binarize """
	binarized = pre_process(colorized_output)
	height, width = binarized.shape[:2]

	""" Filter small element (avoid noise), DM should be big enough to be detected """
	binarized = filter_out_small_elements(binarized, size=30)
	cv2.imwrite('./output/detection.png', binarized)
	anchor = get_anchor_properties_from_DM(binarized)

	print("Current anchor: " + str(anchor))
	""" Mark anchor """
	colorized_output = cv2.circle(colorized_output, (anchor[0], anchor[1]), 2, (128,128, 128), 5)
	""" Output detection """
	cv2.imwrite('./output/detection.png', binarized)

	""" Iterate over jobs and process them """
	for filename in glob.glob('./jobs/*.json'):
		try:
			with open(filename) as job_file:
				job = json.load(job_file)
				process_job(anchor, colorized_output, job, filename)
		except Exception as e:
			print(str(e))
			pass

def get_anchor_properties_from_DM(img):
	""" Get datamatrix """
	decoded = decode(img)
	assert decoded
	height, width = img.shape[:2]
	rect = decoded[0].rect
	value = decoded[0].data.decode('ASCII')

	""" Build  DM contour """
	""" DM will be our anchor """
	""" Here we try to find it's size and if it's tilted in some way """
	""" We assume that DM is always square and paralele to label lower bounds """
	x1, y1, x2, y2 = (rect.left, height - rect.top, rect.left + rect.width, height - rect.top - rect.height)
	cont = np.array([[x1,  y1], [x2,  y2]])
	""" Squared DM """
	size = x2 - x1
	""" Get angle from contour and minAreaRect, should returns angle from origin """
	DM_area = cv2.minAreaRect(cont)
	""" Crop DM """
	DM_image = img[y2-size:y1+size, x1-size:x2+size]

	orientation = DM_area[2] + 45

	return (x1, y1, orientation, size)

def extract_L_from_DM_image(img):
	edges = cv2.Canny(img,50,150,apertureSize = 3)
	cv2.imwrite('canny.png', edges)

def process_job(anchor, img, job, file_path):
	"""
	Process job file on current picture
	@param anchor = Current scene anchor
	@param img = Current picture
	@param job = Job object
	@param file_path = Job file path
	"""
	print("Processing job " + file_path)
	""" Unpack detected anchor """
	a_x, a_y = (anchor[0], anchor[1])
	rotation = anchor[2]
	anchor_size = int(anchor[3])

	for item_i in job:
		item = job[item_i]
		if 'anchor' in item:
			""" Apply size rate """
			size_rate = anchor_size / int(item['anchor']['size'])
			"""" Item anchor pos """
			i_a_x, i_a_y = int(item['anchor']['x']), int(item['anchor']['y'])
			""" Calculate transformation """
			""" Scaling """
			S = np.array([
							[size_rate, 0, 0],
							[ 0, size_rate, 0],
							[ 0, 0, 1]
						])

			""" Rotation """
			angle = rotation - int(item['anchor']['o'])
			theta = np.radians(angle)
			c, s = np.cos(theta), np.sin(theta)

			R = np.array((
						(c, -s, 0),
			 			(s, c, 0),
						(0, 0, 1)
						))

			""" Translation """
			x_scale = a_x - i_a_x
			y_scale = a_y - i_a_y

			T = np.array([
							[1, 0, x_scale],
							[0,  1, y_scale],
							[0, 0, 1]
						])

			print("Scaling: " + str(size_rate) + " Rotation:" + str(angle) + " Translation:" + str((x_scale, y_scale)))
			if 'rect' in item:
				""" Unpack rectangle """
				""" (r_x1, r_y1) top-left corner """
				""" (r_x2, r_y2) bottom right corner """
				r_x1, r_y1, r_x2, r_y2 = (int(item['rect']['x1']), int(item['rect']['y1']), int(item['rect']['x2']), int(item['rect']['y2']))
				""" Center """
				cx = (r_x1 + r_x2 ) / 2
				cy = (r_y1 + r_y2 ) / 2
				""" As np arrays """
				rect_1 = np.array([r_x1, r_y1, 1])
				rect_2 = np.array([r_x2, r_y2, 1])

				""" Translate to origen """
				T_c = np.array([
								[1, 0, -cx],
								[0,  1, -cy],
								[0, 0, 1]
							])

				""" Back to postion """
				T_r = np.array([
								[1, 0, cx],
								[0,  1, cy],
								[0, 0, 1]
							])

				""" Apply transformations """
				final_1 =  S @ T @ T_r @ R @ T_c @ rect_1
				final_2 =  S @ T @ T_r @ R @ T_c @ rect_2
				x1, y1, x2, y2 = final_1[0], final_1[1], final_2[0], final_2[1]

				print("From " + str((r_x1, r_y1, r_x2, r_y2)))
				print("To " + str((int(x1), int(y1), int(x2), int(y2))))

				cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), \
							(0,0,0), 2)

	cv2.imwrite('./output/job.png', img)

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

	return np.array([[x,y ,1],
					])

def load_template_from_file(file_path):
	with open(file_path) as f:
		template = json.load(f)

	return template

if __name__ == "__main__":
	""" Load from file """
	a = datetime.datetime.now()
	execute_from_file('./samples/100x50_DM_ray_bad_def.png')
	b = datetime.datetime.now()
	print("Done in: " + str(b - a))

	sys.exit(0)
