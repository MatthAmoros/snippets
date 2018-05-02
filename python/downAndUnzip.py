import subprocess
import zipfile
import tempfile
from smb.SMBConnection import *
import configparser

""" Flag """
fileDownloaded = False

""" Configuration file """
""" Reading configuration """
appConfig = configparser.ConfigParser()
appConfig.read("./cfg/config.ini")

print("Sections found : " + str(appConfig.sections()))

if len(appConfig.sections()) == 0:
	raise RuntimeError("Could not open configuration file")
		
SMB_USER = appConfig.get("SMB", "User")
SMB_PASSWORD = appConfig.get("SMB", "Password")
SMB_DOMAIN  = appConfig.get("SMB", "Domain")
SMB_PATH = appConfig.get("SMB", "Path")
SMB_HOST = appConfig.get("SMB", "Host")
SMB_SHARE = appConfig.get("SMB", "SharedRessourceName")

""" SMB """
try:
	print("Connecting to shared directory...")

	conn = SMBConnection(SMB_USER, SMB_PASSWORD, 'python-zeep', SMB_HOST, SMB_DOMAIN, use_ntlm_v2=True,
						sign_options=SMBConnection.SIGN_WHEN_SUPPORTED,
						is_direct_tcp=True) 
						
	connected = conn.connect(SMB_HOST, 445)    

	print("Getting " + str(SMB_PATH) + " ...")
	""" Saving to temporary file """

	file_obj = tempfile.NamedTemporaryFile()
	file_attributes, filesize = conn.retrieveFile(SMB_SHARE, SMB_PATH, file_obj)

	file_obj.seek(0)
	fileDownloaded = True
except ConnectionResetError:
	print("Connection closed")
except OperationFailure:
	print("File : " + str(SMB_PATH) + " not found")

if fileDownloaded == True:
	try:
		""" Unzipping file """
		print("Unzipping ... ")
		if file_obj is not None:
			zip_ref = zipfile.ZipFile(file_obj.name, 'r')
			zip_ref.extractall('./unzipped/')
			zip_ref.close()
			file_obj.close()

	except zipfile.BadZipFile:
		print("Loaded file is not a zipped filed")			

try:
	""" Loading file """
	print("Loading ...")
	RUN_EXE_NAME = appConfig.get("RUN", "ExecutableName")
	subprocess.run("./unzipped/" + str(RUN_EXE_NAME), shell=True, check=True)
except:
	pass

