import sys
import os
import boto3

POOL_REGION = os.getenv('POOL_REGION')
ACCESS_KEY = os.getenv('ACCESS_KEY')
SECRET_ACCESS_KEY = os.getenv('SECRET_ACCESS_KEY')
AWS_CLIENT_ID = os.getenv('AWS_CLIENT_ID')
USER_POOL_ID = os.getenv('USER_POOL_ID')

""" Cliente IDP """
idp_client = boto3.client(
	'cognito-idp', 
	aws_access_key_id=ACCESS_KEY,
	aws_secret_access_key=SECRET_ACCESS_KEY, 
	region_name=POOL_REGION
)

def login_aws(username, password):
	""" Login mediante Cognito AWS """
	try:
		result = idp_client.initiate_auth(
				AuthFlow='USER_PASSWORD_AUTH',
				AuthParameters={
					'USERNAME': username,
					'PASSWORD': password},
				ClientId=AWS_CLIENT_ID)
		""" Verificamos si hay que modificar la contraseña """
		if 'ChallengeName' in result:
			pending_challenge = result['ChallengeName']
			print(result)
			""" NEW_PASSWORD_REQUIRED: Debe modificar la contraseña """
		if 'AccessToken' in result['AuthenticationResult']:
			""" Recuperamos un token, estamos OK """
			return "OK", 200
	except idp_client.exceptions.NotAuthorizedException:
		""" Error usuario/password """
		pass
	return "NOK", 401

def signup_aws(username, password):
	""" Creación de cuenta AWS mediante Congito IDP """
	""" Se requiere validación del correo/contraseña para habilitar la contraseñq """

	try:
		response = idp_client.sign_up(
			ClientId=AWS_CLIENT_ID,
			Username=username,
			Password=password,
			UserAttributes=[
				{
					'Name': 'email',
					'Value': username
				},
			],
			UserContextData={
				'EncodedData': 'devicefingerprint'
			}
		)
		""" ClientMetadata es una asociación libre que se puede pasar a un trigger lambda """
	except idp_client.exceptions.UsernameExistsException:
		return "Ya existe una cuenta con este correo.", 401
		pass
	return "OK", 200

def validate_signup(username, temp_pass):
	""" Validación de cuenta """

	""" Hay que obtener el ip de la solicitud y el fingerprint del equipo
	Eso perimite a AWS saber si es un equipo nuevo que solicita la confirmación """
	device_fingerprint = ''

	try:
		response = idp_client.confirm_sign_up(
			ClientId=AWS_CLIENT_ID,
			Username=username,
			ConfirmationCode=temp_pass,
			ForceAliasCreation=True,
			UserContextData={
				'EncodedData': device_fingerprint
			}
		)
	except idp_client.exceptions.ExpiredCodeException:
		""" Código de validación vencido """
		return "Solicitar nuevo código de validación", 401
	return "OK", 200

def resend_confirmation(username):
	""" Solicitar el reenvío de un código de confirmación """
	device_fingerprint = ''

	response = idp_client.resend_confirmation_code(
		ClientId=AWS_CLIENT_ID,
		UserContextData={
			'EncodedData': device_fingerprint
		},
		Username=username,
	)

	return "OK", 200

def forget_pass(username):
	""" Contraseña olvidada """
	response = idp_client.forgot_password(
		ClientId=AWS_CLIENT_ID,
		Username=username
	)

def confirm_forgot_password(username, confirm_code, new_password):
	try:
		response = idp_client.confirm_forgot_password(
			ClientId=AWS_CLIENT_ID,
			Username=username,
			ConfirmationCode=confirm_code,
			Password=new_password
		)

		return "OK", 200
	except idp_client.exceptions.InvalidPasswordException:
		""" No conforme a las reglas de contraseña (no tiene cifras etc.) """
		pass
	return "NOK", 401

if __name__ == '__main__':
	args = sys.argv[1:]

	if args[0] == 'login':
		""" Probemos la autenticación con AWS """
		username = args[1]
		password = args[2]
		_, status = login_aws(username, password)
		print(status)
	elif args[0] == 'forget_pass':
		""" Contraseña olvidada """
		username = args[1]
		forget_pass(username)
		confirm = input("Confirmación contraseña olvidada:")
		new_pass = input("Nueva contraseña:")
		confirm_forgot_password(username, confirm, new_pass)
	elif args[0] == 'full_cycle':
		""" Probemos la creación de cuenta con AWS """
		username = args[1]
		password = args[2]
		_, status = signup_aws(username, password)
		if status == 200:
			""" OK; cuenta creada hay que confirmarla """
			pass

		confirm_code = input("Código de confirmación:")
		_, status = validate_signup(username, str(confirm_code))
		if status == 401:
			""" Confirmación vencida, hay que solicitar de nuevo """
			_, status = resend_confirmation(username)
		elif status == 200:
			""" Probemos el login """
			_, status = login_aws(username, password)
			if status == 200:
				""" Todo OK pudimos conectarnos """
				print("Ciclo de auth OK!")
