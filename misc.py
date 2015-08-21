def pwd_gen(pwdlength):
	import string
	import random
	chars=string.ascii_uppercase+string.ascii_lowercase+string.digits
	pwd=""
	for p in range(pwdlength):
		pwd+=chars[random.randint(0,62)]
	
	return pwd