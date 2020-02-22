import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

fromEmail = 'rpismartcamera@gmail.com'

fromEmailPassword = 'flatronl1730s'

toEmail = 'ntesla102@gmail.com'

def sendEmail():
	msgRoot = MIMEMultipart('related')
	msgRoot['Subject'] = 'Оповіщення SmartSecurityCamera'
	msgRoot['From'] = fromEmail
	msgRoot['To'] = toEmail

	msgAlternative = MIMEMultipart('alternative')
	msgRoot.attach(msgAlternative)
	msgText = MIMEText('На Вашій території знаходиться постороння людина!')
	msgAlternative.attach(msgText)

	smtp = smtplib.SMTP('smtp.gmail.com', 587)
	smtp.starttls()
	smtp.login(fromEmail, fromEmailPassword)
	smtp.sendmail(fromEmail, toEmail, msgRoot.as_string())
	smtp.quit()