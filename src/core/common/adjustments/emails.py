import re

email_re = re.compile(r"([^@]+)@([^@]+)\.([^@]+)")

at_re = re.compile(r'\s*@\s*')


def replace_mail_addresses(text):
  text = re.sub(email_re, r"\1 at \2 dot \3", text)
  return text


def replace_at_symbols(text):
  text = re.sub(at_re, ' at ', text)
  return text
