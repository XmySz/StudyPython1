import string
import random

password_length = 16
total = string.ascii_letters + string.digits + string.punctuation
password = random.sample(total,password_length)

for i in range(10):
    print("".join(password))
