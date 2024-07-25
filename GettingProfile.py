import os

profile = os.environ.get("AWS_PROFILE", "default")
print(profile)
