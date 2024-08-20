import os

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'y0u-c0u1d-pr0b@bly-gu3$$-th1$!'

config = Config()
