from chat_crawler import Extract
from preprocessing import Transform
from load_database import Load


if __name__ == '__main__':
    Extract()
    Transform()
    Load()