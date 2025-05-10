"""
Cài đặt NLTK và khắc phục lỗi SSL.
Chạy file này trước để đảm bảo NLTK được cài đặt đúng cách.
"""

import ssl
import nltk

# Khắc phục lỗi SSL khi tải dữ liệu NLTK
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Tải các tài nguyên NLTK cần thiết
print("Đang tải dữ liệu NLTK...")
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

print("Đã cài đặt NLTK thành công!")