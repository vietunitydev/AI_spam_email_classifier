"""
Bước 2: Tiền xử lý dữ liệu email.
"""

import os
import pickle
import sys

# Import từ project
from src.data.preprocessor import EmailPreprocessor


def preprocess_data():
    """Tiền xử lý dữ liệu email để chuẩn bị cho huấn luyện."""
    print("\n" + "=" * 50)
    print("BƯỚC 2: TIỀN XỬ LÝ DỮ LIỆU")
    print("=" * 50 + "\n")

    # Thiết lập thông số
    vectorizer_type = 'tfidf'  # 'tfidf', 'count', hoặc 'binary'

    # Tải dữ liệu từ bước 1
    input_file = 'email_data.pkl'
    if not os.path.exists(input_file):
        print(f"Không tìm thấy file {input_file}. Vui lòng chạy step1_download_data.py trước.")
        sys.exit(1)

    with open(input_file, 'rb') as f:
        email_df = pickle.load(f)

    print(f"Đã tải {len(email_df)} email.")

    # Khởi tạo preprocessor
    print(f"Tiền xử lý dữ liệu sử dụng phương pháp véc-tơ hóa: {vectorizer_type}")
    preprocessor = EmailPreprocessor()

    # Tiền xử lý dữ liệu
    data = preprocessor.prepare_data(email_df, vectorizer_type=vectorizer_type)

    print(f"Số mẫu huấn luyện: {data['X_train'].shape[0]}")
    print(f"Số mẫu kiểm tra: {data['X_test'].shape[0]}")
    print(f"Số đặc trưng: {data['X_train'].shape[1]}")

    # Hiển thị 10 đặc trưng đầu tiên
    print("\nVí dụ về các đặc trưng:")
    for i, feature in enumerate(data['feature_names'][:10]):
        print(f"  {i + 1}. {feature}")

    # Lưu kết quả tiền xử lý
    output_file = 'preprocessed_data.pkl'
    with open(output_file, 'wb') as f:
        pickle.dump(data, f)

    print(f"\nĐã lưu dữ liệu đã tiền xử lý vào file {output_file} để các bước tiếp theo sử dụng")

    return data


if __name__ == "__main__":
    preprocess_data()