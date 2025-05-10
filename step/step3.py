"""
Bước 3: Huấn luyện mô hình Naive Bayes.
"""

import os
import pickle
import sys
import time

# Import từ project
from src.models.naive_bayes import NaiveBayesModel


def train_model():
    """Huấn luyện mô hình Naive Bayes."""
    print("\n" + "=" * 50)
    print("BƯỚC 3: HUẤN LUYỆN MÔ HÌNH")
    print("=" * 50 + "\n")

    # Thiết lập thông số
    model_type = 'multinomial'  # 'multinomial', 'bernoulli', 'complement', 'custom'

    # Tải dữ liệu đã tiền xử lý từ bước 2
    input_file = 'preprocessed_data.pkl'
    if not os.path.exists(input_file):
        print(f"Không tìm thấy file {input_file}. Vui lòng chạy step2_preprocess.py trước.")
        sys.exit(1)

    with open(input_file, 'rb') as f:
        data = pickle.load(f)

    print(f"Đã tải dữ liệu tiền xử lý ({data['X_train'].shape[0]} mẫu huấn luyện).")

    # Khởi tạo mô hình
    print(f"Khởi tạo mô hình Naive Bayes loại: {model_type}")
    model = NaiveBayesModel(model_type=model_type)

    # Huấn luyện mô hình
    print("Đang huấn luyện mô hình...")
    start_time = time.time()
    model.fit(data['X_train'], data['y_train'])
    train_time = time.time() - start_time

    print(f"Đã huấn luyện mô hình trong {train_time:.2f} giây")

    # Tiến hành dự đoán trên một số mẫu để kiểm tra tốc độ
    print("\nKiểm tra tốc độ dự đoán...")
    sample_size = min(100, data['X_test'].shape[0])
    start_time = time.time()
    predictions = model.predict(data['X_test'][:sample_size])
    predict_time = time.time() - start_time

    print(f"Thời gian dự đoán trung bình: {(predict_time / sample_size) * 1000:.2f} ms/email")

    # Lưu mô hình
    model_file = f"{model_type}_model.joblib"
    model_path = model.save_model(model_file)
    print(f"Đã lưu mô hình vào: {model_path}")

    # Lưu mô hình và dữ liệu kiểm tra cho bước tiếp theo
    output_file = 'trained_model_data.pkl'
    with open(output_file, 'wb') as f:
        pickle.dump({
            'model_path': model_path,
            'model_type': model_type,
            'data': data  # Bao gồm dữ liệu kiểm tra
        }, f)

    print(f"Đã lưu thông tin mô hình vào file {output_file} để các bước tiếp theo sử dụng")

    return model, data


if __name__ == "__main__":
    train_model()