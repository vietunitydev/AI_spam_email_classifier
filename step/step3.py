"""
Bước 3: Huấn luyện mô hình Naive Bayes (sử dụng Custom Implementation).
"""

import os
import pickle
import sys
import time

# Import từ project
from src.models.naive_bayes import CustomNaiveBayes  # Thay đổi này


def train_model():
    """Huấn luyện mô hình Naive Bayes."""
    print("\n" + "=" * 50)
    print("BƯỚC 3: HUẤN LUYỆN MÔ HÌNH (CUSTOM NAIVE BAYES)")
    print("=" * 50 + "\n")

    # Thiết lập thông số cho Custom Naive Bayes
    alpha = 1.0  # Laplace smoothing parameter
    fit_prior = True  # Có học class prior hay không

    # Tải dữ liệu đã tiền xử lý từ bước 2
    input_file = 'preprocessed_data.pkl'
    if not os.path.exists(input_file):
        print(f"Không tìm thấy file {input_file}. Vui lòng chạy step2_preprocess.py trước.")
        sys.exit(1)

    with open(input_file, 'rb') as f:
        data = pickle.load(f)

    print(f"Đã tải dữ liệu tiền xử lý ({data['X_train'].shape[0]} mẫu huấn luyện).")
    print(f"Số đặc trưng: {data['X_train'].shape[1]}")

    # Khởi tạo mô hình Custom Naive Bayes
    print(f"Khởi tạo Custom Naive Bayes với alpha={alpha}, fit_prior={fit_prior}")
    model = CustomNaiveBayes(alpha=alpha, fit_prior=fit_prior)

    # Chuyển đổi dữ liệu sparse matrix thành dense array (nếu cần)
    X_train = data['X_train']
    X_test = data['X_test']

    # Custom Naive Bayes cần dense array
    if hasattr(X_train, 'toarray'):
        print("Chuyển đổi sparse matrix thành dense array...")
        X_train = X_train.toarray()
        X_test = X_test.toarray()

    print(f"Kích thước dữ liệu huấn luyện: {X_train.shape}")

    # Huấn luyện mô hình
    print("Đang huấn luyện mô hình Custom Naive Bayes...")
    start_time = time.time()
    model.fit(X_train, data['y_train'])
    train_time = time.time() - start_time

    print(f"Đã huấn luyện mô hình trong {train_time:.2f} giây")

    # Hiển thị thông tin mô hình sau khi huấn luyện
    print(f"Số lớp: {len(model.classes_)}")
    print(f"Các lớp: {model.classes_}")
    print(f"Class log priors: {model.class_log_prior_}")

    # Tiến hành dự đoán trên một số mẫu để kiểm tra tốc độ
    print("\nKiểm tra tốc độ dự đoán...")
    sample_size = min(100, X_test.shape[0])
    start_time = time.time()
    predictions = model.predict(X_test[:sample_size])
    predict_time = time.time() - start_time

    print(f"Thời gian dự đoán trung bình: {(predict_time / sample_size) * 1000:.2f} ms/email")

    # Kiểm tra một số dự đoán mẫu
    print(f"\nMột số dự đoán mẫu (10 mẫu đầu):")
    sample_predictions = model.predict(X_test[:10])
    sample_probabilities = model.predict_proba(X_test[:10])

    for i in range(min(10, len(sample_predictions))):
        pred_label = "SPAM" if sample_predictions[i] == 1 else "HAM"
        spam_prob = sample_probabilities[i][1] * 100
        print(f"  Mẫu {i+1}: {pred_label} (Xác suất spam: {spam_prob:.2f}%)")

    # Lưu mô hình
    model_file = "custom_naive_bayes_model.joblib"
    model_path = model.save_model(model_file)
    print(f"Đã lưu mô hình vào: {model_path}")

    # Cập nhật dữ liệu với dense arrays
    data['X_train'] = X_train
    data['X_test'] = X_test

    # Lưu mô hình và dữ liệu kiểm tra cho bước tiếp theo
    output_file = 'trained_model_data.pkl'
    with open(output_file, 'wb') as f:
        pickle.dump({
            'model_path': model_path,
            'model_type': 'custom',
            'model_params': {'alpha': alpha, 'fit_prior': fit_prior},
            'data': data  # Bao gồm dữ liệu kiểm tra
        }, f)

    print(f"Đã lưu thông tin mô hình vào file {output_file} để các bước tiếp theo sử dụng")

    return model, data


if __name__ == "__main__":
    train_model()