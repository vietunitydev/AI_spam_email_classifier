"""
Chạy toàn bộ quy trình phân loại email spam với Naive Bayes.
File này thực hiện các bước: tải dữ liệu, tiền xử lý, huấn luyện mô hình,
đánh giá và phân loại một email mẫu.
"""

import os
import sys
import ssl
import nltk
import pandas as pd
import time

# Khắc phục lỗi SSL khi tải dữ liệu NLTK
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Tải các tài nguyên NLTK cần thiết
print("Đang tải dữ liệu NLTK...")
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

# Import các module từ dự án
from src.data.data_loader import EmailDataLoader
from src.data.preprocessor import EmailPreprocessor
from src.models.naive_bayes import NaiveBayesModel
from src.evaluation.metrics import ModelEvaluator
from src.utils.logger import logger


def main():
    """Hàm chính thực hiện toàn bộ quy trình."""
    print("\n" + "=" * 50)
    print("HỆ THỐNG PHÂN LOẠI EMAIL SPAM SỬ DỤNG NAIVE BAYES")
    print("=" * 50 + "\n")

    # Thiết lập các thông số
    dataset_name = 'enron'  # 'spamassassin' hoặc 'enron'
    model_type = 'multinomial'  # 'multinomial', 'bernoulli', 'complement', 'custom'
    vectorizer_type = 'tfidf'  # 'tfidf', 'count', 'binary'
    use_existing_data = True  # True: sử dụng dữ liệu đã tải (nếu có)
    limit_emails = 100  # Số lượng email để xử lý (None để xử lý tất cả)

    # 1. Tải và xử lý dữ liệu
    print(f"BƯỚC 1: TẢI VÀ XỬ LÝ DỮ LIỆU ({dataset_name.upper()})")
    data_loader = EmailDataLoader(dataset_name)

    # Kiểm tra xem dữ liệu đã được xử lý chưa
    csv_path = os.path.join(data_loader.processed_dir, 'email_data.csv')
    if os.path.exists(csv_path) and use_existing_data:
        print(f"Đang tải dữ liệu đã xử lý từ {csv_path}")
        email_df = data_loader.load_from_csv()
    else:
        print("Đang tải và xử lý bộ dữ liệu mới...")
        email_df = data_loader.process_dataset(limit=limit_emails)

    if email_df is None or len(email_df) == 0:
        print("Không thể tải dữ liệu. Vui lòng kiểm tra lại cài đặt.")
        return

    print(
        f"Đã tải {len(email_df)} email ({email_df['label'].value_counts()['spam']} spam, {email_df['label'].value_counts()['ham']} ham)")

    # 2. Tiền xử lý dữ liệu
    print("\nBƯỚC 2: TIỀN XỬ LÝ DỮ LIỆU")
    preprocessor = EmailPreprocessor()
    data = preprocessor.prepare_data(email_df, vectorizer_type=vectorizer_type)

    print(f"Số mẫu huấn luyện: {data['X_train'].shape[0]}")
    print(f"Số mẫu kiểm tra: {data['X_test'].shape[0]}")
    print(f"Số đặc trưng: {data['X_train'].shape[1]}")

    # # 3. Huấn luyện mô hình
    # print(f"\nBƯỚC 3: HUẤN LUYỆN MÔ HÌNH ({model_type.upper()})")
    # model = NaiveBayesModel(model_type=model_type)
    #
    # start_time = time.time()
    # model.fit(data['X_train'], data['y_train'])
    # train_time = time.time() - start_time
    #
    # print(f"Thời gian huấn luyện: {train_time:.2f} giây")

    # # 4. Đánh giá mô hình
    # print("\nBƯỚC 4: ĐÁNH GIÁ MÔ HÌNH")
    # start_time = time.time()
    # y_pred = model.predict(data['X_test'])
    # predict_time = time.time() - start_time
    #
    # y_prob = model.predict_proba(data['X_test'])[:, 1] if hasattr(model, 'predict_proba') else None
    #
    # evaluator = ModelEvaluator(data['y_test'], y_pred, y_prob, data['feature_names'])
    # metrics = evaluator.calculate_metrics()
    # evaluator.print_metrics(metrics)
    #
    # print(f"Thời gian dự đoán trung bình: {(predict_time / len(data['X_test'])) * 1000:.2f} ms/email")
    #
    # # 5. Phân loại một email mẫu
    # print("\nBƯỚC 5: PHÂN LOẠI EMAIL MẪU")
    #
    # # Email mẫu
    # sample_emails = [
    #     "Congratulations! You've won a $1,000 gift card. Click here to claim your prize now!",
    #     "Hi team, please find the project report attached. Let's discuss this in our meeting tomorrow."
    # ]
    #
    # # Tiền xử lý email mẫu
    # for i, email_text in enumerate(sample_emails):
    #     print(f"\nPhân loại email mẫu #{i + 1}:")
    #     print(f"Nội dung: {email_text}")
    #
    #     # Tạo DataFrame từ email mẫu
    #     sample_df = pd.DataFrame({
    #         'body': [email_text],
    #         'label': ['unknown']
    #     })
    #
    #     # Tiền xử lý
    #     processed_df = preprocessor.process_emails(sample_df)
    #
    #     # Véc-tơ hóa
    #     sample_X = data['vectorizer'].transform([processed_df['processed_text'].iloc[0]])
    #
    #     # Dự đoán
    #     prediction = model.predict(sample_X)[0]
    #     probability = model.predict_proba(sample_X)[0][1] if hasattr(model, 'predict_proba') else None
    #
    #     # In kết quả
    #     result = "SPAM" if prediction == 1 else "HAM"
    #     confidence = f"({probability * 100:.2f}%)" if probability is not None else ""
    #     print(f"Kết quả phân loại: {result} {confidence}")
    #
    #     # Nếu có phân tích đặc trưng, hiển thị các từ quan trọng
    #     if hasattr(model, 'get_feature_importances'):
    #         importances = model.get_feature_importances(data['feature_names'])
    #         if importances:
    #             print("\nCác từ quan trọng nhất:")
    #             top_category = 'spam' if prediction == 1 else 'ham'
    #             for word, score in importances[top_category][:5]:
    #                 print(f"- {word}: {score:.4f}")
    #
    # # 6. Lưu mô hình (tùy chọn)
    # model_filename = f"{model_type}_{dataset_name}_model.joblib"
    # model_path = model.save_model(model_filename)
    # print(f"\nĐã lưu mô hình vào: {model_path}")
    #
    # print("\nQuy trình hoàn tất!")


if __name__ == "__main__":
    main()