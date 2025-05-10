"""
Bước 5: Phân loại các email mẫu bằng mô hình đã huấn luyện.
"""

import os
import pickle
import sys
import pandas as pd
import numpy as np

# Import từ project
from src.models.naive_bayes import NaiveBayesModel
from src.data.preprocessor import EmailPreprocessor


def classify_samples():
    """Phân loại các email mẫu bằng mô hình đã huấn luyện."""
    print("\n" + "=" * 50)
    print("BƯỚC 5: PHÂN LOẠI EMAIL MẪU")
    print("=" * 50 + "\n")

    # Tải thông tin mô hình từ bước trước
    input_file = 'trained_model_data.pkl'
    if not os.path.exists(input_file):
        print(f"Không tìm thấy file {input_file}. Vui lòng chạy step3_train_model.py trước.")
        sys.exit(1)

    with open(input_file, 'rb') as f:
        model_data = pickle.load(f)

    model_path = model_data['model_path']
    model_type = model_data['model_type']
    data = model_data['data']

    print(f"Đã tải thông tin mô hình ({model_type}).")

    # Tải mô hình
    print(f"Đang tải mô hình từ {model_path}...")
    if 'custom' in model_type:
        from src.models.naive_bayes import CustomNaiveBayes
        model = CustomNaiveBayes.load_model(model_path)
    else:
        model = NaiveBayesModel.load_model(model_path)

    # Khởi tạo preprocessor
    preprocessor = EmailPreprocessor()

    # Danh sách các email mẫu để phân loại
    sample_emails = [
        "Congratulations! You've won a $1,000 gift card. Click here to claim your prize now! Limited time offer.",
        "URGENT: Your account has been suspended. Please verify your information by clicking this link immediately!",
        "Hi team, please find the project report attached. Let's discuss this in our meeting tomorrow at 2pm.",
        "Dear Mom, I'll be home for dinner tonight. Can't wait to see you!",
        # Thêm email mẫu của riêng bạn ở đây
    ]

    # Cho phép người dùng nhập email mẫu
    user_email = input("\nBạn có muốn nhập email riêng để phân loại không? (y/n): ").lower()
    if user_email == 'y':
        email_text = input("\nNhập nội dung email: ")
        sample_emails.append(email_text)

    # Phân loại từng email
    for i, email_text in enumerate(sample_emails):
        print(f"\n{'-' * 70}")
        print(f"Email #{i + 1}:")
        print(f"Nội dung: {email_text[:100]}{'...' if len(email_text) > 100 else ''}")

        # Tạo DataFrame từ email
        sample_df = pd.DataFrame({
            'body': [email_text],
            'label': ['unknown']
        })

        # Tiền xử lý
        processed_df = preprocessor.process_emails(sample_df)
        processed_text = processed_df['processed_text'].iloc[0]

        print(f"Sau khi tiền xử lý: {processed_text[:100]}{'...' if len(processed_text) > 100 else ''}")

        # Véc-tơ hóa
        sample_X = data['vectorizer'].transform([processed_text])

        # Dự đoán
        prediction = model.predict(sample_X)[0]
        probability = None
        if hasattr(model, 'predict_proba'):
            probability = model.predict_proba(sample_X)[0][1]

        # In kết quả
        result = "SPAM" if prediction == 1 else "HAM (không phải spam)"
        confidence = f" với độ tin cậy {probability * 100:.2f}%" if probability is not None else ""
        print(f"\nKẾT QUẢ PHÂN LOẠI: {result}{confidence}")

        # Hiển thị từ quan trọng nếu mô hình hỗ trợ
        if hasattr(model, 'get_feature_importances'):
            importances = model.get_feature_importances(data['feature_names'])
            if importances:
                # Tìm các từ trong email đang được xét
                print("\nCác từ đóng góp vào kết quả phân loại:")
                words = processed_text.split()

                # Từ điển từ -> điểm số
                word_scores = {}
                for category in ['spam', 'ham']:
                    for word, score in importances[category]:
                        if word in words:
                            word_scores[word] = (score, category)

                # Sắp xếp theo giá trị tuyệt đối của điểm số
                sorted_words = sorted(word_scores.items(),
                                      key=lambda x: abs(x[1][0] if category == 'spam' else -x[1][0]),
                                      reverse=True)

                # Hiển thị top 5 từ (hoặc ít hơn nếu không đủ)
                for word, (score, category) in sorted_words[:5]:
                    indicator = "spam" if category == 'spam' else "ham"
                    print(f"- '{word}': Chỉ báo {indicator} (điểm số: {score:.4f})")

    print(f"\n{'-' * 70}")
    print("Phân loại hoàn tất!")


if __name__ == "__main__":
    classify_samples()