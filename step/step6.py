"""
Bước 6: Ứng dụng phân loại email tương tác đơn giản.
"""

import os
import pickle
import sys
import pandas as pd
import numpy as np
import time

# Import từ project
from src.models.naive_bayes import NaiveBayesModel
from src.data.preprocessor import EmailPreprocessor

def interactive_classifier():
    """Ứng dụng phân loại email tương tác đơn giản."""
    print("\n" + "="*50)
    print("ỨNG DỤNG PHÂN LOẠI EMAIL SPAM")
    print("="*50 + "\n")

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

    # Hàm phân loại email
    def classify_email(email_text):
        # Đo thời gian xử lý
        start_time = time.time()

        # Tạo DataFrame từ email
        sample_df = pd.DataFrame({
            'body': [email_text],
            'label': ['unknown']
        })

        # Tiền xử lý
        processed_df = preprocessor.process_emails(sample_df)
        processed_text = processed_df['processed_text'].iloc[0]

        # Véc-tơ hóa
        sample_X = data['vectorizer'].transform([processed_text])

        # Dự đoán
        prediction = model.predict(sample_X)[0]
        probability = None
        if hasattr(model, 'predict_proba'):
            probability = model.predict_proba(sample_X)[0][1]

        # Tính thời gian xử lý
        process_time = (time.time() - start_time) * 1000  # Đổi sang milliseconds

        return {
            'is_spam': prediction == 1,
            'probability': probability,
            'processed_text': processed_text,
            'process_time': process_time
        }

    # Vòng lặp ứng dụng tương tác
    while True:
        print("\n" + "-"*70)
        print("Nhập nội dung email để phân loại (gõ 'exit' để thoát):")
        email_text = input("> ")

        if email_text.lower() == 'exit':
            print("\nCảm ơn bạn đã sử dụng ứng dụng phân loại email spam!")
            break

        if not email_text.strip():
            print("Email không được để trống. Vui lòng thử lại.")
            continue

        # Phân loại email
        result = classify_email(email_text)

        # Hiển thị kết quả
        print("\nKẾT QUẢ PHÂN LOẠI:")

        spam_status = "SPAM" if result['is_spam'] else "HAM (không phải spam)"
        confidence = f"{result['probability']*100:.2f}%" if result['probability'] is not None else "N/A"

        print(f"- Phân loại: {spam_status}")
        print(f"- Độ tin cậy: {confidence}")
        print(f"- Thời gian xử lý: {result['process_time']:.2f} ms")

        # Hiển thị chi tiết khi được yêu cầu
        show_details = input("\nBạn có muốn xem chi tiết phân tích không? (y/n): ").lower()
        if show_details == 'y':
            print(f"\nVăn bản sau khi tiền xử lý: {result['processed_text']}")

            # Hiển thị từ quan trọng nếu mô hình hỗ trợ
            if hasattr(model, 'get_feature_importances'):
                importances = model.get_feature_importances(data['feature_names'])
                if importances:
                    # Tìm các từ trong email đang được xét
                    print("\nCác từ quan trọng đóng góp vào kết quả phân loại:")
                    words = result['processed_text'].split()

                    # Từ điển từ -> điểm số
                    word_scores = {}
                    for category in ['spam', 'ham']:
                        for word, score in importances[category]:
                            if word in words:
                                word_scores[word] = (score, category)

                    # Sắp xếp theo giá trị tuyệt đối của điểm số
                    sorted_words = sorted(word_scores.items(),
                                         key=lambda x: abs(x[1][0]),
                                         reverse=True)

                    # Hiển thị các từ
                    for word, (score, category) in sorted_words[:10]:  # Top 10 từ
                        indicator = "spam" if category == 'spam' else "ham"
                        print(f"- '{word}': Chỉ báo {indicator} (điểm số: {score:.4f})")

if __name__ == "__main__":
    interactive_classifier()