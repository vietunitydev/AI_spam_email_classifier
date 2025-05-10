"""
Bước 4: Đánh giá mô hình Naive Bayes.
"""

import os
import pickle
import sys
import time
import numpy as np
import matplotlib.pyplot as plt

# Import từ project
from src.models.naive_bayes import NaiveBayesModel
from src.evaluation.metrics import ModelEvaluator


def evaluate_model():
    """Đánh giá hiệu suất của mô hình Naive Bayes."""
    print("\n" + "=" * 50)
    print("BƯỚC 4: ĐÁNH GIÁ MÔ HÌNH")
    print("=" * 50 + "\n")

    # Tải thông tin mô hình và dữ liệu kiểm tra từ bước 3
    input_file = 'trained_model_data.pkl'
    if not os.path.exists(input_file):
        print(f"Không tìm thấy file {input_file}. Vui lòng chạy step3_train_model.py trước.")
        sys.exit(1)

    with open(input_file, 'rb') as f:
        model_data = pickle.load(f)

    model_path = model_data['model_path']
    model_type = model_data['model_type']
    data = model_data['data']

    print(f"Đã tải thông tin mô hình ({model_type}) và dữ liệu kiểm tra.")

    # Tải mô hình
    print(f"Đang tải mô hình từ {model_path}...")
    if 'custom' in model_type:
        from src.models.naive_bayes import CustomNaiveBayes
        model = CustomNaiveBayes.load_model(model_path)
    else:
        model = NaiveBayesModel.load_model(model_path)

    # Đánh giá mô hình
    print("Đang đánh giá mô hình trên tập kiểm tra...")
    y_test = data['y_test']
    X_test = data['X_test']

    # Thực hiện dự đoán
    start_time = time.time()
    y_pred = model.predict(X_test)
    predict_time = time.time() - start_time

    # Tính xác suất (nếu mô hình hỗ trợ)
    y_prob = None
    if hasattr(model, 'predict_proba'):
        y_prob = model.predict_proba(X_test)[:, 1]

    # Tạo đối tượng đánh giá
    evaluator = ModelEvaluator(y_test, y_pred, y_prob, data['feature_names'])

    # Tính các chỉ số đánh giá
    metrics = evaluator.calculate_metrics()

    # In các chỉ số
    print("\nKẾT QUẢ ĐÁNH GIÁ MÔ HÌNH:")
    evaluator.print_metrics(metrics)

    # Vẽ ma trận nhầm lẫn
    print("\nVẽ ma trận nhầm lẫn...")
    evaluator.plot_confusion_matrix(metrics)

    # Vẽ đường cong ROC nếu có xác suất
    if y_prob is not None:
        print("Vẽ đường cong ROC...")
        evaluator.plot_roc_curve(metrics)

        print("Vẽ đường cong Precision-Recall...")
        evaluator.plot_precision_recall_curve(metrics)

    # Vẽ biểu đồ tầm quan trọng của các đặc trưng
    if hasattr(model, 'get_feature_importances'):
        print("Phân tích tầm quan trọng của các đặc trưng...")
        evaluator.plot_feature_importance(model, top_n=15)

    print(f"\nTổng thời gian dự đoán: {predict_time:.2f} giây")
    print(f"Thời gian dự đoán trung bình: {(predict_time / len(y_test)) * 1000:.2f} ms/email")

    # Lưu kết quả đánh giá
    output_file = 'evaluation_results.pkl'
    with open(output_file, 'wb') as f:
        pickle.dump({
            'metrics': metrics,
            'model_type': model_type,
            'model_path': model_path
        }, f)

    print(f"Đã lưu kết quả đánh giá vào file {output_file} để tham khảo sau")

    return metrics, model


if __name__ == "__main__":
    evaluate_model()
    # Để giữ cửa sổ đồ thị hiển thị
    plt.show()