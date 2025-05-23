"""
Bước 4: Đánh giá mô hình Custom Naive Bayes.
"""

import os
import pickle
import sys
import time
import numpy as np
import matplotlib.pyplot as plt

# Import từ project
from src.models.naive_bayes import CustomNaiveBayes
from src.evaluation.metrics import ModelEvaluator


def evaluate_model():
    """Đánh giá hiệu suất của mô hình Custom Naive Bayes."""
    print("\n" + "=" * 50)
    print("BƯỚC 4: ĐÁNH GIÁ MÔ HÌNH (CUSTOM NAIVE BAYES)")
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
    model_params = model_data.get('model_params', {})
    data = model_data['data']

    print(f"Đã tải thông tin mô hình ({model_type}) và dữ liệu kiểm tra.")
    print(f"Tham số mô hình: {model_params}")

    # Tải mô hình Custom Naive Bayes
    print(f"Đang tải mô hình từ {model_path}...")
    model = CustomNaiveBayes.load_model(model_path)

    # Hiển thị thông tin mô hình
    print(f"Mô hình đã tải:")
    print(f"  - Số lớp: {len(model.classes_)}")
    print(f"  - Các lớp: {model.classes_}")
    print(f"  - Số đặc trưng: {model.n_features_}")
    print(f"  - Alpha (Laplace smoothing): {model.alpha}")

    # Đánh giá mô hình
    print("\nĐang đánh giá mô hình trên tập kiểm tra...")
    y_test = data['y_test']
    X_test = data['X_test']

    print(f"Kích thước tập kiểm tra: {X_test.shape}")

    # Thực hiện dự đoán
    start_time = time.time()
    y_pred = model.predict(X_test)
    predict_time = time.time() - start_time

    # Tính xác suất
    start_time = time.time()
    y_prob = model.predict_proba(X_test)[:, 1]  # Xác suất của lớp positive (spam)
    proba_time = time.time() - start_time

    print(f"Thời gian dự đoán: {predict_time:.4f} giây")
    print(f"Thời gian tính xác suất: {proba_time:.4f} giây")

    # Tạo đối tượng đánh giá
    evaluator = ModelEvaluator(y_test, y_pred, y_prob, data['feature_names'])

    # Tính các chỉ số đánh giá
    metrics = evaluator.calculate_metrics()

    # In các chỉ số
    print("\nKẾT QUẢ ĐÁNH GIÁ MÔ HÌNH CUSTOM NAIVE BAYES:")
    evaluator.print_metrics(metrics)

    # Vẽ ma trận nhầm lẫn
    print("\nVẽ ma trận nhầm lẫn...")
    evaluator.plot_confusion_matrix(metrics)

    # Vẽ đường cong ROC
    print("Vẽ đường cong ROC...")
    evaluator.plot_roc_curve(metrics)

    # Vẽ đường cong Precision-Recall
    print("Vẽ đường cong Precision-Recall...")
    evaluator.plot_precision_recall_curve(metrics)

    # Phân tích đặc trưng quan trọng cho Custom Naive Bayes
    print("\nPhân tích đặc trưng quan trọng...")
    analyze_feature_importance(model, data['feature_names'])

    # Thống kê chi tiết về mô hình
    print("\nTHỐNG KÊ CHI TIẾT VỀ MÔ HÌNH:")
    print(f"Tổng thời gian dự đoán: {predict_time:.4f} giây")
    print(f"Thời gian dự đoán trung bình: {(predict_time / len(y_test)) * 1000:.4f} ms/email")

    # Phân tích phân phối xác suất
    spam_probs = y_prob[y_test == 1]  # Xác suất của các email spam thật
    ham_probs = y_prob[y_test == 0]   # Xác suất của các email ham thật

    print(f"\nPhân tích phân phối xác suất:")
    print(f"Email SPAM thật - Xác suất trung bình: {np.mean(spam_probs):.4f}")
    print(f"Email HAM thật - Xác suất trung bình: {np.mean(ham_probs):.4f}")

    # Lưu kết quả đánh giá
    output_file = 'evaluation_results.pkl'
    with open(output_file, 'wb') as f:
        pickle.dump({
            'metrics': metrics,
            'model_type': model_type,
            'model_path': model_path,
            'model_params': model_params,
            'feature_analysis': analyze_feature_importance(model, data['feature_names'], return_data=True)
        }, f)

    print(f"\nĐã lưu kết quả đánh giá vào file {output_file} để tham khảo sau")

    return metrics, model


def analyze_feature_importance(model, feature_names, top_n=20, return_data=False):
    """
    Phân tích tầm quan trọng của các đặc trưng trong Custom Naive Bayes.
    """
    if not hasattr(model, 'feature_log_prob_'):
        print("Mô hình không hỗ trợ phân tích đặc trưng.")
        return None

    # Tính toán log odds ratio
    # feature_log_prob_[0] là cho class 0 (ham)
    # feature_log_prob_[1] là cho class 1 (spam)
    log_odds_ratio = model.feature_log_prob_[1] - model.feature_log_prob_[0]

    # Tìm các đặc trưng quan trọng nhất cho spam
    top_spam_indices = np.argsort(log_odds_ratio)[-top_n:][::-1]

    # Tìm các đặc trưng quan trọng nhất cho ham
    top_ham_indices = np.argsort(log_odds_ratio)[:top_n]

    spam_features = [(feature_names[i], log_odds_ratio[i]) for i in top_spam_indices]
    ham_features = [(feature_names[i], log_odds_ratio[i]) for i in top_ham_indices]

    if not return_data:
        print(f"\nTop {top_n} từ quan trọng nhất cho SPAM:")
        for i, (word, score) in enumerate(spam_features):
            print(f"  {i+1:2d}. {word:15s} (log odds ratio: {score:8.4f})")

        print(f"\nTop {top_n} từ quan trọng nhất cho HAM:")
        for i, (word, score) in enumerate(ham_features):
            print(f"  {i+1:2d}. {word:15s} (log odds ratio: {score:8.4f})")

        # Vẽ biểu đồ
        plot_feature_importance_custom(spam_features[:10], ham_features[:10])

    return {
        'spam_features': spam_features,
        'ham_features': ham_features,
        'log_odds_ratio': log_odds_ratio
    }


def plot_feature_importance_custom(spam_features, ham_features):
    """Vẽ biểu đồ tầm quan trọng của các đặc trưng."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Biểu đồ cho SPAM
    words_spam = [item[0] for item in spam_features]
    scores_spam = [item[1] for item in spam_features]

    ax1.barh(range(len(words_spam)), scores_spam, color='red', alpha=0.7)
    ax1.set_yticks(range(len(words_spam)))
    ax1.set_yticklabels(words_spam)
    ax1.set_xlabel('Log Odds Ratio')
    ax1.set_title('Top từ quan trọng cho SPAM')
    ax1.grid(True, alpha=0.3)

    # Biểu đồ cho HAM
    words_ham = [item[0] for item in ham_features]
    scores_ham = [abs(item[1]) for item in ham_features]  # Lấy giá trị tuyệt đối

    ax2.barh(range(len(words_ham)), scores_ham, color='green', alpha=0.7)
    ax2.set_yticks(range(len(words_ham)))
    ax2.set_yticklabels(words_ham)
    ax2.set_xlabel('|Log Odds Ratio|')
    ax2.set_title('Top từ quan trọng cho HAM')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('custom_naive_bayes_feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    evaluate_model()
    # Để giữ cửa sổ đồ thị hiển thị
    plt.show()