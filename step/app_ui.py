import os
import pickle
import sys
import pandas as pd
import numpy as np
import time
import tkinter as tk
from tkinter import filedialog, scrolledtext, ttk, messagebox
from tkinter.font import Font
import traceback

# Import từ project
from src.models.naive_bayes import NaiveBayesModel
from src.data.preprocessor import EmailPreprocessor


class SpamClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Ứng dụng Phân loại Email Spam")
        self.root.geometry("800x700")
        self.root.resizable(True, True)
        self.root.configure(bg="#e0e7ff")  # Màu nền pastel tím nhạt

        # Initialize variables
        self.model = None
        self.model_type = None
        self.data = None
        self.preprocessor = None

        # Load model
        try:
            self.load_model()
            print(f"Đã tải mô hình thành công: {self.model_type}")
        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể tải mô hình: {str(e)}")
            print(f"Chi tiết lỗi: {traceback.format_exc()}")
            sys.exit(1)

        # Setup UI
        self.setup_ui()

    def load_model(self):
        """Tải mô hình và dữ liệu liên quan."""
        input_file = 'trained_model_data.pkl'
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Không tìm thấy file {input_file}. Vui lòng chạy step3_train_model.py trước.")

        with open(input_file, 'rb') as f:
            model_data = pickle.load(f)

        self.model_path = model_data['model_path']
        self.model_type = model_data['model_type']
        self.data = model_data['data']

        print(f"Đang tải mô hình loại: {self.model_type}")
        print(f"Đường dẫn mô hình: {self.model_path}")

        # Tải mô hình dựa trên loại
        if 'custom' in self.model_type:
            from src.models.naive_bayes import CustomNaiveBayes
            self.model = CustomNaiveBayes.load_model(self.model_path)
            print("Đã tải Custom Naive Bayes")
        else:
            self.model = NaiveBayesModel.load_model(self.model_path)
            print("Đã tải Sklearn Naive Bayes")

        self.preprocessor = EmailPreprocessor()

        # Kiểm tra cấu trúc dữ liệu
        print(f"Có vectorizer: {'vectorizer' in self.data}")
        print(f"Có feature_names: {'feature_names' in self.data}")
        if 'feature_names' in self.data:
            print(f"Số lượng features: {len(self.data['feature_names'])}")

    def setup_ui(self):
        """Thiết lập giao diện người dùng với theme mới."""
        # Style cho ttk
        style = ttk.Style()
        style.configure("TNotebook", background="#e0e7ff", borderwidth=0)
        style.configure("TNotebook.Tab", padding=[10, 5], font=("Segoe UI", 10))
        style.map("TNotebook.Tab", background=[("selected", "#6366f1")], foreground=[("selected", "white")])

        # Frame chính
        main_frame = tk.Frame(self.root, bg="#e0e7ff")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Tiêu đề
        title_label = tk.Label(
            main_frame,
            text="PHÂN LOẠI EMAIL SPAM",
            font=("Segoe UI", 18, "bold"),
            bg="#e0e7ff",
            fg="#1e3a8a"  # Màu xanh đậm
        )
        title_label.pack(pady=(0, 20))

        # Thông tin mô hình
        model_frame = tk.LabelFrame(
            main_frame,
            text="Thông tin mô hình",
            font=("Segoe UI", 11, "bold"),
            bg="#f5f5ff",
            fg="#1e3a8a",
            padx=15,
            pady=15,
            relief=tk.FLAT,
            borderwidth=2
        )
        model_frame.pack(fill=tk.X, pady=(0, 20))

        model_info = f"Loại mô hình: {self.model_type.capitalize()}\nĐường dẫn: {self.model_path}"
        model_label = tk.Label(
            model_frame,
            text=model_info,
            justify=tk.LEFT,
            bg="#f5f5ff",
            fg="#1f2937",
            font=("Segoe UI", 10)
        )
        model_label.pack(anchor=tk.W)

        # Frame nhập liệu
        input_frame = tk.LabelFrame(
            main_frame,
            text="Nhập email",
            font=("Segoe UI", 11, "bold"),
            bg="#f5f5ff",
            fg="#1e3a8a",
            padx=15,
            pady=15,
            relief=tk.FLAT
        )
        input_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 20))

        # Tab control
        tab_control = ttk.Notebook(input_frame)

        # Tab nhập text
        text_tab = tk.Frame(tab_control, bg="#f5f5ff")
        tab_control.add(text_tab, text="Nhập văn bản")

        self.text_input = scrolledtext.ScrolledText(
            text_tab,
            wrap=tk.WORD,
            height=10,
            font=("Segoe UI", 10),
            bg="white",
            fg="#1f2937",
            insertbackground="#1e3a8a",
            relief=tk.FLAT,
            borderwidth=1
        )
        self.text_input.pack(fill=tk.BOTH, expand=True, pady=(10, 5))

        text_buttons_frame = tk.Frame(text_tab, bg="#f5f5ff")
        text_buttons_frame.pack(fill=tk.X, pady=5)

        def button_style(btn, bg_color, hover_color):
            btn.configure(bg=bg_color, fg="white", font=("Segoe UI", 10, "bold"), relief=tk.FLAT)
            btn.bind("<Enter>", lambda e: btn.configure(bg=hover_color))
            btn.bind("<Leave>", lambda e: btn.configure(bg=bg_color))

        classify_text_btn = tk.Button(
            text_buttons_frame,
            text="Phân loại",
            command=self.classify_text
        )
        button_style(classify_text_btn, "#4f46e5", "#6366f1")  # Indigo
        classify_text_btn.pack(side=tk.LEFT, padx=5)

        clear_text_btn = tk.Button(
            text_buttons_frame,
            text="Xóa",
            command=lambda: self.text_input.delete(1.0, tk.END)
        )
        button_style(clear_text_btn, "#ef4444", "#dc2626")  # Red
        clear_text_btn.pack(side=tk.LEFT, padx=5)

        # Thêm nút debug
        debug_btn = tk.Button(
            text_buttons_frame,
            text="Debug Info",
            command=self.show_debug_info
        )
        button_style(debug_btn, "#10b981", "#059669")  # Green
        debug_btn.pack(side=tk.LEFT, padx=5)

        # Tab chọn file
        file_tab = tk.Frame(tab_control, bg="#f5f5ff")
        tab_control.add(file_tab, text="Chọn file")

        file_frame = tk.Frame(file_tab, bg="#f5f5ff")
        file_frame.pack(fill=tk.X, pady=10)

        self.file_path_var = tk.StringVar()
        file_entry = tk.Entry(
            file_frame,
            textvariable=self.file_path_var,
            width=50,
            bg="white",
            fg="#1f2937",
            font=("Segoe UI", 10),
            relief=tk.FLAT,
            borderwidth=1
        )
        file_entry.pack(side=tk.LEFT, padx=(0, 5), fill=tk.X, expand=True)

        browse_btn = tk.Button(
            file_frame,
            text="Chọn file",
            command=self.browse_file
        )
        button_style(browse_btn, "#3b82f6", "#2563eb")  # Blue
        browse_btn.pack(side=tk.RIGHT)

        classify_file_btn = tk.Button(
            file_tab,
            text="Phân loại",
            command=self.classify_file
        )
        button_style(classify_file_btn, "#4f46e5", "#6366f1")
        classify_file_btn.pack(pady=5)

        tab_control.pack(fill=tk.BOTH, expand=True)

        # Frame kết quả
        result_frame = tk.LabelFrame(
            main_frame,
            text="Kết quả phân loại",
            font=("Segoe UI", 11, "bold"),
            bg="#f5f5ff",
            fg="#1e3a8a",
            padx=15,
            pady=15,
            relief=tk.FLAT
        )
        result_frame.pack(fill=tk.BOTH, expand=True)

        self.result_label = tk.Label(
            result_frame,
            text="Chưa có kết quả phân loại",
            font=("Segoe UI", 12, "bold"),
            bg="#f5f5ff",
            fg="#1f2937"
        )
        self.result_label.pack(pady=10)

        self.progress = ttk.Progressbar(
            result_frame,
            orient=tk.HORIZONTAL,
            length=100,
            mode='determinate',
            style="Custom.Horizontal.TProgressbar"
        )
        style.configure("Custom.Horizontal.TProgressbar", troughcolor="#e0e7ff", background="#4f46e5")
        self.progress.pack(fill=tk.X, pady=5)

        details_frame = tk.Frame(result_frame, bg="#f5f5ff")
        details_frame.pack(fill=tk.BOTH, expand=True)

        self.details_text = scrolledtext.ScrolledText(
            details_frame,
            wrap=tk.WORD,
            height=10,
            font=("Segoe UI", 10),
            bg="white",
            fg="#1f2937",
            relief=tk.FLAT,
            borderwidth=1
        )
        self.details_text.pack(fill=tk.BOTH, expand=True)
        self.details_text.config(state=tk.DISABLED)

    def show_debug_info(self):
        """Hiển thị thông tin debug."""
        debug_info = f"""
Debug Information:
- Model type: {self.model_type}
- Model path: {self.model_path}
- Model loaded: {self.model is not None}
- Data keys: {list(self.data.keys()) if self.data else 'None'}
- Vectorizer available: {'vectorizer' in self.data if self.data else False}
- Feature names available: {'feature_names' in self.data if self.data else False}
"""
        if self.data and 'feature_names' in self.data:
            debug_info += f"- Number of features: {len(self.data['feature_names'])}\n"

        messagebox.showinfo("Debug Info", debug_info)

    def browse_file(self):
        """Mở hộp thoại chọn file."""
        file_path = filedialog.askopenfilename(
            title="Chọn file email",
            filetypes=[("Text files", "*.txt"), ("Email files", "*.eml"), ("All files", "*.*")]
        )
        if file_path:
            self.file_path_var.set(file_path)

    def classify_text(self):
        """Phân loại email từ văn bản nhập vào."""
        email_text = self.text_input.get(1.0, tk.END).strip()
        if not email_text:
            messagebox.showwarning("Cảnh báo", "Vui lòng nhập nội dung email.")
            return
        self.classify_and_display(email_text)

    def classify_file(self):
        """Phân loại email từ file."""
        file_path = self.file_path_var.get()
        if not file_path or not os.path.exists(file_path):
            messagebox.showwarning("Cảnh báo", "Vui lòng chọn file email hợp lệ.")
            return
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                email_text = f.read()
            self.classify_and_display(email_text)
        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể đọc file: {str(e)}")

    def classify_and_display(self, email_text):
        """Phân loại email và hiển thị kết quả."""
        try:
            print(f"Bắt đầu phân loại email, độ dài: {len(email_text)} ký tự")

            self.progress['value'] = 20
            self.root.update_idletasks()

            start_time = time.time()

            # Tạo DataFrame
            sample_df = pd.DataFrame({
                'body': [email_text],
                'label': ['unknown']
            })
            print("Đã tạo DataFrame")

            self.progress['value'] = 40
            self.root.update_idletasks()

            # Tiền xử lý
            processed_df = self.preprocessor.process_emails(sample_df)
            processed_text = processed_df['processed_text'].iloc[0]
            print(f"Đã tiền xử lý, độ dài sau xử lý: {len(processed_text)} ký tự")

            self.progress['value'] = 60
            self.root.update_idletasks()

            # Kiểm tra vectorizer
            if 'vectorizer' not in self.data:
                raise ValueError("Không tìm thấy vectorizer trong dữ liệu mô hình")

            # Véc-tơ hóa
            sample_X = self.data['vectorizer'].transform([processed_text])
            print(f"Đã véc-tơ hóa, shape: {sample_X.shape}")

            # Chuyển đổi sparse matrix thành dense array nếu cần (cho Custom NB)
            if 'custom' in self.model_type and hasattr(sample_X, 'toarray'):
                sample_X = sample_X.toarray()
                print("Đã chuyển đổi thành dense array")

            self.progress['value'] = 80
            self.root.update_idletasks()

            # Dự đoán
            prediction = self.model.predict(sample_X)
            print(f"Prediction shape: {prediction.shape if hasattr(prediction, 'shape') else 'scalar'}")
            print(f"Prediction: {prediction}")

            # Lấy giá trị đầu tiên nếu là array
            if hasattr(prediction, 'shape') and len(prediction.shape) > 0:
                prediction_value = prediction[0]
            else:
                prediction_value = prediction

            print(f"Prediction value: {prediction_value}")

            # Tính xác suất
            probability = None
            if hasattr(self.model, 'predict_proba'):
                try:
                    prob_result = self.model.predict_proba(sample_X)
                    print(f"Probability shape: {prob_result.shape}")
                    print(f"Probability result: {prob_result}")

                    # Xử lý xác suất an toàn
                    if len(prob_result.shape) == 2 and prob_result.shape[1] >= 2:
                        probability = prob_result[0, 1]  # Xác suất của class 1 (spam)
                    elif len(prob_result.shape) == 1 and len(prob_result) >= 2:
                        probability = prob_result[1]
                    else:
                        print(f"Unexpected probability shape: {prob_result.shape}")
                        probability = None

                except Exception as e:
                    print(f"Lỗi khi tính xác suất: {e}")
                    probability = None

            process_time = (time.time() - start_time) * 1000

            self.progress['value'] = 100
            self.root.update_idletasks()

            # Hiển thị kết quả
            spam_status = "SPAM" if prediction_value == 1 else "HAM (không phải spam)"
            confidence = f"{probability * 100:.2f}%" if probability is not None else "N/A"

            result_text = f"Phân loại: {spam_status}\nĐộ tin cậy: {confidence}\nThời gian xử lý: {process_time:.2f} ms"
            self.result_label.config(
                text=result_text,
                fg="#dc2626" if prediction_value == 1 else "#15803d",
                font=("Segoe UI", 12, "bold")
            )

            # Hiện thị chi tiết
            self.show_classification_details(processed_text, prediction_value)

        except Exception as e:
            error_msg = f"Lỗi khi phân loại email: {str(e)}"
            print(f"Chi tiết lỗi: {traceback.format_exc()}")
            messagebox.showerror("Lỗi", error_msg)
            self.progress['value'] = 0

    def show_classification_details(self, processed_text, prediction_value):
        """Hiển thị chi tiết phân loại."""
        try:
            self.details_text.config(state=tk.NORMAL)
            self.details_text.delete(1.0, tk.END)
            self.details_text.insert(tk.END, f"Văn bản sau khi tiền xử lý:\n{processed_text}\n\n")

            # Phân tích features cho Custom NB
            if 'custom' in self.model_type:
                self.analyze_custom_features(processed_text, prediction_value)
            elif hasattr(self.model, 'get_feature_importances'):
                self.analyze_sklearn_features(processed_text)

            self.details_text.config(state=tk.DISABLED)
            self.details_text.tag_configure("spam_word", foreground="#dc2626")
            self.details_text.tag_configure("ham_word", foreground="#15803d")

        except Exception as e:
            print(f"Lỗi khi hiển thị chi tiết: {e}")

    def analyze_custom_features(self, processed_text, prediction_value):
        """Phân tích features cho Custom Naive Bayes."""
        try:
            if not hasattr(self.model, 'feature_log_prob_'):
                self.details_text.insert(tk.END, "Mô hình không hỗ trợ phân tích feature importance.\n")
                return

            self.details_text.insert(tk.END, "Phân tích từ quan trọng (Custom Naive Bayes):\n")

            # Tính log odds ratio
            log_odds_ratio = self.model.feature_log_prob_[1] - self.model.feature_log_prob_[0]

            # Lấy từ trong văn bản
            words = processed_text.split()
            feature_names = self.data.get('feature_names', [])

            if not feature_names:
                self.details_text.insert(tk.END, "Không có thông tin về tên features.\n")
                return

            # Tạo mapping từ -> index
            word_to_idx = {word: idx for idx, word in enumerate(feature_names)}

            # Tìm từ quan trọng
            word_scores = []
            for word in words:
                if word in word_to_idx:
                    idx = word_to_idx[word]
                    score = log_odds_ratio[idx]
                    word_scores.append((word, score))

            # Sắp xếp theo importance
            word_scores.sort(key=lambda x: abs(x[1]), reverse=True)

            # Hiển thị top 10
            for word, score in word_scores[:10]:
                if score > 0:
                    indicator = "spam"
                    tag = "spam_word"
                else:
                    indicator = "ham"
                    tag = "ham_word"
                line = f"- '{word}': Chỉ báo {indicator} (log odds: {score:.4f})\n"
                self.details_text.insert(tk.END, line, (tag,))

        except Exception as e:
            print(f"Lỗi phân tích custom features: {e}")
            self.details_text.insert(tk.END, f"Lỗi phân tích: {str(e)}\n")

    def analyze_sklearn_features(self, processed_text):
        """Phân tích features cho Sklearn Naive Bayes."""
        try:
            importances = self.model.get_feature_importances(self.data['feature_names'])
            if importances:
                self.details_text.insert(tk.END, "Các từ quan trọng đóng góp vào kết quả phân loại:\n")
                words = processed_text.split()

                word_scores = {}
                for category in ['spam', 'ham']:
                    for word, score in importances[category]:
                        if word in words:
                            word_scores[word] = (score, category)

                sorted_words = sorted(word_scores.items(),
                                      key=lambda x: abs(x[1][0]),
                                      reverse=True)

                for word, (score, category) in sorted_words[:10]:
                    indicator = "spam" if category == 'spam' else "ham"
                    tag = indicator + "_word"
                    line = f"- '{word}': Chỉ báo {indicator} (điểm số: {score:.4f})\n"
                    self.details_text.insert(tk.END, line, (tag,))
        except Exception as e:
            print(f"Lỗi phân tích sklearn features: {e}")


def main():
    root = tk.Tk()
    app = SpamClassifierApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()