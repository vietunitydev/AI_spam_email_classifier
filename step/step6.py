import os
import pickle
import sys
import pandas as pd
import numpy as np
import time
import tkinter as tk
from tkinter import filedialog, scrolledtext, ttk, messagebox
from tkinter.font import Font

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

        # Load model
        self.load_model()

        # Setup UI
        self.setup_ui()

    def load_model(self):
        """Tải mô hình và dữ liệu liên quan."""
        input_file = 'trained_model_data.pkl'
        if not os.path.exists(input_file):
            messagebox.showerror("Lỗi", f"Không tìm thấy file {input_file}. Vui lòng chạy step3_train_model.py trước.")
            sys.exit(1)

        with open(input_file, 'rb') as f:
            model_data = pickle.load(f)

        self.model_path = model_data['model_path']
        self.model_type = model_data['model_type']
        self.data = model_data['data']

        if 'custom' in self.model_type:
            from src.models.naive_bayes import CustomNaiveBayes
            self.model = CustomNaiveBayes.load_model(self.model_path)
        else:
            self.model = NaiveBayesModel.load_model(self.model_path)

        self.preprocessor = EmailPreprocessor()

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
        self.progress['value'] = 20
        self.root.update_idletasks()

        try:
            start_time = time.time()
            sample_df = pd.DataFrame({
                'body': [email_text],
                'label': ['unknown']
            })

            self.progress['value'] = 40
            self.root.update_idletasks()

            processed_df = self.preprocessor.process_emails(sample_df)
            processed_text = processed_df['processed_text'].iloc[0]

            self.progress['value'] = 60
            self.root.update_idletasks()

            sample_X = self.data['vectorizer'].transform([processed_text])

            self.progress['value'] = 80
            self.root.update_idletasks()

            prediction = self.model.predict(sample_X)[0]
            probability = None
            if hasattr(self.model, 'predict_proba'):
                probability = self.model.predict_proba(sample_X)[0][1]

            process_time = (time.time() - start_time) * 1000

            self.progress['value'] = 100
            self.root.update_idletasks()

            spam_status = "SPAM" if prediction == 1 else "HAM (không phải spam)"
            confidence = f"{probability*100:.2f}%" if probability is not None else "N/A"

            result_text = f"Phân loại: {spam_status}\nĐộ tin cậy: {confidence}\nThời gian xử lý: {process_time:.2f} ms"
            self.result_label.config(
                text=result_text,
                fg="#dc2626" if prediction == 1 else "#15803d",
                font=("Segoe UI", 12, "bold")
            )

            self.details_text.config(state=tk.NORMAL)
            self.details_text.delete(1.0, tk.END)
            self.details_text.insert(tk.END, f"Văn bản sau khi tiền xử lý:\n{processed_text}\n\n")

            if hasattr(self.model, 'get_feature_importances'):
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
                        line = f"- '{word}': Chỉ báo {indicator} (điểm số: {score:.4f})\n"
                        self.details_text.insert(tk.END, line, (indicator + "_word"))

            self.details_text.config(state=tk.DISABLED)
            self.details_text.tag_configure("spam_word", foreground="#dc2626")
            self.details_text.tag_configure("ham_word", foreground="#15803d")

        except Exception as e:
            messagebox.showerror("Lỗi", f"Lỗi khi phân loại email: {str(e)}")
            self.progress['value'] = 0

def main():
    root = tk.Tk()
    app = SpamClassifierApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()