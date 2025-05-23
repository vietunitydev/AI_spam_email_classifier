"""
UI application for spam email classification.
"""

import os
import pickle
import sys
import pandas as pd
import numpy as np
import time
import tkinter as tk
from tkinter import filedialog, scrolledtext, ttk, messagebox
import traceback

from src.models.custom_naive_bayes import CustomNaiveBayes
from src.data.preprocessor import EmailPreprocessor


class SpamClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Spam Email Classifier")
        self.root.geometry("800x700")
        self.root.resizable(True, True)
        self.root.configure(bg="#e0e7ff")

        self.model = None
        self.model_type = None
        self.data = None
        self.preprocessor = None

        try:
            self.load_model()
            print(f"Model loaded successfully: {self.model_type}")
        except Exception as e:
            messagebox.showerror("Error", f"Cannot load model: {str(e)}")
            print(f"Error details: {traceback.format_exc()}")
            sys.exit(1)

        self.setup_ui()

    def load_model(self):
        input_file = 'trained_model_data.pkl'
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"File {input_file} not found. Please run train_model.py first.")

        with open(input_file, 'rb') as f:
            model_data = pickle.load(f)

        self.model_path = model_data['model_path']
        self.model_type = model_data['model_type']
        self.data = model_data['data']

        print(f"Loading model type: {self.model_type}")
        print(f"Model path: {self.model_path}")

        self.model = CustomNaiveBayes.load_model(self.model_path)
        print("Custom Naive Bayes loaded")

        self.preprocessor = EmailPreprocessor()

        print(f"Has vectorizer: {'vectorizer' in self.data}")
        print(f"Has feature_names: {'feature_names' in self.data}")
        if 'feature_names' in self.data:
            print(f"Number of features: {len(self.data['feature_names'])}")

    def setup_ui(self):
        style = ttk.Style()
        style.configure("TNotebook", background="#e0e7ff", borderwidth=0)
        style.configure("TNotebook.Tab", padding=[10, 5], font=("Segoe UI", 10))
        style.map("TNotebook.Tab", background=[("selected", "#6366f1")], foreground=[("selected", "white")])

        main_frame = tk.Frame(self.root, bg="#e0e7ff")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        title_label = tk.Label(
            main_frame,
            text="SPAM EMAIL CLASSIFIER",
            font=("Segoe UI", 18, "bold"),
            bg="#e0e7ff",
            fg="#1e3a8a"
        )
        title_label.pack(pady=(0, 20))

        model_frame = tk.LabelFrame(
            main_frame,
            text="Model Information",
            font=("Segoe UI", 11, "bold"),
            bg="#f5f5ff",
            fg="#1e3a8a",
            padx=15,
            pady=15,
            relief=tk.FLAT,
            borderwidth=2
        )
        model_frame.pack(fill=tk.X, pady=(0, 20))

        model_info = f"Model type: {self.model_type.capitalize()}\nPath: {self.model_path}"
        model_label = tk.Label(
            model_frame,
            text=model_info,
            justify=tk.LEFT,
            bg="#f5f5ff",
            fg="#1f2937",
            font=("Segoe UI", 10)
        )
        model_label.pack(anchor=tk.W)

        input_frame = tk.LabelFrame(
            main_frame,
            text="Input Email",
            font=("Segoe UI", 11, "bold"),
            bg="#f5f5ff",
            fg="#1e3a8a",
            padx=15,
            pady=15,
            relief=tk.FLAT
        )
        input_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 20))

        tab_control = ttk.Notebook(input_frame)

        text_tab = tk.Frame(tab_control, bg="#f5f5ff")
        tab_control.add(text_tab, text="Text Input")

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
            text="Classify",
            command=self.classify_text
        )
        button_style(classify_text_btn, "#4f46e5", "#6366f1")
        classify_text_btn.pack(side=tk.LEFT, padx=5)

        clear_text_btn = tk.Button(
            text_buttons_frame,
            text="Clear",
            command=lambda: self.text_input.delete(1.0, tk.END)
        )
        button_style(clear_text_btn, "#ef4444", "#dc2626")
        clear_text_btn.pack(side=tk.LEFT, padx=5)

        debug_btn = tk.Button(
            text_buttons_frame,
            text="Debug Info",
            command=self.show_debug_info
        )
        button_style(debug_btn, "#10b981", "#059669")
        debug_btn.pack(side=tk.LEFT, padx=5)

        file_tab = tk.Frame(tab_control, bg="#f5f5ff")
        tab_control.add(file_tab, text="File Input")

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
            text="Browse",
            command=self.browse_file
        )
        button_style(browse_btn, "#3b82f6", "#2563eb")
        browse_btn.pack(side=tk.RIGHT)

        classify_file_btn = tk.Button(
            file_tab,
            text="Classify",
            command=self.classify_file
        )
        button_style(classify_file_btn, "#4f46e5", "#6366f1")
        classify_file_btn.pack(pady=5)

        tab_control.pack(fill=tk.BOTH, expand=True)

        result_frame = tk.LabelFrame(
            main_frame,
            text="Classification Result",
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
            text="No classification result yet",
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
        file_path = filedialog.askopenfilename(
            title="Select email file",
            filetypes=[("Text files", "*.txt"), ("Email files", "*.eml"), ("All files", "*.*")]
        )
        if file_path:
            self.file_path_var.set(file_path)

    def classify_text(self):
        email_text = self.text_input.get(1.0, tk.END).strip()
        if not email_text:
            messagebox.showwarning("Warning", "Please enter email content.")
            return
        self.classify_and_display(email_text)

    def classify_file(self):
        file_path = self.file_path_var.get()
        if not file_path or not os.path.exists(file_path):
            messagebox.showwarning("Warning", "Please select a valid email file.")
            return
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                email_text = f.read()
            self.classify_and_display(email_text)
        except Exception as e:
            messagebox.showerror("Error", f"Cannot read file: {str(e)}")

    def classify_and_display(self, email_text):
        try:
            print(f"Starting email classification, length: {len(email_text)} characters")

            self.progress['value'] = 20
            self.root.update_idletasks()

            start_time = time.time()

            sample_df = pd.DataFrame({
                'body': [email_text],
                'label': ['unknown']
            })
            print("DataFrame created")

            self.progress['value'] = 40
            self.root.update_idletasks()

            processed_df = self.preprocessor.process_emails(sample_df)
            processed_text = processed_df['processed_text'].iloc[0]
            print(f"Preprocessed, length after processing: {len(processed_text)} characters")

            self.progress['value'] = 60
            self.root.update_idletasks()

            if 'vectorizer' not in self.data:
                raise ValueError("Vectorizer not found in model data")

            sample_X = self.data['vectorizer'].transform([processed_text])
            print(f"Vectorized, shape: {sample_X.shape}")

            if hasattr(sample_X, 'toarray'):
                sample_X = sample_X.toarray()
                print("Converted to dense array")

            self.progress['value'] = 80
            self.root.update_idletasks()

            prediction = self.model.predict(sample_X)
            print(f"Prediction shape: {prediction.shape if hasattr(prediction, 'shape') else 'scalar'}")
            print(f"Prediction: {prediction}")

            if hasattr(prediction, 'shape') and len(prediction.shape) > 0:
                prediction_value = prediction[0]
            else:
                prediction_value = prediction

            print(f"Prediction value: {prediction_value}")

            probability = None
            if hasattr(self.model, 'predict_proba'):
                try:
                    prob_result = self.model.predict_proba(sample_X)
                    print(f"Probability shape: {prob_result.shape}")
                    print(f"Probability result: {prob_result}")

                    if len(prob_result.shape) == 2 and prob_result.shape[1] >= 2:
                        probability = prob_result[0, 1]
                    elif len(prob_result.shape) == 1 and len(prob_result) >= 2:
                        probability = prob_result[1]
                    else:
                        print(f"Unexpected probability shape: {prob_result.shape}")
                        probability = None

                except Exception as e:
                    print(f"Error calculating probability: {e}")
                    probability = None

            process_time = (time.time() - start_time) * 1000

            self.progress['value'] = 100
            self.root.update_idletasks()

            spam_status = "SPAM" if prediction_value == 1 else "HAM (not spam)"
            confidence = f"{probability * 100:.2f}%" if probability is not None else "N/A"

            result_text = f"Classification: {spam_status}\nConfidence: {confidence}\nProcessing time: {process_time:.2f} ms"
            self.result_label.config(
                text=result_text,
                fg="#dc2626" if prediction_value == 1 else "#15803d",
                font=("Segoe UI", 12, "bold")
            )

            self.show_classification_details(processed_text, prediction_value)

        except Exception as e:
            error_msg = f"Error classifying email: {str(e)}"
            print(f"Error details: {traceback.format_exc()}")
            messagebox.showerror("Error", error_msg)
            self.progress['value'] = 0

    def show_classification_details(self, processed_text, prediction_value):
        try:
            self.details_text.config(state=tk.NORMAL)
            self.details_text.delete(1.0, tk.END)
            self.details_text.insert(tk.END, f"Preprocessed text:\n{processed_text}\n\n")

            self.analyze_custom_features(processed_text, prediction_value)

            self.details_text.config(state=tk.DISABLED)
            self.details_text.tag_configure("spam_word", foreground="#dc2626")
            self.details_text.tag_configure("ham_word", foreground="#15803d")

        except Exception as e:
            print(f"Error showing details: {e}")

    def analyze_custom_features(self, processed_text, prediction_value):
        try:
            if not hasattr(self.model, 'feature_log_prob_'):
                self.details_text.insert(tk.END, "Model does not support feature importance analysis.\n")
                return

            self.details_text.insert(tk.END, "Important word analysis (Custom Naive Bayes):\n")

            log_odds_ratio = self.model.feature_log_prob_[1] - self.model.feature_log_prob_[0]

            words = processed_text.split()
            feature_names = self.data.get('feature_names', [])

            if not feature_names:
                self.details_text.insert(tk.END, "No feature names information available.\n")
                return

            word_to_idx = {word: idx for idx, word in enumerate(feature_names)}

            word_scores = []
            for word in words:
                if word in word_to_idx:
                    idx = word_to_idx[word]
                    score = log_odds_ratio[idx]
                    word_scores.append((word, score))

            word_scores.sort(key=lambda x: abs(x[1]), reverse=True)

            for word, score in word_scores[:10]:
                if score > 0:
                    indicator = "spam"
                    tag = "spam_word"
                else:
                    indicator = "ham"
                    tag = "ham_word"
                line = f"- '{word}': Indicates {indicator} (log odds: {score:.4f})\n"
                self.details_text.insert(tk.END, line, (tag,))

        except Exception as e:
            print(f"Error analyzing custom features: {e}")
            self.details_text.insert(tk.END, f"Analysis error: {str(e)}\n")


def main():
    root = tk.Tk()
    app = SpamClassifierApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()