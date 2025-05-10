"""
Bước 1: Tải và xử lý dữ liệu email.
"""

import os
import pickle
import sys

# Import từ project
from src.data.data_loader import EmailDataLoader


def download_and_process_data():
    """Tải và xử lý dữ liệu email."""
    print("\n" + "=" * 50)
    print("BƯỚC 1: TẢI VÀ XỬ LÝ DỮ LIỆU")
    print("=" * 50 + "\n")

    # Thiết lập thông số
    dataset_name = 'enron'  # 'spamassassin' hoặc 'enron'
    limit_emails = None  # Số lượng email để xử lý (None để xử lý tất cả)

    # Khởi tạo data loader
    data_loader = EmailDataLoader(dataset_name)

    # Kiểm tra xem dữ liệu đã được xử lý chưa
    # csv_path = os.path.join(data_loader.processed_dir, 'email_data.csv')
    # if os.path.exists(csv_path):
    #     print(f"Phát hiện dữ liệu đã xử lý tại {csv_path}")
    #     choice = input("Bạn muốn sử dụng dữ liệu này (y) hay tải lại (n)? [y/n]: ").lower()
    #
    #     if choice == 'y':
    #         print(f"Đang tải dữ liệu đã xử lý từ {csv_path}")
    #         email_df = data_loader.load_from_csv()
    #     else:
    #         print("Đang tải và xử lý bộ dữ liệu mới...")
    #         email_df = data_loader.process_dataset(limit=limit_emails)
    # else:
    #     print("Không tìm thấy dữ liệu đã xử lý.")
    #     print("Đang tải và xử lý bộ dữ liệu mới...")
    #     email_df = data_loader.process_dataset(limit=limit_emails)

    print("Bỏ qua dữ liệu đã xử lý trước đó và tải lại toàn bộ dữ liệu...")
    email_df = data_loader.process_dataset(limit=None)

    if email_df is None or len(email_df) == 0:
        print("Không thể tải dữ liệu. Vui lòng kiểm tra lại cài đặt.")
        sys.exit(1)

    print(
        f"Đã tải {len(email_df)} email ({email_df['label'].value_counts()['spam']} spam, {email_df['label'].value_counts()['ham']} ham)")

    # Lưu DataFrame để các bước tiếp theo sử dụng
    output_file = 'email_data.pkl'
    with open(output_file, 'wb') as f:
        pickle.dump(email_df, f)

    print(f"Đã lưu dữ liệu vào file {output_file} để các bước tiếp theo sử dụng")

    return email_df


if __name__ == "__main__":
    download_and_process_data()