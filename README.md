# 5c-Model-test
# Ứng Dụng Dự Báo Rủi Ro Khách Hàng (Hồi Quy Logistic)

Đây là ứng dụng web tương tác xây dựng bằng Streamlit, nhằm tự động hóa quy trình dự báo trạng thái phân loại rủi ro (PD) dựa trên mô hình Hồi quy Logistic (Logistic Regression). Mô hình được huấn luyện trên 24 biến phân tích (TC1-TC5, NL1-NL4, DK1-DK5, V1-V6, TS1-TS4).

## 1. Yêu cầu hệ thống
Đảm bảo bạn đã cài đặt Python 3.9 trở lên. Các thư viện bắt buộc được mô tả trong tệp `requirements.txt`. Khuyến nghị sử dụng phiên bản Streamlit mới nhất.

## 2. Cài đặt và Khởi chạy

Bước 1: Cài đặt các thư viện cần thiết:
```bash
pip install -r requirements.txt
