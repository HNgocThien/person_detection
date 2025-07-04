# Sử dụng Python 3.9 làm base image
FROM python:3.9-slim

# Thiết lập thư mục làm việc trong container
WORKDIR /app

# Sao chép file requirements.txt vào container trước để tận dụng cache
COPY requirements.txt .

# Cài đặt các thư viện cần thiết (tăng timeout nếu mạng yếu)
RUN pip install --default-timeout=200 --no-cache-dir -r requirements.txt

# Sao chép các file/thư mục cần thiết vào container
COPY app.py .
COPY models/ models/

# Expose cổng 8501 để chạy ứng dụng Streamlit
EXPOSE 8501

# Lệnh để chạy ứng dụng Streamlit
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]