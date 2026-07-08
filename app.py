import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
import io

# ==========================================
# KHỐI MASTER - CẤU HÌNH TRANG VÀ HÀM CHUNG
# ==========================================
st.set_page_config(
    page_title="Hệ thống Dự báo rủi ro - Logistic Regression",
    page_icon="🎯",
    layout="wide"
)

@st.cache_data
def load_data(file):
    try:
        # Hỗ trợ cả csv và excel
        if file.name.endswith('.csv'):
            return pd.read_csv(file)
        else:
            return pd.read_excel(file)
    except Exception as e:
        return None

# Các biến đầu vào từ notebook
FEATURES = ['TC1', 'TC2', 'TC3', 'TC4', 'TC5', 
            'NL1', 'NL2', 'NL3', 'NL4', 
            'DK1', 'DK2', 'DK3', 'DK4', 'DK5', 
            'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 
            'TS1', 'TS2', 'TS3', 'TS4']
TARGET = 'PD'

# ==========================================
# THÀNH PHẦN 1: SIDEBAR - VÙNG CẤU HÌNH
# ==========================================
with st.sidebar:
    st.header("⚙️ Cấu hình & Tải dữ liệu")
    
    uploaded_file = st.file_uploader("Tải lên dữ liệu (.csv, .xlsx)", type=['csv', 'xlsx'])
    
    st.divider()
    st.subheader("Tham số mô hình AI")
    
    test_size = st.slider("Tỷ lệ tập kiểm thử (test_size)", 0.1, 0.5, 0.2, 0.05, 
                          help="Tỷ lệ chia dữ liệu cho tập kiểm tra (mặc định 0.2 như trong notebook).")
    random_state = st.number_input("Random State", min_value=0, max_value=9999, value=23, step=1,
                                   help="Seed để đảm bảo kết quả có thể tái hiện.")
    
    with st.expander("Tham số Logistic Regression"):
        c_param = st.number_input("Tham số điều chuẩn C (Nghịch đảo của regularization)", value=1.0, min_value=0.01, step=0.1)
        max_iter = st.slider("Số vòng lặp tối đa (max_iter)", 100, 1000, 100, step=100)
    
    st.divider()
    train_button = st.button("🚀 Huấn luyện mô hình", type="primary", use_container_width=True)

# ==========================================
# THÀNH PHẦN 2: HEADER - VÙNG ĐỊNH HƯỚNG
# ==========================================
st.title("🎯 Ứng dụng Dự báo Trạng thái (Rủi ro Khách hàng)")
st.caption("Ứng dụng xây dựng mô hình Phân loại bằng thuật toán Hồi quy Logistic dựa trên 24 tiêu chí đánh giá.")

if uploaded_file is None:
    st.info("👋 Vui lòng tải lên tập dữ liệu (.csv hoặc .xlsx) ở thanh bên trái để bắt đầu.")
    st.stop()

df = load_data(uploaded_file)
if df is None:
    st.error("Lỗi khi đọc file. Vui lòng kiểm tra lại định dạng.")
    st.stop()

# Kiểm tra dữ liệu có đủ cột không
missing_cols = [col for col in FEATURES + [TARGET] if col not in df.columns]
if missing_cols:
    st.error(f"Dữ liệu tải lên thiếu các cột bắt buộc: {', '.join(missing_cols)}")
    st.stop()

st.caption(f"📁 Đang dùng tệp: {uploaded_file.name}")
st.divider()

# ==========================================
# KHỐI TRAIN (Xử lý khi bấm nút ở Sidebar)
# ==========================================
if train_button:
    with st.spinner("Đang huấn luyện mô hình..."):
        X = df[FEATURES]
        y = df[TARGET]
        
        # Chia tập dữ liệu (như trong notebook test_size=0.2, random_state=23)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        
        # Khởi tạo và huấn luyện
        model = LogisticRegression(C=c_param, max_iter=max_iter)
        model.fit(X_train, y_train)
        
        # Dự báo và chấm điểm
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        
        # Lưu vào session_state
        st.session_state['model'] = model
        st.session_state['accuracy'] = acc
        st.session_state['cm'] = cm
        st.session_state['is_trained'] = True
        st.toast("✅ Huấn luyện mô hình thành công!", icon="🎉")

# ==========================================
# KHỞI TẠO TABS
# ==========================================
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Tổng quan Dữ liệu", 
    "📈 Trực quan hóa Dữ liệu", 
    "⚙️ Kết quả Huấn luyện", 
    "🔮 Sử dụng Mô hình"
])

# ==========================================
# THÀNH PHẦN 3: TAB "TỔNG QUAN DỮ LIỆU"
# ==========================================
with tab1:
    col1, col2, col3 = st.columns(3)
    col1.metric("Số lượng bản ghi (Dòng)", df.shape[0])
    col2.metric("Số lượng biến (Cột)", df.shape[1])
    file_size_mb = uploaded_file.size / (1024 * 1024)
    col3.metric("Dung lượng tệp", f"{file_size_mb:.2f} MB")
    
    st.subheader("Dữ liệu thô (5 dòng đầu)")
    with st.container(height=250):
        st.dataframe(df.head(), use_container_width=True)
        
    st.subheader("Thống kê mô tả (Các biến đầu vào & mục tiêu)")
    st.dataframe(df[FEATURES + [TARGET]].describe(), use_container_width=True)

# ==========================================
# THÀNH PHẦN 4: TAB "TRỰC QUAN HÓA DỮ LIỆU"
# ==========================================
with tab2:
    st.subheader("Khám phá phân phối dữ liệu")
    selected_features = st.multiselect("Chọn các biến để xem biểu đồ (Mặc định: biến mục tiêu PD và 3 biến đầu):", 
                                       options=[TARGET] + FEATURES, 
                                       default=[TARGET, 'TC1', 'TC2', 'TC3'])
    
    if selected_features:
        # Lưới 2x2
        cols = st.columns(2)
        for idx, col_name in enumerate(selected_features):
            with cols[idx % 2]:
                if col_name == TARGET or df[col_name].nunique() < 10:
                    # Biến rời rạc / Phân loại
                    counts = df[col_name].value_counts().reset_index()
                    counts.columns = [col_name, 'Số lượng']
                    fig = px.bar(counts, x=col_name, y='Số lượng', title=f"Phân phối của {col_name}", text_auto=True)
                else:
                    # Biến liên tục
                    fig = px.histogram(df, x=col_name, title=f"Phân phối của {col_name}", marginal="box")
                st.plotly_chart(fig, use_container_width=True)

# ==========================================
# THÀNH PHẦN 5: TAB "KẾT QUẢ HUẤN LUYỆN"
# ==========================================
with tab3:
    if 'is_trained' not in st.session_state:
        st.info("👈 Vui lòng bấm nút **Huấn luyện mô hình** ở thanh công cụ bên trái để xem kết quả.")
    else:
        st.subheader("Chỉ tiêu Đánh giá (Logistic Regression)")
        
        # Độ chính xác
        acc = st.session_state['accuracy']
        st.metric("Độ chính xác (Accuracy) trên tập Test", f"{acc * 100:.2f}%")
        
        # Ma trận nhầm lẫn
        st.subheader("Ma trận nhầm lẫn (Confusion Matrix)")
        cm = st.session_state['cm']
        
        fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale='Blues',
                           labels=dict(x="Dự báo (Predicted)", y="Thực tế (Actual)", color="Số lượng"),
                           x=['Class 0', 'Class 1'], y=['Class 0', 'Class 1'],
                           title="Confusion Matrix")
        st.plotly_chart(fig_cm, use_container_width=True)

# ==========================================
# THÀNH PHẦN 6: TAB "SỬ DỤNG MÔ HÌNH"
# ==========================================
with tab4:
    if 'is_trained' not in st.session_state:
        st.info("👈 Vui lòng bấm nút **Huấn luyện mô hình** ở thanh công cụ bên trái để sử dụng chức năng dự báo.")
    else:
        pred_mode = st.radio("Chọn phương thức dự báo:", 
                             ["Nhập liệu trực tiếp (Một mẫu)", "Tải lên tệp dữ liệu (Hàng loạt)"], 
                             horizontal=True)
        
        model = st.session_state['model']
        
        if pred_mode == "Nhập liệu trực tiếp (Một mẫu)":
            with st.form("predict_form"):
                st.write("Nhập các thông số để dự báo:")
                
                # Bố trí 4 cột cho 24 biến (6 hàng)
                form_cols = st.columns(4)
                input_data = {}
                
                for idx, feature in enumerate(FEATURES):
                    min_val = int(df[feature].min())
                    max_val = int(df[feature].max())
                    median_val = int(df[feature].median())
                    
                    with form_cols[idx % 4]:
                        input_data[feature] = st.number_input(
                            f"{feature}", 
                            min_value=min_val, 
                            max_value=max_val, 
                            value=median_val,
                            step=1
                        )
                
                submitted = st.form_submit_button("🔍 Dự báo", type="primary")
                
                if submitted:
                    # Tiền xử lý thành DataFrame
                    input_df = pd.DataFrame([input_data])
                    
                    # Dự đoán
                    prediction = model.predict(input_df)[0]
                    probabilities = model.predict_proba(input_df)[0]
                    
                    st.divider()
                    st.subheader("Kết quả Dự báo")
                    if prediction == 1:
                        st.error(f"Khả năng có rủi ro (Class 1). Xác suất: {probabilities[1]*100:.2f}%")
                    else:
                        st.success(f"Khả năng an toàn (Class 0). Xác suất: {probabilities[0]*100:.2f}%")
        
        else:
            st.write("Tải lên tệp CSV/Excel chứa dữ liệu mới với các cột tương ứng (không cần cột PD).")
            predict_file = st.file_uploader("Tệp dữ liệu mới", type=['csv', 'xlsx'], key="predict_file")
            
            if predict_file:
                new_data = load_data(predict_file)
                
                # Kiểm tra cấu trúc
                missing_pred_cols = [col for col in FEATURES if col not in new_data.columns]
                if missing_pred_cols:
                    st.error(f"Tệp tải lên thiếu các cột bắt buộc cho mô hình: {', '.join(missing_pred_cols)}")
                else:
                    # Lấy đúng các cột cần thiết
                    X_new = new_data[FEATURES]
                    
                    # Chạy dự báo
                    new_data['Dự báo (PD)'] = model.predict(X_new)
                    prob_array = model.predict_proba(X_new)
                    new_data['Xác suất (Class 0)'] = prob_array[:, 0].round(4)
                    new_data['Xác suất (Class 1)'] = prob_array[:, 1].round(4)
                    
                    st.success("Đã hoàn thành dự báo hàng loạt!")
                    
                    with st.container(height=300):
                        st.dataframe(new_data, use_container_width=True)
                    
                    # Tải xuống
                    csv = new_data.to_csv(index=False).encode('utf-8-sig')
                    st.download_button(
                        label="📥 Tải kết quả xuống (CSV)",
                        data=csv,
                        file_name='ket_qua_du_bao.csv',
                        mime='text/csv',
                    )
