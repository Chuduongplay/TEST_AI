# ==============================================================================
# 1. IMPORT CÁC THƯ VIỆN BẮT BUỘC
# ==============================================================================
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import pickle
import os
import base64
import random

# ==============================================================================
# 2. CẤU HÌNH TRANG VÀ GIAO DIỆN CHUNG (CSS)
# ==============================================================================
st.set_page_config(layout="wide", page_title="AI Hybrid Cross-selling Tiki", page_icon="📘")

# Bạn dán link ảnh nền muốn thay vào đây
BACKGROUND_URL = "https://cdn.corenexis.com/files/c/7785826720.png"

st.markdown("""
    <style>
    /* Lớp nền chính của trang web */
    .stApp {
        background-image: url('""" + BACKGROUND_URL + """');
        background-size: cover; 
        background-position: center;
        background-attachment: fixed;
        min-height: 100vh;
        position: relative;
        overflow: hidden;
    }
    
    /* Ẩn hình túi xách và hộp quà cũ của Tiki để không đè lên ảnh nền mới */
    .stApp::before {
        display: none !important;
    }
    
    .stApp > * { position: relative; z-index: 1; }
    
    /* Khung trắng chứa bảng điều khiển admin cột trái */
    div[data-testid="column"]:first-child > div[data-testid="stVerticalBlock"] {
        background: rgba(255,255,255,0.97);
        border-radius: 20px;
        padding: 24px 20px 20px 20px;
        box-shadow: 0 8px 40px rgba(0,60,150,0.22);
        border: 1.5px solid rgba(255,255,255,0.7);
    }
    
    /* Các class CSS giữ nguyên cấu trúc hiển thị sách gốc của bạn */
    .book-card {
        background: white; border-radius: 14px; padding: 16px;
        border: 1px solid #e1e4e8; box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        display: flex; gap: 16px; align-items: flex-start; margin-top: 8px;
    }
    .book-cover { width:90px; height:120px; object-fit:cover; border-radius:8px; box-shadow:0 4px 12px rgba(0,0,0,0.15); flex-shrink:0; }
    .book-info { flex:1; }
    .book-title { font-size:17px; font-weight:700; color:#1A94FF; margin-bottom:6px; }
    .book-meta { font-size:13px; color:#555; margin-bottom:4px; }
    .book-price { font-size:18px; font-weight:800; color:#e8453c; margin-top:8px; }
    </style>
""", unsafe_allow_html=True)

# ==============================================================================
# 3. HÀM CHUYỂN ĐỔI ẢNH NỀN GIAO DIỆN SANG BASE64
# ==============================================================================
bg_css = "url('https://cdn.corenexis.com/files/c/2513965720.png')"

# ==============================================================================
# 4. TẢI DỮ LIỆU & MODEL AI
# ==============================================================================
@st.cache_resource
def load_ai_models():
    """
    Load 4 artifacts:
    - knn_model.pkl   : BUNDLE dict {knn_model, ohe_cat, ohe_mfr, X_books, W_CATEGORY, W_MANUFACTURER}
                       (KNN fit trên One-Hot Category + Manufacturer, có weighted)
    - rf_model.pkl    : RandomForestClassifier, fit trên [TheLoai_Encoded]
    - le_the_loai.pkl : LabelEncoder cho 6 nhóm thể loại sách (từ khảo sát)
    - le_phu_kien.pkl : LabelEncoder cho 5 loại phụ kiện
    """
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(current_dir, "knn_model.pkl"), "rb") as f:
            knn_model = pickle.load(f)   # ← thực ra là bundle dict, không phải KNN object trực tiếp
        with open(os.path.join(current_dir, "rf_model.pkl"), "rb") as f:
            rf_model = pickle.load(f)
        with open(os.path.join(current_dir, "le_the_loai.pkl"), "rb") as f:
            le_the_loai = pickle.load(f)
        with open(os.path.join(current_dir, "le_phu_kien.pkl"), "rb") as f:
            le_phu_kien = pickle.load(f)
        return knn_model, rf_model, le_the_loai, le_phu_kien
    except Exception as e:
        st.error(f"Lỗi load model/encoder: {e}")
        return None, None, None, None

@st.cache_data
def load_datasets():
    try:
        book_df = pd.read_csv("book_data_clean.csv")
        acc_df = pd.read_csv("phu_kien_clean.csv")
        return book_df, acc_df
    except Exception as e:
        st.error(f"Lỗi load dataset: {e}")
        book_mock = pd.DataFrame({
            "product_id": ["BK1", "BK2"], "title": ["Sách 1", "Sách 2"],
            "authors": ["A", "B"], "category": ["IT", "Self-help"],
            "mapped_category": ["Khác", "Nhóm Phát triển bản thân (Self-help) - Tâm lý"],
            "current_price": [120000, 79000], "cover_link": ["", ""],
            "Price_Scaled": [0.5, 0.3], "Rating_Scaled": [0.8, 0.7], "Category_Scaled": [0.2, 0.5]
        })
        acc_mock = pd.DataFrame({
            "accessory_ID": ["PK1", "PK2"], "name": ["Phụ kiện 1", "Phụ kiện 2"],
            "category": ["Bookmark", "Sticker"], "current_price": [20000, 55000], "cover_link": ["", ""]
        })
        return book_mock, acc_mock

knn_model, rf_model, le_the_loai, le_phu_kien = load_ai_models()
book_df, accessories_df = load_datasets()

# ==============================================================================
# 5. CẤU HÌNH RULE-BASED
# ==============================================================================
INCENTIVE_TIERS = [
    {'threshold': 70_000,   'label': 'Freeship', 'benefit': 'MIỄN PHÍ VẬN CHUYỂN'},
    {'threshold': 150_000,  'label': 'Giảm 5%',  'benefit': 'GIẢM THÊM 5%'},
    {'threshold': 250_000,  'label': 'Giảm 10%', 'benefit': 'GIẢM THÊM 10%'},
    {'threshold': 500_000,  'label': 'Giảm 12%', 'benefit': 'GIẢM THÊM 12%'},
]

# Quy tắc ưu tiên hiển thị ĐẦU TIÊN dựa trên mapped_category của sách user vừa add
DISPLAY_PRIORITY = {
    'Nhóm Phát triển bản thân (Self-help) - Tâm lý':            'accessory',
    'Nhóm Văn học - Tiểu thuyết - Truyện tranh':                'book',
    'Khác':                                                      'accessory',
    'Nhóm Kiến thức - Khoa học - Lịch sử':                       'book',
    'Nhóm Sách Thiếu nhi - Gia đình':                            'accessory',
    'Nhóm Giáo dục - Ngoại ngữ':                                 'book',
    'Nhóm Đời sống - Sở thích (chiêm tinh, nấu ăn, làm đẹp,...)':'accessory',
}

PRICE_RANGE_RATIO = 0.30  # Khoảng giá [gap, gap * 1.30]

# Map từ output của le_phu_kien -> giá trị cột `category` trong dataset_phu_kien
def map_phu_kien_to_acc_category(phu_kien_label: str) -> str:
    """'Bookmark (Đánh dấu trang)' -> 'Bookmark', 'Bọc sách (Bìa kiếng/Bọc da)' -> 'Bọc sách'..."""
    return phu_kien_label.split('(')[0].strip()

# ==============================================================================
# 6. CÁC HÀM CORE CỦA THUẬT TOÁN
# ==============================================================================
def get_fallback_mapped_category(book_df, le_the_loai):
    """
    Khi sách user add có mapped_category = 'Khác' (không nằm trong 6 nhóm khảo sát)
    -> Lấy thể loại phổ biến nhất trong book_df (loại 'Khác') VÀ phải có trong classes của le_the_loai.
    """
    valid_cats = set(le_the_loai.classes_)
    counts = book_df[book_df['mapped_category'] != 'Khác']['mapped_category'].value_counts()
    for cat, _ in counts.items():
        if cat in valid_cats:
            return cat
    return le_the_loai.classes_[0]

def module1_knn_books(last_book_row, book_df, knn_model, k_pick=2):
    """
    MODULE 1 - KNN: Lấy k_pick (=2) cuốn sách phù hợp nhất với cuốn user vừa add.
    - Bước 1: KNN tìm K=10 ứng viên gần nhất theo One-Hot (Category + Manufacturer)
    - Bước 2: Loại chính cuốn input, sort theo |giá - giá_hiện_tại| tăng dần
    - Bước 3: Lấy k_pick cuốn đầu (giá gần nhất)
    """
    pid_input = str(last_book_row['product_id'])

    matched = book_df[book_df['product_id'].astype(str) == pid_input]
    if matched.empty:
        return []
    label = matched.index[0]

    # knn_model giờ là BUNDLE dict — extract object và ma trận features
    knn_obj = knn_model['knn_model']
    X_books = knn_model['X_books']
    current_price = last_book_row.get('current_price', 0)

    # Lấy vector cuốn đang xem (giữ shape 2D cho kneighbors)
    feature = X_books[label:label+1]
    distances, indices = knn_obj.kneighbors(feature)

    # Thu thập ứng viên + tính độ lệch giá
    ung_vien = []
    for i in indices[0]:
        sid = str(book_df.iloc[i]['product_id'])
        if sid == pid_input:
            continue
        price = book_df.iloc[i].get('current_price', 0)
        price_diff = abs(price - current_price)
        ung_vien.append((price_diff, book_df.iloc[i].to_dict()))

    # Sort theo độ lệch giá tăng dần
    ung_vien.sort(key=lambda x: x[0])

    # Trả về top k_pick (giá gần nhất ưu tiên)
    return [item[1] for item in ung_vien[:k_pick]]

def module2_rf_top3_categories(last_book_row, book_df, rf_model, le_the_loai, le_phu_kien, top_k=3):
    """
    MODULE 2 - RF: Lấy top_k (=3) thể loại phụ kiện có xác suất cao nhất
    dựa trên mapped_category của cuốn sách user vừa add.
    Trả về list tên thể loại đã map về `category` của dataset_phu_kien.
    """
    mapped_cat = last_book_row.get('mapped_category', 'Khác')
    valid_cats = set(le_the_loai.classes_)

    # Nếu là 'Khác' hoặc không có trong classes -> fallback
    if mapped_cat not in valid_cats:
        mapped_cat = get_fallback_mapped_category(book_df, le_the_loai)

    encoded = le_the_loai.transform([mapped_cat])[0]

    # RF được train với 1 feature ['TheLoai_Encoded'] -> giữ DataFrame để khớp feature_names
    X_input = pd.DataFrame([[encoded]], columns=['TheLoai_Encoded'])
    proba = rf_model.predict_proba(X_input)[0]

    # Sắp xếp giảm dần theo proba
    top_idx = np.argsort(proba)[::-1][:top_k]
    top_labels = le_phu_kien.inverse_transform(top_idx)
    top_acc_categories = [map_phu_kien_to_acc_category(lbl) for lbl in top_labels]
    return top_acc_categories

def compute_gap(cart_total):
    """Số tiền còn thiếu để đạt tier ưu đãi kế tiếp. 0 nếu đã max tier."""
    next_tier = next((t for t in INCENTIVE_TIERS if t['threshold'] > cart_total), None)
    if next_tier is None:
        return 0, None
    return next_tier['threshold'] - cart_total, next_tier

def pick_one_acc_per_category(acc_df, categories, gap, ratio=PRICE_RANGE_RATIO):
    """
    Cho mỗi thể loại trong `categories`, chọn 1 sản phẩm:
    - Nếu gap > 0: filter giá nằm trong [gap, gap*(1+ratio)] -> rẻ nhất trong khoảng.
      Không có sản phẩm trong khoảng -> rẻ nhất của thể loại đó.
    - Nếu gap == 0 (max tier): rẻ nhất của thể loại đó.
    Trả về list candidates (1 per category, có thể < len(categories) nếu thể loại không tồn tại).
    """
    candidates = []
    for cat in categories:
        pool = acc_df[acc_df['category'] == cat]
        if pool.empty:
            continue

        if gap > 0:
            lo, hi = gap, gap * (1 + ratio)
            in_range = pool[(pool['current_price'] >= lo) & (pool['current_price'] <= hi)]
            if not in_range.empty:
                pick = in_range.sort_values('current_price', ascending=True).iloc[0]
            else:
                pick = pool.sort_values('current_price', ascending=True).iloc[0]
        else:
            pick = pool.sort_values('current_price', ascending=True).iloc[0]

        candidates.append(pick.to_dict())
    return candidates

def compute_recommendations(last_book_row, cart_total, book_df, acc_df,
                            knn_model, rf_model, le_the_loai, le_phu_kien):
    """
    Pipeline đầy đủ Module 1 + Module 2 -> trả về dict gồm:
    - books: list 2 cuốn KNN (giữ thứ tự distance ascending)
    - accessory: 1 phụ kiện (random từ 3 candidates)
    - first_priority: 'book' hoặc 'accessory' (dựa rule mapped_category)
    - meta: gap, top3_cats, ... để hiển thị/debug
    """
    # Module 1
    books = module1_knn_books(last_book_row, book_df, knn_model, k_pick=2)

    # Module 2
    top3_cats = module2_rf_top3_categories(last_book_row, book_df, rf_model, le_the_loai, le_phu_kien, top_k=3)

    # Gap
    gap, next_tier = compute_gap(cart_total)

    # 3 candidates phụ kiện -> random 1
    candidates = pick_one_acc_per_category(acc_df, top3_cats, gap, ratio=PRICE_RANGE_RATIO)
    if not candidates:
        # Edge case: không thể loại nào có sản phẩm -> lấy rẻ nhất toàn pool
        if not acc_df.empty:
            chosen_acc = acc_df.sort_values('current_price', ascending=True).iloc[0].to_dict()
        else:
            chosen_acc = None
    else:
        chosen_acc = random.choice(candidates)  # random thực sự

    # Quy tắc thứ tự hiển thị đầu tiên
    last_mapped = last_book_row.get('mapped_category', 'Khác')
    first_priority = DISPLAY_PRIORITY.get(last_mapped, 'book')

    return {
        'books': books,
        'accessory': chosen_acc,
        'first_priority': first_priority,
        'gap': gap,
        'benefit': 'MAX ƯU ĐÃI' if next_tier is None else next_tier['benefit'],
        'top3_cats': top3_cats,
    }

def build_display_sequence(reco):
    """
    Tạo sequence 3 items để xoay vòng khi bấm "Làm mới":
    - first_priority == 'book':       [book1, accessory, book2]
    - first_priority == 'accessory':  [accessory, book1, book2]
    Mỗi item có form {'kind': 'book'|'accessory', 'data': {...}}
    """
    books = reco.get('books') or []
    acc = reco.get('accessory')

    book_items = [{'kind': 'book', 'data': b} for b in books]
    acc_items = [{'kind': 'accessory', 'data': acc}] if acc else []

    if reco['first_priority'] == 'book':
        if len(book_items) >= 2 and acc_items:
            return [book_items[0], acc_items[0], book_items[1]]
        # Edge: thiếu sách/phụ kiện -> ghép tuần tự
        return book_items + acc_items
    else:  # accessory first
        if len(book_items) >= 2 and acc_items:
            return [acc_items[0], book_items[0], book_items[1]]
        return acc_items + book_items

# ==============================================================================
# 7. QUẢN LÝ TRẠNG THÁI (SESSION STATE)
# ==============================================================================
for k, v in [
    ('cart', []),
    ('suggest_index', 0),
    ('payment_method', 'cash'),
    ('recommendations', None),   # cache 3 items (2 books + 1 acc) - chỉ tính lại khi add sách mới
    ('last_added_pid', None),    # id sách được add gần nhất, dùng để biết khi nào cần recompute
]:
    if k not in st.session_state:
        st.session_state[k] = v

def recompute_recommendations():
    """
    Tính lại Module 1 + Module 2 dựa trên cuốn sách vừa được add MỚI NHẤT vào cart.
    Lưu kết quả vào session_state.recommendations. Reset suggest_index.
    Gọi NGAY sau khi append vào cart hoặc khi clear cart.
    """
    if not st.session_state.cart:
        st.session_state.recommendations = None
        st.session_state.last_added_pid = None
        st.session_state.suggest_index = 0
        return

    last_book = st.session_state.cart[-1]
    cart_total = sum(item.get('current_price', 0) for item in st.session_state.cart)

    if knn_model is None or rf_model is None or le_the_loai is None or le_phu_kien is None:
        st.session_state.recommendations = None
        return

    reco = compute_recommendations(
        last_book_row=last_book,
        cart_total=cart_total,
        book_df=book_df,
        acc_df=accessories_df,
        knn_model=knn_model,
        rf_model=rf_model,
        le_the_loai=le_the_loai,
        le_phu_kien=le_phu_kien,
    )
    st.session_state.recommendations = reco
    st.session_state.last_added_pid = str(last_book.get('product_id', ''))
    st.session_state.suggest_index = 0

# ==============================================================================
# 8. GIAO DIỆN CHÍNH
# ==============================================================================
col_admin, col_app = st.columns([1.3, 1], gap="large")

# ───────── CỘT TRÁI: BẢNG ĐIỀU KHIỂN ADMIN ─────────
with col_admin:
    st.markdown("""
        <div style='display:flex;align-items:center;gap:12px;margin-bottom:4px;'>
            <div style='background:linear-gradient(135deg,#1A94FF,#0055cc);border-radius:12px;
                        padding:8px 14px;color:white;font-weight:800;font-size:18px;letter-spacing:1px;'>
                 ADMIN DASHBOARD</div>
            <span style='color:#888;font-size:13px;'>(Backend)</span>
        </div><hr>""", unsafe_allow_html=True)

    st.markdown("#### 1. Chọn sách từ Database")
    display_options = {row['product_id']: f"{row['product_id']} - {row['title']}" for _, row in book_df.iterrows()}
    selected_id = st.selectbox(
        "Chọn ID sách:",
        options=list(display_options.keys()),
        format_func=lambda x: display_options[x],
        label_visibility="collapsed"
    )
    selected_book_row = book_df[book_df['product_id'] == selected_id].iloc[0]

    st.markdown("#### 2. Thông tin sách (Preview)")
    st.markdown(f"""
        <div class="book-card">
            <img class="book-cover" src="{selected_book_row.get('cover_link', '')}" onerror="this.src='https://via.placeholder.com/90x120?text=No+Cover'"/>
            <div class="book-info">
                <div class="book-title">{selected_book_row.get('title', 'N/A')}</div>
                <div class="book-meta">Tác giả: <span>{selected_book_row.get('authors', 'N/A')}</span></div>
                <div class="book-meta">Thể loại: <span>{selected_book_row.get('category', 'N/A')}</span></div>
                <div class="book-meta">Nhóm: <span>{selected_book_row.get('mapped_category', 'N/A')}</span></div>
                <div class="book-price">{int(selected_book_row.get('current_price', 0)):,} đ</div>
            </div>
        </div><br>""", unsafe_allow_html=True)

    if st.button("➕ Thêm cuốn này vào App", use_container_width=True, type="primary"):
        st.session_state.cart.append(selected_book_row.to_dict())
        # Tính lại Module 1 + Module 2 cho cuốn vừa add (RESET kết quả cũ)
        recompute_recommendations()
        st.rerun()

    st.markdown("#### 3. Quản lý phiên demo")
    if st.button("🔄 Làm mới / Xóa Giỏ hàng", use_container_width=True):
        st.session_state.cart = []
        st.session_state.suggest_index = 0
        st.session_state.payment_method = "cash"
        st.session_state.recommendations = None
        st.session_state.last_added_pid = None
        st.rerun()

if st.session_state.recommendations:
        with st.expander("🔍 Debug - Kết quả AI hiện tại", expanded=False):
            reco = st.session_state.recommendations
            st.write(f"**Ưu tiên hiển thị đầu:** `{reco['first_priority']}`")
            st.write(f"**Top-3 thể loại phụ kiện (RF):** {reco['top3_cats']}")
            st.write(f"**Gap đến tier kế tiếp:** {reco['gap']:,}đ")
            st.write(f"**Số sách KNN gợi ý:** {len(reco['books'])}")
            
            # --- KNN ---
            if reco['books']:
                st.write("**📚 Chi tiết sách KNN gợi ý:**")
                for i, b in enumerate(reco['books']):
                    st.markdown(f"""
                    **Sách {i+1}:** {b.get('product_id')} - {b.get('title')}
                    - Tác giả: {b.get('authors', 'N/A')}
                    - Thể loại: {b.get('category', 'N/A')}
                    - Nhóm: {b.get('mapped_category', 'N/A')}
                    """)
            st.markdown("---")
            # ----------------------------------------------------

            if reco['accessory']:
                st.write(f"**Phụ kiện đã random:** {reco['accessory'].get('name','')} - {int(reco['accessory'].get('current_price',0)):,}đ")
            st.write(f"**Suggest index:** {st.session_state.suggest_index}")

# ───────── CỘT PHẢI: GIẢ LẬP TIKI APP ─────────
with col_app:
    cart_total = sum(item.get('current_price', 0) for item in st.session_state.cart)

    # Lấy item đang hiển thị (nếu có)
    sequence = []
    current_item = None
    if st.session_state.recommendations is not None:
        sequence = build_display_sequence(st.session_state.recommendations)
        if sequence:
            idx = st.session_state.suggest_index % len(sequence)
            current_item = sequence[idx]

# --- NÚT LÀM MỚI GỢI Ý ---
    # Kiểm tra xem có gợi ý hay không
    co_goi_y = (current_item is not None and len(sequence) > 0)

    def next_suggestion_callback():
        if co_goi_y:
            st.session_state.suggest_index = (st.session_state.suggest_index + 1) % len(sequence)

    # Nếu chưa có sách thì nó sẽ bị mờ đi.
    st.button(
        "🔄 Làm mới gợi ý", 
        use_container_width=True, 
        on_click=next_suggestion_callback,
        disabled=not co_goi_y 
    )

    # Tính giá ưu đãi DỰA TRÊN cart + item đang hiển thị
    extended_total = cart_total + (
        current_item['data'].get('current_price', 0) if current_item else 0
    )

    shipping_fee = 30000
    discount_amount = 0
    total_saved = 0
    freeship_msg = ""
    discount_msg = ""

    if extended_total > 0:
        if extended_total >= 70000:
            shipping_fee = 0
            total_saved += 30000
            freeship_msg = "✅ Đã Freeship"
        else:
            freeship_msg = "❌ Phí ship: 30k"

        if extended_total >= 500000:
            discount_amount = int(extended_total * 0.12); discount_msg = "Giảm 12%"
        elif extended_total >= 250000:
            discount_amount = int(extended_total * 0.10); discount_msg = "Giảm 10%"
        elif extended_total >= 150000:
            discount_amount = int(extended_total * 0.05); discount_msg = "Giảm 5%"

        total_saved += discount_amount

    final_total = extended_total + shipping_fee - discount_amount

    # Thông tin tier kế tiếp (sau khi đã cộng item đang hiển thị)
    gap_after, next_tier_after = compute_gap(extended_total)
    benefit_msg_after = "MAX ƯU ĐÃI" if next_tier_after is None else next_tier_after['benefit']

    # --- KHU VỰC 1: SÁCH KHÁCH ĐÃ THÊM ---
    cart_html = ""
    if not st.session_state.cart:
        cart_html = "<div style='color:#888; text-align:center; margin-top:20px; font-size:12px;'>Chưa có sách nào</div>"
    else:
        for item in st.session_state.cart:
            cart_html += f"""
            <div style='display:flex; align-items:center; gap:10px; margin-bottom:8px;'>
                <img src='{item.get('cover_link','')}' style='width:30px; height:40px; object-fit:cover; border-radius:4px;' onerror="this.src='https://via.placeholder.com/30x40?text=S'"/>
                <div style='flex:1; line-height:1.2;'>
                    <div style='font-size:12px; font-weight:600; color:#222; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; width:230px;'>{item.get('title', '')}</div>
                    <div style='font-size:12px; font-weight:700; color:#e8453c;'>{int(item.get('current_price', 0)):,} đ</div>
                </div>
            </div>"""

    # --- KHU VỰC 2: GỢI Ý MUA KÈM (xoay vòng giữa 3 items đã cache) ---
    suggest_html = ""
    if current_item is not None and len(sequence) > 0:

        kind = current_item['kind']
        data = current_item['data']
        idx_display = st.session_state.suggest_index % len(sequence)

        if kind == 'book':
            label_tag = "📚 GỢI Ý"
            name_field = data.get('title', '')
        else:
            label_tag = f"🎁 PHỤ KIỆN ({data.get('category','Khác')})"
            name_field = data.get('name', '')

        suggest_html = f"""
        <div style='display:flex; align-items:center; justify-content:space-between; margin-bottom:5px;'>
            <span style='font-size:10px; font-weight:bold; color:#1A94FF;'>{label_tag} ({idx_display+1}/{len(sequence)})</span>
            <span style='font-size:10px; color:#1A94FF; font-weight:bold;'>🔄 Làm mới</span>
        </div>
        <div style='display:flex; align-items:center; gap:10px; background:#f4f6f8; padding:8px; border-radius:8px;'>
            <img src="{data.get('cover_link', '')}" onerror="this.src='https://via.placeholder.com/30?text=PK'" style="width:30px; height:30px; border-radius:4px;" />
            <div style='flex:1;'>
                <div style='font-size:11px; font-weight:600; color:#222; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; width:190px;'>{name_field}</div>
                <div style='font-size:12px; font-weight:800; color:#e8453c;'>{int(data.get('current_price', 0)):,} đ</div>
            </div>
            <div style='background:#1A94FF; color:white; width:24px; height:24px; border-radius:50%; display:flex; align-items:center; justify-content:center; font-weight:bold; cursor:pointer;'>+</div>
        </div>
        <div style='font-size:9px; color:#d46b08; margin-top:4px; text-align:right;'>
            {('Mua thêm <b>' + f'{gap_after:,}' + 'đ</b> để ' + benefit_msg_after) if gap_after > 0 else '🎉 ' + benefit_msg_after}
        </div>
        """

    # --- ĐỔ TOÀN BỘ VÀO IFRAME ---
    phone_html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
    <style>
        * {{ margin:0; padding:0; box-sizing:border-box; font-family:-apple-system,BlinkMacSystemFont,sans-serif; }}
        body {{ display:flex; justify-content:center; padding:10px; background:transparent; }}
        
        .phone-frame {{
            position: relative;
            width: 350px;
            height: 700px;
            background-image: {bg_css};
            background-size: 100% 100%;
            background-repeat: no-repeat;
            border-radius: 40px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            overflow: hidden;
        }}

        .zone-cart {{
            position: absolute;
            top: 20%;
            left: 8%;
            width: 86%;
            height: 12%;
            overflow-y: auto;
            background: rgba(255, 255, 255, 0.95);
            padding: 5px;
            border-radius: 8px;
        }}
        .zone-cart::-webkit-scrollbar {{ display: none; }}

        .zone-suggest {{
            position: absolute;
            top: 42%;
            left: 8%;
            width: 85%;
            background: rgba(255, 255, 255, 0.98);
            padding: 8px;
            border-radius: 8px;
            z-index: 10;
        }}

        .zone-freeship {{
            position: absolute;
            bottom: 11%;
            right: 8%;
            width: 40%;
            text-align: right;
            padding: 2px;
            background: rgba(255,255,255,0.8);
        }}

        .zone-total {{
            position: absolute;
            bottom: 4%;
            left: 8%;
            width: 45%;
            background: white;
            padding: 5px;
            border-radius: 5px;
        }}
    </style>
    </head><body>
    
    <div class="phone-frame">
        
        <div class="zone-cart">
            {cart_html}
        </div>

        <div class="zone-suggest">
            {suggest_html}
        </div>

        <div class="zone-freeship">
            <div style="font-size:11px; font-weight:700; color:#389e0d;">{freeship_msg}</div>
            <div style="font-size:11px; font-weight:700; color:#e8453c;">
                {f"- {discount_msg}" if discount_amount > 0 else ""}
            </div>
        </div>

        <div class="zone-total">
            <div style="font-size:10px; color:#555; font-weight:600;">Tổng thanh toán:</div>
            <div style="font-size:16px; font-weight:900; color:#e8453c; margin-top:2px;">{final_total:,}đ</div>
            <div style="font-size:10px; font-weight:700; color:#389e0d; margin-top:2px;">
                Tiết kiệm: {total_saved:,}đ
            </div>
        </div>
    </div>
    </body></html>"""

    components.html(phone_html, height=750, scrolling=False)
