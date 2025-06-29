import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 聚類算法
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering, Birch
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# 數據集
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_digits, make_blobs, make_moons, make_circles

# 評價指標
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, fowlkes_mallows_score

import warnings
warnings.filterwarnings('ignore')

# 設置頁面配置
st.set_page_config(
    page_title="無監督學習-聚類互動教學平台",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定義CSS樣式
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.8rem;
        color: #ff7f0e;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .small-text {
        font-size: 0.7rem !important;
        line-height: 1.2 !important;
    }
</style>
""", unsafe_allow_html=True)

# 側邊欄導航
st.sidebar.title("🔍 課程導航")
page = st.sidebar.radio(
    "選擇學習模塊：",
    [
        "🏠 無監督學習概述",
        "📊 數據集探索", 
        "🎯 K-Means聚類",
        "🌊 DBSCAN密度聚類",
        "🌳 層次聚類",
        "🎲 高斯混合模型",
        "🕸️ 譜聚類",
        "🌿 BIRCH聚類",
        "📏 評價指標詳解",
        "🏆 聚類算法比較"
    ],
    key="page_navigation"
)

# 數據集選擇
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 數據集選擇")
dataset_choice = st.sidebar.selectbox("選擇數據集：", [
    "鳶尾花", "紅酒", "人工球形", "月亮形狀", "乳腺癌", "手寫數字"
], key="dataset_selection")

# 數據集載入函數
@st.cache_data
def load_datasets():
    datasets = {}
    
    # 1. 鳶尾花數據集 (經典3類)
    iris = load_iris()
    iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
    iris_df['true_labels'] = iris.target
    datasets['鳶尾花'] = iris_df
    
    # 2. 紅酒數據集 (3類化學成分)
    wine = load_wine()
    wine_df = pd.DataFrame(wine.data, columns=wine.feature_names)
    wine_df['true_labels'] = wine.target
    datasets['紅酒'] = wine_df
    
    # 3. 乳腺癌數據集 (2類醫療數據)
    cancer = load_breast_cancer()
    cancer_df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
    cancer_df['true_labels'] = cancer.target
    datasets['乳腺癌'] = cancer_df
    
    # 4. 手寫數字數據集 (10類，降維到2D展示)
    digits = load_digits()
    # 使用PCA降維到2D便於可視化
    pca = PCA(n_components=8)  # 保留8個主成分
    digits_reduced = pca.fit_transform(digits.data)
    digits_df = pd.DataFrame(digits_reduced, columns=[f'PC{i+1}' for i in range(8)])
    digits_df['true_labels'] = digits.target
    datasets['手寫數字'] = digits_df
    
    # 5. 人工球形數據集 (可控制的理想聚類)
    np.random.seed(42)
    blobs_X, blobs_y = make_blobs(n_samples=300, centers=4, n_features=2, 
                                  random_state=42, cluster_std=1.0)
    blobs_df = pd.DataFrame(blobs_X, columns=['Feature_1', 'Feature_2'])
    blobs_df['true_labels'] = blobs_y
    datasets['人工球形'] = blobs_df
    
    # 6. 月亮形狀數據集 (非線性聚類)
    np.random.seed(42)
    moons_X, moons_y = make_moons(n_samples=300, noise=0.1, random_state=42)
    moons_df = pd.DataFrame(moons_X, columns=['Feature_1', 'Feature_2'])
    moons_df['true_labels'] = moons_y
    datasets['月亮形狀'] = moons_df
    
    return datasets

all_datasets = load_datasets()

# 通用數據獲取函數
def get_current_data():
    current_dataset = all_datasets[dataset_choice]
    X = current_dataset.drop('true_labels', axis=1)
    true_labels = current_dataset['true_labels']
    return X, true_labels

# 數據集簡介 (按複雜度排序)
dataset_info = {
    "鳶尾花": "🌸 入門級-經典3類，4特徵，適合所有算法 (150樣本)",
    "紅酒": "🍷 簡單級-3類化學，13特徵，球形聚類 (178樣本)", 
    "人工球形": "⭕ 演示級-4個理想球形，2特徵，演示用 (300樣本)",
    "月亮形狀": "🌙 中等級-非線性2類，2特徵，密度聚類 (300樣本)",
    "乳腺癌": "🏥 進階級-2類醫療，30特徵，特徵豐富 (569樣本)",
    "手寫數字": "✏️ 挑戰級-10類高維，8特徵PCA，最複雜 (1797樣本)"
}

st.sidebar.markdown("### 📝 數據集特點")
for dataset, description in dataset_info.items():
    if dataset == dataset_choice:
        st.sidebar.markdown(f'<div class="small-text">✅ <strong>{dataset}</strong>: {description}</div>', unsafe_allow_html=True)
    else:
        st.sidebar.markdown(f'<div class="small-text"><strong>{dataset}</strong>: {description}</div>', unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.markdown("### 👨‍💻 作者信息")
st.sidebar.info("**This tutorial was made by CCChang18** 🚀")

# 主要內容區域
if page == "🏠 無監督學習概述":
    st.markdown('<h1 class="main-header">無監督學習-聚類互動教學平台</h1>', unsafe_allow_html=True)
    
    st.markdown("## 🏅 什麼是無監督學習？")
    
    st.markdown("""
    **無監督學習**是機器學習的重要分支，其特點是：
    
    1. **無標籤數據**：只有輸入特徵，沒有對應的標準答案
    2. **發現隱藏模式**：從數據中挖掘潛在的結構和規律
    3. **探索性分析**：幫助理解數據的內在特性
    """)
    
    st.markdown("## 🎯 什麼是聚類分析？")
    
    st.markdown("""
    聚類(Clustering)是無監督學習的核心任務之一：
    
    1. **相似性分組**：將相似的數據點分為同一組
    2. **數據結構發現**：揭示數據的自然分組模式
    3. **無需先驗知識**：不需要事先知道分組數量或標準
    """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 🔧 核心算法")
        st.markdown("""
        - K-Means 聚類
        - DBSCAN 密度聚類
        - 層次聚類
        - 高斯混合模型
        - 譜聚類
        - BIRCH 聚類
        """)
    
    with col2:
        st.markdown("### 📏 評價指標")
        st.markdown("""
        **內部指標**：
        - 輪廓係數
        - CH指數
        - DB指數
        
        **外部指標**：
        - 調整蘭德指數
        - 標準化互信息
        """)
    
    with col3:
        st.markdown("### 🎯 應用場景")
        st.markdown("""
        - 客戶群體分析
        - 市場細分
        - 圖像分割
        - 基因序列分析
        - 社交網絡分析
        - 推薦系統
        """)
    
    st.markdown("## 🎯 學習目標")
    st.info("""
    通過本課程，您將能夠：
    1. 理解不同聚類算法的原理和適用場景
    2. 掌握聚類結果的評估和解釋方法
    3. 學會選擇合適的聚類算法和參數
    4. 了解聚類分析在實際應用中的價值
    """)
    
    st.markdown("## 🗺️ 六大聚類算法特點比較")
    
    # 創建算法比較圖表
    algorithms_data = {
        '算法': ['K-Means', 'DBSCAN', 'Agglomerative', 'GMM', 'Spectral', 'BIRCH'],
        '需要預設k': ['是', '否', '是', '是', '是', '是'],
        '處理噪聲': ['差', '優', '中', '中', '差', '差'],
        '任意形狀': ['差', '優', '中', '中', '優', '差'],
        '計算速度': ['快', '中', '慢', '中', '慢', '快'],
        '大數據集': ['優', '中', '差', '中', '差', '優'],
        '適用場景': ['球形聚類', '密度聚類', '層次結構', '軟聚類', '非凸聚類', '增量聚類']
    }
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # 創建雷達圖比較算法特點
        import plotly.graph_objects as go
        
        # 定義評分標準 (1-5分)
        algorithm_scores = {
            'K-Means': [5, 2, 2, 5, 5],      # 速度快、需要預設k、不能處理噪聲、球形、適合大數據
            'DBSCAN': [3, 5, 5, 3, 3],       # 中等速度、不需要k、處理噪聲好、任意形狀、中等大數據
            'Agglomerative': [2, 3, 3, 2, 1], # 慢、需要預設k、中等噪聲、中等形狀、不適合大數據
            'GMM': [3, 3, 3, 3, 3],          # 中等各項能力
            'Spectral': [2, 2, 5, 2, 1],     # 慢、需要預設k、不處理噪聲、非凸形狀、不適合大數據
            'BIRCH': [5, 2, 2, 5, 5]         # 快、需要預設k、不處理噪聲、球形、適合大數據
        }
        
        categories = ['計算速度', '噪聲處理', '任意形狀', '球形聚類', '大數據集']
        
        fig = go.Figure()
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
        
        for i, (algo, scores) in enumerate(algorithm_scores.items()):
            fig.add_trace(go.Scatterpolar(
                r=scores,
                theta=categories,
                fill='toself',
                name=algo,
                line_color=colors[i],
                fillcolor=colors[i],
                opacity=0.6
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 5],
                    tickvals=[1, 2, 3, 4, 5],
                    ticktext=['差', '一般', '中等', '好', '優秀']
                )
            ),
            showlegend=True,
            title="六大聚類算法能力雷達圖",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 📋 算法選擇指南")
        
        st.success("""
        **🎯 球形聚類**
        - K-Means (首選)
        - BIRCH (大數據)
        - GMM (軟聚類)
        """)
        
        st.info("""
        **🌊 任意形狀**
        - DBSCAN (密度)
        - Spectral (非凸)
        - Agglomerative (層次)
        """)
        
        st.warning("""
        **🚫 有噪聲數據**
        - DBSCAN (最佳)
        - Agglomerative (次選)
        - GMM (第三)
        """)
        
        st.error("""
        **💾 大數據集**
        - BIRCH (增量)
        - K-Means (經典)
        - DBSCAN (中等)
        """)
    
    # 算法特點表格
    st.markdown("### 📊 詳細特點對比表")
    import pandas as pd
    df_comparison = pd.DataFrame(algorithms_data)
    
    # 使用顏色標記優缺點
    def highlight_performance(val):
        if val in ['優', '是', '快']:
            color = 'background-color: #d4edda'  # 綠色
        elif val in ['差', '否', '慢']:
            color = 'background-color: #f8d7da'  # 紅色
        elif val in ['中', '一般']:
            color = 'background-color: #fff3cd'  # 黃色
        else:
            color = ''
        return color
    
    styled_df = df_comparison.style.applymap(highlight_performance)
    st.dataframe(styled_df, use_container_width=True)
    
    # 聚類算法與數據結構關係示意圖
    st.markdown("## 🎨 聚類算法與數據結構關係示意圖")
    st.markdown("以下示意圖展示了6種聚類算法在不同數據結構上的表現，幫助您直觀理解各算法的特點：")
    
    # 創建示範數據集
    @st.cache_data
    def create_demo_datasets():
        """創建6個不同特性的示範數據集"""
        np.random.seed(0)
        n_samples = 1500
        
        datasets = []
        
        # 1. 嘈雜圓環 - 測試非凸聚類能力
        noisy_circles = make_circles(n_samples=n_samples, factor=.5, noise=.05)
        datasets.append(("嘈雜圓環", noisy_circles[0], noisy_circles[1]))
        
        # 2. 嘈雜月牙 - 測試非線性聚類能力  
        noisy_moons = make_moons(n_samples=n_samples, noise=.05)
        datasets.append(("嘈雜月牙", noisy_moons[0], noisy_moons[1]))
        
        # 3. 球形聚類 - 測試標準聚類能力
        blobs = make_blobs(n_samples=n_samples, random_state=8, centers=3)
        datasets.append(("球形聚類", blobs[0], blobs[1]))
        
        # 4. 隨機數據 - 測試抗噪聲能力
        no_structure = np.random.rand(n_samples, 2)
        datasets.append(("隨機分佈", no_structure, np.zeros(n_samples)))
        
        # 5. 變密度聚類 - 測試密度適應能力
        # 創建不同密度的聚類
        X1, y1 = make_blobs(n_samples=n_samples//3, centers=1, cluster_std=0.1, center_box=(0, 0), random_state=42)
        X2, y2 = make_blobs(n_samples=n_samples//3, centers=1, cluster_std=0.3, center_box=(2, 2), random_state=42)
        X3, y3 = make_blobs(n_samples=n_samples//3, centers=1, cluster_std=0.05, center_box=(1, -1), random_state=42)
        varied_density_X = np.vstack([X1, X2, X3])
        varied_density_y = np.hstack([y1, y2+1, y3+2])
        datasets.append(("變密度聚類", varied_density_X, varied_density_y))
        
        # 6. 長橢圓聚類 - 測試非球形聚類能力
        elongated_X, elongated_y = make_blobs(n_samples=n_samples, centers=2, cluster_std=1.5, random_state=42)
        # 拉伸數據使其變成橢圓形
        transformation = np.array([[0.6, -0.6], [-0.4, 0.8]])
        elongated_X = np.dot(elongated_X, transformation)
        datasets.append(("長橢圓聚類", elongated_X, elongated_y))
        
        return datasets
    
    demo_datasets = create_demo_datasets()
    
    # 聚類算法配置
    clustering_algorithms = [
        ("K-Means", lambda: KMeans(n_clusters=2, random_state=42, n_init=10)),
        ("DBSCAN", lambda: DBSCAN(eps=0.3, min_samples=5)),
        ("Agglomerative", lambda: AgglomerativeClustering(n_clusters=2)),
        ("GMM", lambda: GaussianMixture(n_components=2, random_state=42)),
        ("Spectral", lambda: SpectralClustering(n_clusters=2, random_state=42)),
        ("BIRCH", lambda: Birch(n_clusters=2, threshold=0.5))
    ]
    
    # 執行聚類並創建可視化
    fig = make_subplots(
        rows=6, cols=6,
        subplot_titles=[f"{ds_name} + {algo_name}" for ds_name, _, _ in demo_datasets for algo_name, _ in clustering_algorithms],
        horizontal_spacing=0.05,
        vertical_spacing=0.08,
        specs=[[{"type": "scatter"}] * 6 for _ in range(6)]
    )
    
    # 統一的顏色配色方案
    colors = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6', '#1ABC9C']
    
    # 為每個數據集和算法組合創建聚類結果
    for ds_idx, (ds_name, X_data, y_true) in enumerate(demo_datasets):
        # 標準化數據
        X_scaled = StandardScaler().fit_transform(X_data)
        
        for algo_idx, (algo_name, algo_func) in enumerate(clustering_algorithms):
            try:
                # 執行聚類
                model = algo_func()
                if algo_name == "GMM":
                    cluster_labels = model.fit_predict(X_scaled)
                else:
                    cluster_labels = model.fit_predict(X_scaled)
                
                # 處理噪聲點（DBSCAN）
                unique_labels = np.unique(cluster_labels)
                point_colors = []
                for label in cluster_labels:
                    if label == -1:
                        # 噪聲點用灰色
                        point_colors.append('#888888')
                    else:
                        # 正常聚類點根據標籤分配顏色
                        point_colors.append(colors[label % len(colors)])
                
                # 添加散點圖
                fig.add_trace(
                    go.Scatter(
                        x=X_data[:, 0],
                        y=X_data[:, 1],
                        mode='markers',
                        marker=dict(
                            color=point_colors,
                            size=4,
                            opacity=0.8
                        ),
                        showlegend=False
                    ),
                    row=ds_idx + 1, col=algo_idx + 1
                )
                
            except Exception:
                # 如果算法失敗，顯示原始數據
                fig.add_trace(
                    go.Scatter(
                        x=X_data[:, 0],
                        y=X_data[:, 1],
                        mode='markers',
                        marker=dict(
                            color='gray',
                            size=4,
                            opacity=0.5
                        ),
                        showlegend=False
                    ),
                    row=ds_idx + 1, col=algo_idx + 1
                )
    
    # 更新布局
    fig.update_layout(
        height=1200,
        showlegend=False
    )
    
    # 隱藏軸標籤以保持簡潔
    for i in range(1, 7):
        for j in range(1, 7):
            fig.update_xaxes(showticklabels=False, row=i, col=j)
            fig.update_yaxes(showticklabels=False, row=i, col=j)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # 添加說明
    st.markdown("### 📝 示意圖解讀")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🔍 數據集特點")
        st.markdown("""
        - **嘈雜圓環**: 測試處理非凸形狀的能力
        - **嘈雜月牙**: 測試處理彎曲邊界的能力  
        - **球形聚類**: 測試標準聚類場景的表現
        - **隨機分佈**: 測試在無結構數據中的穩定性
        - **變密度聚類**: 測試處理不同密度聚類的能力
        - **長橢圓聚類**: 測試處理非球形聚類的能力
        """)
    
    with col2:
        st.markdown("#### 🎯 關鍵觀察")
        st.markdown("""
        - **K-Means**: 在球形數據表現最佳，其他形狀效果差
        - **DBSCAN**: 在密度變化和噪聲處理方面表現優秀
        - **Spectral**: 在非凸形狀（圓環、月牙）表現最佳
        - **Agglomerative**: 在多數情況下表現穩定
        - **GMM**: 處理橢圓形聚類效果良好
        - **BIRCH**: 快速但傾向於球形聚類
        """)
    
    st.success("""
    💡 **選擇建議**: 
    - 數據形狀規則 → K-Means、GMM、BIRCH
    - 數據形狀復雜 → DBSCAN、Spectral、Agglomerative  
    - 有噪聲數據 → DBSCAN
    - 大數據集 → BIRCH、K-Means
    """)

elif page == "📊 數據集探索":
    st.markdown('<h1 class="main-header">📊 數據集探索</h1>', unsafe_allow_html=True)
    
    st.info("💡 您可以在左側選擇不同的數據集來探索其特性")
    
    # 獲取當前選擇的數據集
    X, true_labels = get_current_data()
    
    # 數據集信息映射 (按複雜度排序)
    dataset_descriptions = {
        "鳶尾花": {
            "title": "🌸 鳶尾花數據集 (入門級)",
            "description": "經典3類花卉分類數據集，4特徵，聚類入門首選",
            "n_classes": 3,
            "features": ["花萼長度", "花萼寬度", "花瓣長度", "花瓣寬度"],
            "color": "lightblue",
            "complexity": "入門級"
        },
        "紅酒": {
            "title": "🍷 紅酒化學成分數據集 (簡單級)",
            "description": "178個紅酒樣本，13種化學成分，適合球形聚類算法",
            "n_classes": 3,
            "features": ["酒精", "蘋果酸", "灰分", "鹼性", "鎂", "總酚", "類黃酮等"],
            "color": "lightcoral",
            "complexity": "簡單級"
        },
        "人工球形": {
            "title": "⭕ 人工球形聚類數據集 (演示級)",
            "description": "人工生成的4個理想球形聚類，2特徵，算法演示專用",
            "n_classes": 4,
            "features": ["Feature_1", "Feature_2"],
            "color": "lightpink",
            "complexity": "演示級"
        },
        "月亮形狀": {
            "title": "🌙 月亮形狀數據集 (中等級)",
            "description": "非線性月牙形數據，2特徵，測試密度聚類算法",
            "n_classes": 2,
            "features": ["Feature_1", "Feature_2"],
            "color": "lightgray",
            "complexity": "中等級"
        },
        "乳腺癌": {
            "title": "🏥 乳腺癌診斷數據集 (進階級)",
            "description": "基於細胞核特徵的良性/惡性診斷，30特徵高維數據",
            "n_classes": 2,
            "features": ["半徑", "紋理", "周長", "面積", "光滑度等30個特徵"],
            "color": "lightgreen",
            "complexity": "進階級"
        },
        "手寫數字": {
            "title": "✏️ 手寫數字數據集 (挑戰級)",
            "description": "8x8像素手寫數字圖像，10類別，PCA降維後最複雜",
            "n_classes": 10,
            "features": ["PCA降維後的8個主成分"],
            "color": "lightyellow",
            "complexity": "挑戰級"
        }
    }
    
    desc = dataset_descriptions[dataset_choice or "鳶尾花"]
    st.markdown(f"## {desc['title']}")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📋 數據集資訊")
        st.info(f"""
        - **樣本數量**: {len(X)} 個樣本
        - **特徵數量**: {len(X.columns)} 個特徵
        - **真實類別數**: {desc['n_classes']} 類
        - **複雜度**: {desc['complexity']}
        - **描述**: {desc['description']}
        """)
    
    with col2:
        st.markdown("### 🔬 特徵說明")
        for feature in desc['features']:
            st.markdown(f"- {feature}")
    
    # 數據可視化
    st.markdown("### 📈 數據可視化")
    
    if len(X.columns) >= 2:
        # 選擇前兩個特徵進行可視化
        feature_x = X.columns[0]
        feature_y = X.columns[1]
        
        fig = px.scatter(
            x=X[feature_x], 
            y=X[feature_y],
            color=true_labels,  # 使用數值，配合連續色彩映射
            title=f"真實標籤分布: {feature_x} vs {feature_y}",
            labels={'color': '真實類別'},
            color_continuous_scale='viridis'
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    # 特徵分布
    st.markdown("### 📊 特徵分布分析")
    
    # 選擇要展示的特徵（對所有數據集都顯示）
    max_default_features = min(4, len(X.columns))
    selected_features = st.multiselect(
        "選擇要展示的特徵：",
        X.columns.tolist(),
        default=X.columns.tolist()[:max_default_features],
        key="dataset_exploration_features",
        help="可以選擇最多4個特徵進行分布分析"
    )
    
    if selected_features:
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=selected_features[:4]
        )
        
        for i, feature in enumerate(selected_features[:4]):
            row = i // 2 + 1
            col = i % 2 + 1
            fig.add_trace(
                go.Histogram(x=X[feature], name=feature, nbinsx=20),
                row=row, col=col
            )
        
        fig.update_layout(height=600, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("請選擇至少一個特徵來查看分布分析")

elif page == "🎯 K-Means聚類":
    st.markdown('<h1 class="main-header">🎯 K-Means聚類</h1>', unsafe_allow_html=True)
    
    st.markdown("## 🧮 算法原理")
    
    st.markdown("### 📐 K-Means算法步驟")
    st.markdown("""
    K-Means是最經典的聚類算法，通過迭代優化聚類中心：
    
    1. **初始化**：隨機選擇k個聚類中心
    2. **分配**：將每個點分配給最近的聚類中心
    3. **更新**：重新計算每個聚類的中心點
    4. **重複**：直到聚類中心不再變化
    """)
    
    st.latex(r'''
    \text{目標函數：} \min \sum_{i=1}^{k} \sum_{x \in C_i} ||x - \mu_i||^2
    ''')
    
    st.markdown("### 🔤 公式變數解釋")
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("""
        **主要變數含義**：
        - **k**: 聚類數量（需要預先設定）
        - **Cᵢ**: 第i個聚類包含的所有數據點
        - **x**: 聚類中的單個數據點
        - **μᵢ**: 第i個聚類的中心點（重心）
        """)
    
    with col2:
        st.success("""
        **目標函數解讀**：
        - **||x - μᵢ||²**: 點x到聚類中心的歐氏距離的平方
        - **∑**: 對所有點和聚類求和
        - **目標**: 最小化所有點到其聚類中心的距離平方和
        - **意義**: 讓聚類內部更緊密，中心更代表性
        """)
    
    st.markdown("### ⚖️ 模型優缺點")


    col1, col2 = st.columns(2)
    
    with col1:
        st.success("### ✅ 優點")
        st.markdown("""
        - 🚀 **簡單高效**：算法直觀易懂
        - ⚡ **計算速度快**：時間複雜度線性
        - 🎯 **適合球形聚類**：處理圓形/球形簇
        - 📊 **可解釋性強**：結果易於理解
        - 💾 **記憶體效率高**：只需存儲聚類中心
        """)
    
    with col2:
        st.error("### ❌ 缺點")
        st.markdown("""
        - 🎲 **需要預設k值**：需要事先知道聚類數
        - 🔄 **對初始化敏感**：不同初始值可能導致不同結果
        - 📐 **假設球形聚類**：無法處理任意形狀
        - 🎯 **對離群值敏感**：極值會影響聚類中心
        """)
    
    st.markdown("## 💡 K-Means使用建議")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.success("### ✅ 新手友好的選擇")
        st.markdown("""
        **為什麼K-Means是入門首選？**
        - 🎯 **直觀易懂**：概念簡單，結果可解釋
        - ⚡ **速度快**：適合快速驗證想法
        - 📚 **資料豐富**：網上教程和案例很多
        - 🔧 **參數簡單**：主要只需調整k值
        """)
    
    with col2:
        st.info("### 🎯 實用調參技巧")
        st.markdown("""
        **如何選擇最佳k值？**
        - 📊 **肘部法則**：成本下降趨勢的轉折點
        - 📏 **輪廓係數**：選擇使輪廓係數最大的k
        - 🧠 **業務理解**：結合實際業務需求
        - 🔄 **多次嘗試**：k-means++減少隨機性
        """)
    
    st.warning("""
    💡 **快速上手建議**：
    1. 先用預設k=3開始實驗
    2. 觀察聚類結果是否符合直覺  
    3. 使用輪廓係數評估不同k值
    4. 確保特徵已標準化（特別重要！）
    5. 如果聚類形狀不規則，考慮其他算法
    """)
    
    st.markdown("## 🎛️ 互動式實驗")
    
    X, true_labels = get_current_data()
    
    # 參數設置
    col1, col2 = st.columns(2)
    
    with col1:
        n_clusters = st.slider("聚類數量 (k)：", 2, 10, 3, key="kmeans_n_clusters")
        init_method = st.selectbox("初始化方法：", ["k-means++", "random"], key="kmeans_init_method",
                                   help="k-means++：智能初始化，選擇相互距離較遠的初始中心，收斂更快更穩定；random：隨機選擇初始中心，可能需要更多迭代")
        max_iter = st.slider("最大迭代次數：", 100, 1000, 300, key="kmeans_max_iter")
    
    with col2:
        n_init = st.slider("不同初始化次數：", 1, 20, 10, key="kmeans_n_init")
        random_state = st.slider("隨機種子：", 1, 100, 42, key="kmeans_random_state")
        
        # 特徵選擇（如果特徵太多）
        if len(X.columns) > 2:
            selected_features = st.multiselect(
                "選擇用於聚類的特徵：",
                X.columns.tolist(),
                default=X.columns.tolist()[:2],
                key="kmeans_selected_features"
            )
        else:
            selected_features = X.columns.tolist()
    
    if len(selected_features) >= 2:
        X_selected = X[selected_features]
        
        # 標準化選項
        normalize = st.checkbox("是否進行特徵標準化？", value=True,
                               help="K-Means對特徵尺度敏感，強烈建議標準化以確保所有特徵等權重貢獻",
                               key="kmeans_normalize")
        
        if normalize:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_selected)
            X_for_clustering = X_scaled
        else:
            X_for_clustering = X_selected.values
        
        # 執行K-Means聚類
        kmeans = KMeans(
            n_clusters=n_clusters,
            init=init_method or "k-means++",
            max_iter=max_iter,
            n_init="auto" if n_init == 10 else n_init,
            random_state=random_state
        )
        
        cluster_labels = kmeans.fit_predict(X_for_clustering)
        
        # 計算評價指標
        silhouette_avg = silhouette_score(X_for_clustering, cluster_labels)
        ch_score = calinski_harabasz_score(X_for_clustering, cluster_labels)
        db_score = davies_bouldin_score(X_for_clustering, cluster_labels)
        
        # 如果有真實標籤，計算外部指標
        if len(np.unique(true_labels)) > 1:
            ari_score = adjusted_rand_score(true_labels, cluster_labels)
            nmi_score = normalized_mutual_info_score(true_labels, cluster_labels)
            fmi_score = fowlkes_mallows_score(true_labels, cluster_labels)
        
        # 顯示結果
        st.markdown("### 📊 聚類結果")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("輪廓係數", f"{silhouette_avg:.4f}")
            st.metric("CH指數", f"{ch_score:.2f}")
        
        with col2:
            st.metric("DB指數", f"{db_score:.4f}")
            st.metric("迭代次數", f"{kmeans.n_iter_}")
        
        with col3:
            if len(np.unique(true_labels)) > 1:
                st.metric("調整蘭德指數", f"{ari_score:.4f}")
                st.metric("標準化互信息", f"{nmi_score:.4f}")
        
        # 可視化結果
        st.markdown("### 📈 聚類結果可視化")
        
        if len(selected_features) >= 2:
            col1, col2 = st.columns(2)
            
            with col1:
                # K-Means聚類結果 - 使用plasma色彩映射
                fig = px.scatter(
                    x=X_selected.iloc[:, 0], 
                    y=X_selected.iloc[:, 1],
                    color=cluster_labels,  # 使用數值，配合連續色彩映射
                    title=f"K-Means聚類結果 ({n_clusters}個聚類, {n_clusters}個中心)",
                    labels={'color': '聚類標籤'},
                    color_continuous_scale='plasma'
                )
                
                # 添加聚類中心
                if normalize:
                    centers_original = scaler.inverse_transform(kmeans.cluster_centers_)
                else:
                    centers_original = kmeans.cluster_centers_
                
                fig.add_scatter(
                    x=centers_original[:, 0],
                    y=centers_original[:, 1],
                    mode='markers',
                    marker=dict(symbol='x', size=15, color='black'),
                    name='聚類中心'
                )
                
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # 真實標籤 - 使用viridis色彩映射
                fig = px.scatter(
                    x=X_selected.iloc[:, 0], 
                    y=X_selected.iloc[:, 1],
                    color=true_labels,  # 使用數值，配合連續色彩映射
                    title="真實標籤分布",
                    labels={'color': '真實標籤'},
                    color_continuous_scale='viridis'
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
        
        # 評價指標解釋
        st.markdown("### 📏 評價指標解釋")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 內部指標（無需真實標籤）")
            
            if silhouette_avg > 0.7:
                st.success(f"✅ 輪廓係數 {silhouette_avg:.4f} - 聚類效果優秀")
            elif silhouette_avg > 0.5:
                st.info(f"ℹ️ 輪廓係數 {silhouette_avg:.4f} - 聚類效果良好")
            else:
                st.warning(f"⚠️ 輪廓係數 {silhouette_avg:.4f} - 聚類效果較差")
            
            st.markdown(f"- **CH指數**: {ch_score:.2f} (越高越好)")
            st.markdown(f"- **DB指數**: {db_score:.4f} (越低越好)")
        
        with col2:
            if len(np.unique(true_labels)) > 1:
                st.markdown("#### 外部指標（與真實標籤比較）")
                
                if ari_score > 0.8:
                    st.success(f"✅ ARI {ari_score:.4f} - 與真實標籤高度一致")
                elif ari_score > 0.5:
                    st.info(f"ℹ️ ARI {ari_score:.4f} - 與真實標籤較為一致")
                else:
                    st.warning(f"⚠️ ARI {ari_score:.4f} - 與真實標籤一致性較差")
                
                st.markdown(f"- **NMI**: {nmi_score:.4f} (0-1，越高越好)")
                st.markdown(f"- **FMI**: {fmi_score:.4f} (0-1，越高越好)")
    
    else:
        st.warning("⚠️ 請至少選擇2個特徵進行聚類實驗。")

elif page == "🌊 DBSCAN密度聚類":
    st.markdown('<h1 class="main-header">🌊 DBSCAN密度聚類</h1>', unsafe_allow_html=True)
    
    st.markdown("## 🧮 算法原理")
    
    st.markdown("### 📐 DBSCAN算法概念")
    st.markdown("""
    DBSCAN (Density-Based Spatial Clustering of Applications with Noise) 是基於密度的聚類算法：
    
    **🔍 核心概念解釋**：
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("""
        **🎯 核心點 (Core Point)**
        - 在半徑ε範圍內至少有MinPts個鄰居
        - 想像成"人口密集區的中心"
        - 例：ε=0.5, MinPts=5，某點周圍0.5距離內有≥5個點
        """)
        
        st.warning("""
        **🔘 邊界點 (Border Point)**  
        - 本身不是核心點，但在某個核心點的ε鄰域內
        - 想像成"住在城市邊緣的居民"
        - 例：該點周圍只有3個點，但距離某核心點<0.5
        """)
    
    with col2:
        st.error("""
        **❌ 噪聲點 (Noise Point)**
        - 既不是核心點，也不是邊界點
        - 想像成"偏遠地區的孤立點"
        - 例：周圍鄰居很少，且距離所有核心點都>ε
        """)
        
        st.success("""
        **🔗 密度可達 (Density-Reachable)**
        - 從核心點出發，通過其他核心點可以"跳躍"到達
        - 想像成"通過城市間道路網絡可達"
        """)
    
    st.markdown("### 🎛️ 參數含義")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **ε (Epsilon) - 鄰域半徑**
        - 定義"鄰居"的距離範圍
        - 🔹 ε太小：許多點變成噪聲
        - 🔹 ε太大：所有點合併成一個聚類
        - 💡 建議：先用散點圖觀察數據密度
        """)
    
    with col2:
        st.markdown("""
        **MinPts - 最小點數**
        - 成為核心點需要的最少鄰居數
        - 🔹 MinPts太小：產生很多小聚類
        - 🔹 MinPts太大：產生很多噪聲點
        - 💡 建議：通常設為 2×維度 或更大
        """)
    
    st.markdown("### ⚖️ 模型優缺點")

    col1, col2 = st.columns(2)
    
    with col1:
        st.success("### ✅ 優點")
        st.markdown("""
        - 🌊 **任意形狀聚類**：能發現任意形狀的聚類
        - 🚫 **自動檢測噪聲**：能識別和排除離群值
        - 📊 **無需預設聚類數**：自動確定聚類數量
        - 🛡️ **對離群值穩健**：不受噪聲點影響
        - 🎯 **基於密度**：適合密度不均勻的數據
        """)
    
    with col2:
        st.error("### ❌ 缺點")
        st.markdown("""
        - 🎛️ **參數選擇困難**：ε和MinPts需要調優
        - 📏 **對密度變化敏感**：難以處理密度差異大的聚類
        - 📐 **高維數據表現差**：維度詛咒問題
        - ⚖️ **對參數敏感**：小的參數變化可能導致大的結果差異
        """)
    
    st.markdown("## 🎛️ 互動式實驗")
    
    X, true_labels = get_current_data()
    
    # 參數設置
    col1, col2 = st.columns(2)
    
    with col1:
        eps = st.slider("鄰域半徑 (ε)：", 0.1, 5.0, 0.5, 0.1, key="dbscan_eps")
        min_samples = st.slider("最小樣本數 (MinPts)：", 2, 20, 5, key="dbscan_min_samples")
    
    with col2:
        metric = st.selectbox("距離度量：", ["euclidean", "manhattan", "chebyshev"], key="dbscan_metric")
        
        # 特徵選擇（如果特徵太多）
        if len(X.columns) > 2:
            selected_features = st.multiselect(
                "選擇用於聚類的特徵：",
                X.columns.tolist(),
                default=X.columns.tolist()[:2],
                key="dbscan_selected_features"
            )
        else:
            selected_features = X.columns.tolist()
    
    if len(selected_features) >= 2:
        X_selected = X[selected_features]
        
        # 標準化選項
        normalize = st.checkbox("是否進行特徵標準化？", value=True,
                               help="DBSCAN基於距離計算，不同尺度特徵會影響密度估計，建議標準化",
                               key="dbscan_normalize")
        
        if normalize:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_selected)
            X_for_clustering = X_scaled
        else:
            X_for_clustering = X_selected.values
        
        # 執行DBSCAN聚類
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric=metric or "euclidean")
        cluster_labels = dbscan.fit_predict(X_for_clustering)
        
        # 統計結果
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        n_noise = list(cluster_labels).count(-1)
        
        # 計算評價指標（排除噪聲點）
        if n_clusters > 1:
            # 只對非噪聲點計算指標
            mask = cluster_labels != -1
            if np.sum(mask) > 0:
                X_no_noise = X_for_clustering[mask]
                labels_no_noise = cluster_labels[mask]
                
                if len(np.unique(labels_no_noise)) > 1:
                    silhouette_avg = silhouette_score(X_no_noise, labels_no_noise)
                    ch_score = calinski_harabasz_score(X_no_noise, labels_no_noise)
                    db_score = davies_bouldin_score(X_no_noise, labels_no_noise)
                else:
                    silhouette_avg = ch_score = db_score = np.nan
            else:
                silhouette_avg = ch_score = db_score = np.nan
        else:
            silhouette_avg = ch_score = db_score = np.nan
        
        # 如果有真實標籤，計算外部指標
        if len(np.unique(true_labels)) > 1:
            ari_score = adjusted_rand_score(true_labels, cluster_labels)
            nmi_score = normalized_mutual_info_score(true_labels, cluster_labels)
            fmi_score = fowlkes_mallows_score(true_labels, cluster_labels)
        
        # 顯示結果
        st.markdown("### 📊 聚類結果")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("聚類數量", f"{n_clusters}")
            st.metric("噪聲點數量", f"{n_noise}")
            st.metric("噪聲比例", f"{n_noise/len(X_selected)*100:.1f}%")
        
        with col2:
            if not np.isnan(silhouette_avg):
                st.metric("輪廓係數", f"{silhouette_avg:.4f}")
                st.metric("CH指數", f"{ch_score:.2f}")
                st.metric("DB指數", f"{db_score:.4f}")
            else:
                st.metric("輪廓係數", "N/A")
                st.metric("CH指數", "N/A")
                st.metric("DB指數", "N/A")
        
        with col3:
            if len(np.unique(true_labels)) > 1:
                st.metric("調整蘭德指數", f"{ari_score:.4f}")
                st.metric("標準化互信息", f"{nmi_score:.4f}")
                st.metric("FMI指數", f"{fmi_score:.4f}")
        
        # 可視化結果
        st.markdown("### 📈 聚類結果可視化")
        
        if len(selected_features) >= 2:
            col1, col2 = st.columns(2)
            
            with col1:
                # DBSCAN聚類結果 - 參考K-Means的做法
                # 先用px.scatter創建基本聚類圖（只包含非噪聲點）
                cluster_mask = cluster_labels != -1
                
                if np.sum(cluster_mask) > 0:
                    # 創建非噪聲點的聚類可視化
                    fig = px.scatter(
                        x=X_selected.iloc[cluster_mask, 0], 
                        y=X_selected.iloc[cluster_mask, 1],
                        color=cluster_labels[cluster_mask],
                        title=f"DBSCAN聚類結果 ({n_clusters}個聚類)",
                        labels={'color': '聚類標籤'},
                        color_continuous_scale='plasma'
                    )
                    
                    # 如果有噪聲點，添加噪聲點（類似K-Means添加聚類中心）
                    if n_noise > 0:
                        noise_mask = cluster_labels == -1
                        fig.add_scatter(
                            x=X_selected.iloc[noise_mask, 0],
                            y=X_selected.iloc[noise_mask, 1],
                            mode='markers',
                            marker=dict(symbol='x', size=10, color='red'),
                            name=f'噪聲點'
                        )
                        fig.update_layout(title=f"DBSCAN聚類結果 ({n_clusters}個聚類, {n_noise}個噪聲點)")
                else:
                    # 全部都是噪聲點的特殊情況
                    fig = px.scatter(
                        x=X_selected.iloc[:, 0], 
                        y=X_selected.iloc[:, 1],
                        title=f"DBSCAN聚類結果 (全部為噪聲點)",
                        color_discrete_sequence=['red']
                    )
                    fig.update_traces(marker=dict(symbol='x', size=10))
                
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # 真實標籤 - 使用viridis色彩映射
                fig = px.scatter(
                    x=X_selected.iloc[:, 0], 
                    y=X_selected.iloc[:, 1],
                    color=true_labels,  # 使用數值，配合連續色彩映射
                    title="真實標籤分布",
                    labels={'color': '真實標籤'},
                    color_continuous_scale='viridis'
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
        
        # 參數調優建議
        st.markdown("### 🎯 參數調優建議")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### ε (鄰域半徑) 調優")
            if n_clusters == 0:
                st.error("❌ ε值過小：所有點都被識別為噪聲")
                st.markdown("**建議**: 增加ε值")
            elif n_clusters == 1:
                st.warning("⚠️ ε值過大：所有點合併為一個聚類")
                st.markdown("**建議**: 減少ε值")
            elif n_noise > len(X_selected) * 0.3:
                st.info("ℹ️ 噪聲點較多：可能ε值偏小")
                st.markdown("**建議**: 適當增加ε值")
            else:
                st.success("✅ ε值較為合適")
        
        with col2:
            st.markdown("#### MinPts (最小樣本數) 調優")
            if min_samples < 4:
                st.info("ℹ️ MinPts較小：可能產生過多小聚類")
                st.markdown("**建議**: 一般設為2*維度 或更大")
            elif min_samples > 10:
                st.warning("⚠️ MinPts較大：可能產生過多噪聲點")
                st.markdown("**建議**: 適當減少MinPts值")
            else:
                st.success("✅ MinPts值較為合適")
        
        # DBSCAN特色分析
        st.markdown("### 🔍 DBSCAN特色分析")
        
        # 分析不同類型的點
        core_samples_mask = np.zeros_like(cluster_labels, dtype=bool)
        core_samples_mask[dbscan.core_sample_indices_] = True
        
        n_core = np.sum(core_samples_mask)
        n_border = np.sum((cluster_labels != -1) & (~core_samples_mask))
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 點類型統計")
            st.markdown(f"- **核心點**: {n_core} 個")
            st.markdown(f"- **邊界點**: {n_border} 個") 
            st.markdown(f"- **噪聲點**: {n_noise} 個")
        
        with col2:
            st.markdown("#### 聚類密度分析")
            if n_clusters > 0:
                for i in range(n_clusters):
                    cluster_size = np.sum(cluster_labels == i)
                    st.markdown(f"- **聚類 {i}**: {cluster_size} 個點")
    
    else:
        st.warning("⚠️ 請至少選擇2個特徵進行聚類實驗。")

elif page == "📏 評價指標詳解":
    st.markdown('<h1 class="main-header">📏 評價指標詳解</h1>', unsafe_allow_html=True)
    
    st.markdown("## 🎯 聚類評價指標概述")
    st.info("💡 聚類評價指標幫助我們量化聚類結果的質量，分為內部指標和外部指標兩類。")
    
    # 創建標籤頁式佈局
    tab1, tab2, tab3 = st.tabs(["📊 內部指標", "🎯 外部指標", "🧪 指標比較實驗"])
    
    with tab1:
        st.markdown("## 📊 內部指標 (Internal Metrics)")
        st.markdown("**內部指標**僅使用數據本身的信息，不需要真實標籤")
        
        # 輪廓係數
        st.markdown("### 🔵 輪廓係數 (Silhouette Score)")
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.markdown("**輪廓係數衡量樣本與其聚類內部和外部的相似性。**")
            
            st.latex(r'''
            s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}
            ''')
            
            st.markdown("其中：")
            st.markdown("- $a(i)$ = 樣本i到同聚類其他點的平均距離")
            st.markdown("- $b(i)$ = 樣本i到最近聚類的平均距離")
            st.markdown("- $s(i) \\in [-1, 1]$，越接近1越好")
        
        with col2:
            st.success("### 📊 輪廓係數範圍")
            st.markdown("""
            - **s(i) ≈ 1**: 🎯 聚類效果優秀
            - **s(i) ≈ 0**: 📊 聚類邊界不明確
            - **s(i) < 0**: ❌ 聚類效果差，可能分錯
            - **s(i) > 0.7**: ✅ 通常認為聚類質量高
            """)
        
        # CH指數
        st.markdown("### 🟡 Calinski-Harabasz指數 (CH指數)")
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.markdown("**CH指數衡量聚類間分散度與聚類內緊密度的比值。**")
            
            st.latex(r'''
            CH = \frac{SS_B/(k-1)}{SS_W/(n-k)}
            ''')
            
            st.markdown("其中：")
            st.markdown("- $SS_B$ = 聚類間平方和")
            st.markdown("- $SS_W$ = 聚類內平方和") 
            st.markdown("- $k$ = 聚類數，$n$ = 樣本數")
            st.markdown("- CH值越大表示聚類效果越好")
        
        with col2:
            st.warning("### 📊 CH指數特點")
            st.markdown("""
            - **優點**: 計算簡單，數值穩定
            - **缺點**: 偏向球形聚類
            - **適用**: 密度均勻的聚類
            - **解釋**: 值越大聚類效果越好
            """)
        
        # DB指數
        st.markdown("### 🟢 Davies-Bouldin指數 (DB指數)")
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.markdown("**DB指數衡量聚類內距離與聚類間距離的平均比值。**")
            
            st.latex(r'''
            DB = \frac{1}{k} \sum_{i=1}^{k} \max_{j \neq i} \frac{\sigma_i + \sigma_j}{d(c_i, c_j)}
            ''')
            
            st.markdown("其中：")
            st.markdown("- $\\sigma_i$ = 聚類i內點到中心的平均距離")
            st.markdown("- $d(c_i, c_j)$ = 聚類中心i和j之間的距離")
            st.markdown("- DB值越小表示聚類效果越好")
        
        with col2:
            st.error("### 📊 DB指數特點")
            st.markdown("""
            - **優點**: 對聚類形狀要求較低
            - **缺點**: 計算複雜度較高
            - **適用**: 各種形狀的聚類
            - **解釋**: 值越小聚類效果越好
            """)
    
    with tab2:
        st.markdown("## 🎯 外部指標 (External Metrics)")
        st.markdown("**外部指標**需要真實標籤，用於評估聚類結果與真實分組的一致性")
        
        # ARI指數
        st.markdown("### 🔵 調整蘭德指數 (Adjusted Rand Index)")
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.markdown("**ARI衡量聚類結果與真實標籤的一致性，已調整隨機效應。**")
            
            st.latex(r'''
            ARI = \frac{RI - E[RI]}{\max(RI) - E[RI]}
            ''')
            
            st.markdown("其中：")
            st.markdown("- $RI$ = 原始蘭德指數")
            st.markdown("- $E[RI]$ = 隨機分配的期望RI值")
            st.markdown("- ARI ∈ [-1, 1]，1表示完美匹配")
        
        with col2:
            st.success("### 📊 ARI指數範圍")
            st.markdown("""
            - **ARI = 1**: 🎯 與真實標籤完全一致
            - **ARI = 0**: 📊 隨機分配水平
            - **ARI < 0**: ❌ 比隨機分配更差
            - **ARI > 0.8**: ✅ 高度一致
            """)
        
        # NMI指數
        st.markdown("### 🟡 標準化互信息 (Normalized Mutual Information)")
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.markdown("**NMI基於信息論，衡量聚類標籤和真實標籤間的互信息。**")
            
            st.latex(r'''
            NMI = \frac{2 \times MI(U, V)}{H(U) + H(V)}
            ''')
            
            st.markdown("其中：")
            st.markdown("- $MI(U, V)$ = 聚類U和真實標籤V的互信息")
            st.markdown("- $H(U), H(V)$ = 熵")
            st.markdown("- NMI ∈ [0, 1]，1表示完美匹配")
        
        with col2:
            st.warning("### 📊 NMI指數特點")
            st.markdown("""
            - **優點**: 對聚類數量不敏感
            - **缺點**: 對噪聲點較敏感
            - **適用**: 不同大小的聚類
            - **解釋**: 值越大一致性越高
            """)
        
        # FMI指數
        st.markdown("### 🟢 Fowlkes-Mallows指數")
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.markdown("**FMI是精確度和召回率的幾何平均值。**")
            
            st.latex(r'''
            FMI = \sqrt{\frac{TP}{TP + FP} \times \frac{TP}{TP + FN}}
            ''')
            
            st.markdown("其中：")
            st.markdown("- $TP$ = 真正例（同類同聚類）")
            st.markdown("- $FP$ = 假正例（異類同聚類）")
            st.markdown("- $FN$ = 假負例（同類異聚類）")
            st.markdown("- FMI ∈ [0, 1]，1表示完美匹配")
        
        with col2:
            st.error("### 📊 FMI指數特點")
            st.markdown("""
            - **優點**: 直觀易理解
            - **缺點**: 對聚類大小敏感
            - **適用**: 平衡的聚類分布
            - **解釋**: 類似F1-score概念
            """)
    
    with tab3:
        st.markdown("## 🧪 指標比較實驗")
        st.markdown("在真實數據上比較不同評價指標的表現")
        
        X, true_labels = get_current_data()
        
        # 特徵選擇
        if len(X.columns) > 2:
            selected_features = st.multiselect(
                "選擇用於聚類的特徵：",
                X.columns.tolist(),
                default=X.columns.tolist()[:2],
                key="metrics_selected_features"
            )
        else:
            selected_features = X.columns.tolist()
        
        if len(selected_features) >= 2:
            X_selected = X[selected_features]
            
            # 標準化
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_selected)
            
            # 測試不同聚類數
            k_range = range(2, min(11, len(X_selected)//2))
            
            results = []
            
            for k in k_range:
                # K-Means聚類
                kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
                cluster_labels = kmeans.fit_predict(X_scaled)
                
                # 計算內部指標
                sil_score = silhouette_score(X_scaled, cluster_labels)
                ch_score = calinski_harabasz_score(X_scaled, cluster_labels)
                db_score = davies_bouldin_score(X_scaled, cluster_labels)
                
                # 計算外部指標
                ari_score = adjusted_rand_score(true_labels, cluster_labels)
                nmi_score = normalized_mutual_info_score(true_labels, cluster_labels)
                fmi_score = fowlkes_mallows_score(true_labels, cluster_labels)
                
                results.append({
                    'k': k,
                    'Silhouette': sil_score,
                    'CH Index': ch_score,
                    'DB Index': db_score,
                    'ARI': ari_score,
                    'NMI': nmi_score,
                    'FMI': fmi_score
                })
            
            results_df = pd.DataFrame(results)
            
            # 可視化結果
            col1, col2 = st.columns(2)
            
            with col1:
                # 內部指標
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=results_df['k'], y=results_df['Silhouette'],
                                        mode='lines+markers', name='Silhouette'))
                fig.add_trace(go.Scatter(x=results_df['k'], y=results_df['CH Index']/results_df['CH Index'].max(),
                                        mode='lines+markers', name='CH Index (標準化)'))
                fig.add_trace(go.Scatter(x=results_df['k'], y=1-results_df['DB Index'],
                                        mode='lines+markers', name='1-DB Index'))
                fig.update_layout(title="內部指標隨聚類數變化", xaxis_title="聚類數k", yaxis_title="指標值")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # 外部指標
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=results_df['k'], y=results_df['ARI'],
                                        mode='lines+markers', name='ARI'))
                fig.add_trace(go.Scatter(x=results_df['k'], y=results_df['NMI'],
                                        mode='lines+markers', name='NMI'))
                fig.add_trace(go.Scatter(x=results_df['k'], y=results_df['FMI'],
                                        mode='lines+markers', name='FMI'))
                fig.update_layout(title="外部指標隨聚類數變化", xaxis_title="聚類數k", yaxis_title="指標值")
                st.plotly_chart(fig, use_container_width=True)
            
            # 最佳聚類數推薦
            st.markdown("### 🎯 最佳聚類數推薦")
            
            # 找到各指標的最佳k值
            best_sil_k = results_df.loc[results_df['Silhouette'].idxmax(), 'k']
            best_ch_k = results_df.loc[results_df['CH Index'].idxmax(), 'k']
            best_db_k = results_df.loc[results_df['DB Index'].idxmin(), 'k']
            best_ari_k = results_df.loc[results_df['ARI'].idxmax(), 'k']
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 內部指標推薦")
                st.info(f"""
                - **輪廓係數最佳**: k = {best_sil_k}
                - **CH指數最佳**: k = {best_ch_k}  
                - **DB指數最佳**: k = {best_db_k}
                """)
            
            with col2:
                st.markdown("#### 外部指標推薦")
                st.info(f"""
                - **ARI最佳**: k = {best_ari_k}
                - **真實聚類數**: {len(np.unique(true_labels))}
                - **一致性**: {'高' if best_ari_k == len(np.unique(true_labels)) else '中等'}
                """)
            
            # 詳細結果表格
            st.markdown("### 📊 詳細指標表格")
            st.dataframe(results_df.round(4), use_container_width=True)
        
        else:
            st.warning("⚠️ 請至少選擇2個特徵進行指標比較實驗。")

elif page == "🏆 聚類算法比較":
    st.markdown('<h1 class="main-header">🏆 聚類算法比較</h1>', unsafe_allow_html=True)
    
    st.markdown("## 🎯 多算法性能比較")
    st.markdown("在相同數據集上比較不同聚類算法的性能，幫助您選擇最適合的算法。")
    
    X, true_labels = get_current_data()
    
    # 參數設置
    col1, col2 = st.columns(2)
    
    with col1:
        # 特徵選擇
        if len(X.columns) > 2:
            selected_features = st.multiselect(
                "選擇用於聚類的特徵：",
                X.columns.tolist(),
                default=X.columns.tolist()[:2],
                key="comparison_selected_features"
            )
        else:
            selected_features = X.columns.tolist()
        
        normalize = st.checkbox("是否進行特徵標準化？", value=True, key="comparison_normalize")
    
    with col2:
        # 算法選擇
        selected_algorithms = st.multiselect(
            "選擇要比較的算法：",
            ["K-Means", "DBSCAN", "Agglomerative", "GMM", "Spectral", "Birch"],
            default=["K-Means", "DBSCAN", "Agglomerative", "GMM", "Spectral", "Birch"],
            key="comparison_selected_algorithms"
        )
        
        n_clusters_default = len(np.unique(true_labels))
        n_clusters = st.slider("聚類數量 (適用於需要預設k的算法)：", 2, 10, n_clusters_default, key="comparison_n_clusters")
    
    if len(selected_features) >= 2 and len(selected_algorithms) > 0:
        X_selected = X[selected_features]
        
        # 標準化
        if normalize:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_selected)
        else:
            X_scaled = X_selected.values
        
        # 執行聚類算法比較
        results = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, algorithm in enumerate(selected_algorithms):
            status_text.text(f'正在執行 {algorithm} 聚類...')
            
            try:
                if algorithm == "K-Means":
                    model = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
                    cluster_labels = model.fit_predict(X_scaled)
                    
                elif algorithm == "DBSCAN":
                    # 自動估算eps參數
                    from sklearn.neighbors import NearestNeighbors
                    neighbors = NearestNeighbors(n_neighbors=5)
                    neighbors_fit = neighbors.fit(X_scaled)
                    distances, indices = neighbors_fit.kneighbors(X_scaled)
                    distances = np.sort(distances[:, 4], axis=0)
                    eps = distances[int(len(distances) * 0.95)]  # 95%分位數作為eps
                    
                    model = DBSCAN(eps=eps, min_samples=5)
                    cluster_labels = model.fit_predict(X_scaled)
                    
                elif algorithm == "Agglomerative":
                    model = AgglomerativeClustering(n_clusters=n_clusters)
                    cluster_labels = model.fit_predict(X_scaled)
                    
                elif algorithm == "GMM":
                    model = GaussianMixture(n_components=n_clusters, random_state=42)
                    cluster_labels = model.fit_predict(X_scaled)
                    
                elif algorithm == "Spectral":
                    model = SpectralClustering(n_clusters=n_clusters, random_state=42)
                    cluster_labels = model.fit_predict(X_scaled)
                    
                elif algorithm == "Birch":
                    model = Birch(n_clusters=n_clusters)
                    cluster_labels = model.fit_predict(X_scaled)
                
                # 統計聚類結果
                n_clusters_found = len(np.unique(cluster_labels))
                n_noise = list(cluster_labels).count(-1) if -1 in cluster_labels else 0
                
                # 計算內部指標
                if n_clusters_found > 1:
                    # 排除噪聲點計算指標
                    if n_noise > 0:
                        mask = cluster_labels != -1
                        X_no_noise = X_scaled[mask]
                        labels_no_noise = cluster_labels[mask]
                        if len(np.unique(labels_no_noise)) > 1:
                            sil_score = silhouette_score(X_no_noise, labels_no_noise)
                            ch_score = calinski_harabasz_score(X_no_noise, labels_no_noise)
                            db_score = davies_bouldin_score(X_no_noise, labels_no_noise)
                        else:
                            sil_score = ch_score = db_score = np.nan
                    else:
                        sil_score = silhouette_score(X_scaled, cluster_labels)
                        ch_score = calinski_harabasz_score(X_scaled, cluster_labels)
                        db_score = davies_bouldin_score(X_scaled, cluster_labels)
                else:
                    sil_score = ch_score = db_score = np.nan
                
                # 計算外部指標
                ari_score = adjusted_rand_score(true_labels, cluster_labels)
                nmi_score = normalized_mutual_info_score(true_labels, cluster_labels)
                fmi_score = fowlkes_mallows_score(true_labels, cluster_labels)
                
                results.append({
                    '算法': algorithm,
                    '發現聚類數': n_clusters_found,
                    '噪聲點數': n_noise,
                    '輪廓係數': sil_score,
                    'CH指數': ch_score,
                    'DB指數': db_score,
                    'ARI': ari_score,
                    'NMI': nmi_score,
                    'FMI': fmi_score,
                    '聚類標籤': cluster_labels
                })
                
            except Exception as e:
                st.warning(f"算法 {algorithm} 執行失敗: {str(e)}")
                results.append({
                    '算法': algorithm,
                    '發現聚類數': 0,
                    '噪聲點數': 0,
                    '輪廓係數': np.nan,
                    'CH指數': np.nan,
                    'DB指數': np.nan,
                    'ARI': np.nan,
                    'NMI': np.nan,
                    'FMI': np.nan,
                    '聚類標籤': np.zeros(len(X_scaled))
                })
            
            progress_bar.progress((i + 1) / len(selected_algorithms))
        
        status_text.text('聚類比較完成！')
        
        # 結果DataFrame
        results_df = pd.DataFrame(results)
        
        # 顯示性能比較表
        st.markdown("### 📊 算法性能比較表")
        
        # 創建展示用的表格（排除聚類標籤列）
        display_df = results_df.drop('聚類標籤', axis=1).copy()
        
        # 格式化數值列
        numeric_cols = ['輪廓係數', 'CH指數', 'DB指數', 'ARI', 'NMI', 'FMI']
        for col in numeric_cols:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}" if not np.isnan(x) else "N/A")
        
        st.dataframe(display_df, use_container_width=True)
        
        # 可視化比較
        st.markdown("### 📈 算法性能視覺化比較")
        
        # 只選擇數值有效的算法進行可視化
        valid_results = results_df.dropna(subset=['輪廓係數', 'ARI'])
        
        if len(valid_results) > 0:
            # 創建2x3的圖表布局，顯示6個評價指標
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # 輪廓係數比較
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=valid_results['算法'],
                    y=valid_results['輪廓係數'],
                    name='輪廓係數',
                    marker_color='blue'
                ))
                fig.update_layout(
                    title="輪廓係數 (越高越好)",
                    xaxis_title="算法",
                    yaxis_title="輪廓係數",
                    xaxis_tickangle=45,
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # ARI比較
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=valid_results['算法'],
                    y=valid_results['ARI'],
                    name='ARI',
                    marker_color='green'
                ))
                fig.update_layout(
                    title="調整蘭德指數 (越高越好)",
                    xaxis_title="算法",
                    yaxis_title="ARI",
                    xaxis_tickangle=45,
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # CH指數比較
                valid_ch = results_df.dropna(subset=['CH指數'])
                if len(valid_ch) > 0:
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=valid_ch['算法'],
                        y=valid_ch['CH指數'],
                        name='CH指數',
                        marker_color='orange'
                    ))
                    fig.update_layout(
                        title="CH指數 (越高越好)",
                        xaxis_title="算法",
                        yaxis_title="CH指數",
                        xaxis_tickangle=45,
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # NMI比較
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=valid_results['算法'],
                    y=valid_results['NMI'],
                    name='NMI',
                    marker_color='purple'
                ))
                fig.update_layout(
                    title="標準化互信息 (越高越好)",
                    xaxis_title="算法",
                    yaxis_title="NMI",
                    xaxis_tickangle=45,
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col3:
                # DB指數比較（越低越好，所以顯示1-DB）
                valid_db = results_df.dropna(subset=['DB指數'])
                if len(valid_db) > 0:
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=valid_db['算法'],
                        y=1 - valid_db['DB指數'],
                        name='1-DB指數',
                        marker_color='red'
                    ))
                    fig.update_layout(
                        title="1-DB指數 (越高越好)",
                        xaxis_title="算法",
                        yaxis_title="1-DB指數",
                        xaxis_tickangle=45,
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # FMI比較
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=valid_results['算法'],
                    y=valid_results['FMI'],
                    name='FMI',
                    marker_color='cyan'
                ))
                fig.update_layout(
                    title="Fowlkes-Mallows指數 (越高越好)",
                    xaxis_title="算法",
                    yaxis_title="FMI",
                    xaxis_tickangle=45,
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # 聚類結果可視化
        if len(selected_features) >= 2:
            st.markdown("### 📈 聚類結果可視化對比")
            
            # 計算需要的行數和列數，6個算法用3x2布局
            n_algorithms = len(results)
            if n_algorithms <= 4:
                n_cols = 2
                n_rows = 2
            elif n_algorithms <= 6:
                n_cols = 3
                n_rows = 2
            else:
                n_cols = 3
                n_rows = (n_algorithms + n_cols - 1) // n_cols
            
            # 創建子圖
            fig = make_subplots(
                rows=n_rows, cols=n_cols,
                subplot_titles=[r['算法'] for r in results],
                specs=[[{"type": "scatter"}] * n_cols for _ in range(n_rows)],
                horizontal_spacing=0.08,
                vertical_spacing=0.12
            )
            
            for i, result in enumerate(results):
                row = i // n_cols + 1
                col = i % n_cols + 1
                
                # 統一使用相同的離散顏色配色：紅 藍 綠 黃 紫 橙
                unified_colors = ['#FF4444', '#4169E1', '#32CD32', '#FFD700', '#BA55D3', '#FF8C00', '#00CED1', '#FF1493']
                
                # 為每個聚類標籤分配顏色
                cluster_labels = result['聚類標籤']
                unique_labels = np.unique(cluster_labels)
                color_map = {label: unified_colors[i % len(unified_colors)] for i, label in enumerate(unique_labels)}
                point_colors = [color_map[label] for label in cluster_labels]
                
                fig.add_trace(
                    go.Scatter(
                        x=X_selected.iloc[:, 0],
                        y=X_selected.iloc[:, 1],
                        mode='markers',
                        marker=dict(
                            color=point_colors,
                            size=5,
                            opacity=0.8
                        ),
                        name=result['算法'],
                        showlegend=False
                    ),
                    row=row, col=col
                )
            
            # 調整子圖標題字體大小和加粗
            try:
                for annotation in fig.layout.annotations:
                    annotation.font.size = 16
                    annotation.font.color = '#2C3E50'  # 深藍灰色
                    # 設置粗體字（如果支持的話）
                    try:
                        annotation.font.family = "Arial, sans-serif"
                    except:
                        pass
            except:
                pass  # 某些Plotly版本可能不支持此功能
            
            fig.update_layout(
                height=400*n_rows, 
                showlegend=False
            )
            
            # 更新軸標籤
            for i in range(1, n_rows + 1):
                for j in range(1, n_cols + 1):
                    fig.update_xaxes(title_text=selected_features[0], row=i, col=j, title_font_size=12)
                    fig.update_yaxes(title_text=selected_features[1], row=i, col=j, title_font_size=12)
            
            st.plotly_chart(fig, use_container_width=True)
        
        # 算法推薦
        st.markdown("### 🎯 算法選擇建議")
        
        # 找到最佳算法
        valid_results = results_df.dropna(subset=['ARI'])
        if len(valid_results) > 0:
            best_ari_algo = valid_results.loc[valid_results['ARI'].idxmax(), '算法']
            best_ari_score = valid_results.loc[valid_results['ARI'].idxmax(), 'ARI']
            
            valid_sil = results_df.dropna(subset=['輪廓係數'])
            if len(valid_sil) > 0:
                best_sil_algo = valid_sil.loc[valid_sil['輪廓係數'].idxmax(), '算法']
                best_sil_score = valid_sil.loc[valid_sil['輪廓係數'].idxmax(), '輪廓係數']
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.success(f"### 🏆 最佳外部指標 (與真實標籤最匹配)")
                st.markdown(f"""
                - **算法**: {best_ari_algo}
                - **ARI分數**: {best_ari_score:.4f}
                - **真實聚類數**: {len(np.unique(true_labels))}
                """)
            
            with col2:
                if len(valid_sil) > 0:
                    st.info(f"### 🥇 最佳內部指標 (聚類質量最高)")
                    st.markdown(f"""
                    - **算法**: {best_sil_algo}
                    - **輪廓係數**: {best_sil_score:.4f}
                    - **推薦場景**: 無真實標籤時的最佳選擇
                    """)
        
        # 算法特點總結
        st.markdown("### 📚 算法特點總結")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### 🎯 適合球形聚類")
            st.markdown("""
            - **K-Means**: 經典、快速、需要預設k
            - **GMM**: 軟聚類、概率輸出、處理重疊
            - **Birch**: 大數據集友好、增量學習
            """)
        
        with col2:
            st.markdown("#### 🌊 適合任意形狀")
            st.markdown("""
            - **DBSCAN**: 密度聚類、檢測噪聲、無需預設k
            - **Spectral**: 圖論方法、處理非凸聚類
            - **Agglomerative**: 層次結構、樹狀視圖
            """)
        
        with col3:
            st.markdown("#### 🔧 選擇建議")
            st.markdown(f"""
            - **當前數據**: {dataset_choice}
            - **樣本數**: {len(X_selected)}
            - **特徵數**: {len(selected_features)}
            - **真實聚類數**: {len(np.unique(true_labels))}
            """)
    
    else:
        st.warning("⚠️ 請至少選擇2個特徵和1個算法進行比較實驗。")

elif page == "🌳 層次聚類":
    st.markdown('<h1 class="main-header">🌳 層次聚類</h1>', unsafe_allow_html=True)
    
    st.markdown("## 🧮 算法原理")
    
    st.markdown("### 📐 層次聚類概念")
    st.markdown("""
    層次聚類(Hierarchical Clustering)構建樹狀的聚類結構：
    
    1. **凝聚層次聚類(Agglomerative)**：自底向上，從每個點開始逐步合併
    2. **分裂層次聚類(Divisive)**：自頂向下，從所有點開始逐步分裂
    3. **樹狀圖(Dendrogram)**：可視化聚類的層次結構
    4. **連接準則(Linkage)**：決定如何計算聚類間距離
    """)
    
    st.markdown("### ⚖️ 模型優缺點")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.success("### ✅ 優點")
        st.markdown("""
        - 🌳 **層次結構**：提供完整的聚類層次視圖
        - 🎯 **無需預設k值**：可從樹狀圖選擇聚類數
        - 🔄 **確定性結果**：相同數據總產生相同結果
        - 📊 **可解釋性強**：樹狀圖直觀易懂
        - 🎛️ **多種連接方式**：適應不同聚類需求
        """)
    
    with col2:
        st.error("### ❌ 缺點")
        st.markdown("""
        - ⏰ **計算複雜度高**：O(n³)時間複雜度
        - 🔄 **難以處理大數據**：內存和時間開銷大
        - 🎯 **對離群值敏感**：異常值影響聚類結構
        - 📐 **難以處理非球形**：傾向於球形聚類
        """)
    
    st.markdown("## 🎛️ 互動式實驗")
    
    X, true_labels = get_current_data()
    
    # 參數設置
    col1, col2 = st.columns(2)
    
    with col1:
        n_clusters = st.slider("聚類數量：", 2, 10, len(np.unique(true_labels)), key="agg_n_clusters")
        linkage = st.selectbox("連接準則：", ["ward", "complete", "average", "single"], key="agg_linkage")
        
    with col2:
        affinity = st.selectbox("親和度度量：", ["euclidean", "manhattan", "cosine"], key="agg_affinity")
        
        # 特徵選擇
        if len(X.columns) > 2:
            selected_features = st.multiselect(
                "選擇用於聚類的特徵：",
                X.columns.tolist(),
                default=X.columns.tolist()[:2],
                key="agg_selected_features"
            )
        else:
            selected_features = X.columns.tolist()
    
    if len(selected_features) >= 2:
        X_selected = X[selected_features]
        
        # 標準化選項
        normalize = st.checkbox("是否進行特徵標準化？", value=True,
                               help="層次聚類基於距離矩陣，特徵尺度差異會影響連接準則，建議標準化",
                               key="agg_normalize")
        
        if normalize:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_selected)
        else:
            X_scaled = X_selected.values
        
        # 執行層次聚類
        try:
            agg_clustering = AgglomerativeClustering(
                n_clusters=n_clusters,
                linkage=linkage or "ward",
                metric=affinity or "euclidean" if linkage != "ward" else "euclidean"
            )
            cluster_labels = agg_clustering.fit_predict(X_scaled)
            
            # 計算評價指標
            silhouette_avg = silhouette_score(X_scaled, cluster_labels)
            ch_score = calinski_harabasz_score(X_scaled, cluster_labels)
            db_score = davies_bouldin_score(X_scaled, cluster_labels)
            
            # 外部指標
            ari_score = adjusted_rand_score(true_labels, cluster_labels)
            nmi_score = normalized_mutual_info_score(true_labels, cluster_labels)
            fmi_score = fowlkes_mallows_score(true_labels, cluster_labels)
            
            # 顯示結果
            st.markdown("### 📊 聚類結果")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("輪廓係數", f"{silhouette_avg:.4f}")
                st.metric("CH指數", f"{ch_score:.2f}")
            
            with col2:
                st.metric("DB指數", f"{db_score:.4f}")
                st.metric("聚類數量", f"{n_clusters}")
            
            with col3:
                st.metric("調整蘭德指數", f"{ari_score:.4f}")
                st.metric("標準化互信息", f"{nmi_score:.4f}")
            
            # 可視化結果
            st.markdown("### 📈 聚類結果可視化")
            
            if len(selected_features) >= 2:
                col1, col2 = st.columns(2)
                
                with col1:
                    # 層次聚類結果 - 使用plasma色彩映射
                    fig = px.scatter(
                        x=X_selected.iloc[:, 0], 
                        y=X_selected.iloc[:, 1],
                        color=cluster_labels,  # 使用數值，配合連續色彩映射
                        title=f"層次聚類結果 ({n_clusters}個聚類, {linkage} linkage)",
                        labels={'color': '聚類標籤'},
                        color_continuous_scale='plasma'
                    )
                    fig.update_layout(height=500)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # 真實標籤 - 使用viridis色彩映射
                    fig = px.scatter(
                        x=X_selected.iloc[:, 0], 
                        y=X_selected.iloc[:, 1],
                        color=true_labels,  # 使用數值，配合連續色彩映射
                        title="真實標籤分布",
                        labels={'color': '真實標籤'},
                        color_continuous_scale='viridis'
                    )
                    fig.update_layout(height=500)
                    st.plotly_chart(fig, use_container_width=True)
            
            # 連接準則說明
            st.markdown("### 🔗 連接準則解釋")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if linkage == "ward":
                    st.info("""
                    **Ward連接**：
                    - 最小化聚類內方差
                    - 傾向於產生相似大小的聚類
                    - 適合球形、相似大小的聚類
                    - 只能使用歐氏距離
                    """)
                elif linkage == "complete":
                    st.info("""
                    **Complete連接**：
                    - 使用兩聚類間最遠點距離
                    - 傾向於產生緊密、球形的聚類
                    - 對離群值敏感
                    - 適合密集、分離良好的聚類
                    """)
                elif linkage == "average":
                    st.info("""
                    **Average連接**：
                    - 使用兩聚類間所有點對的平均距離
                    - 在complete和single之間的折衷
                    - 相對穩定的聚類
                    - 適合大多數聚類問題
                    """)
                else:  # single
                    st.info("""
                    **Single連接**：
                    - 使用兩聚類間最近點距離
                    - 容易產生鏈式聚類
                    - 對噪聲敏感
                    - 適合細長、不規則形狀的聚類
                    """)
            
            with col2:
                st.markdown("#### 📊 當前設置評估")
                
                # 簡單的性能評估
                if silhouette_avg > 0.7:
                    st.success("✅ 聚類效果優秀")
                elif silhouette_avg > 0.5:
                    st.info("ℹ️ 聚類效果良好")
                else:
                    st.warning("⚠️ 可能需要調整參數")
                
                st.markdown(f"""
                **參數組合評估**：
                - 連接準則：{linkage}
                - 親和度：{affinity}
                - 聚類數：{n_clusters}
                - 數據維度：{len(selected_features)}D
                """)
        
        except Exception as e:
            st.error(f"聚類執行失敗：{str(e)}")
            st.info("提示：ward連接只能使用euclidean距離")
    
    else:
        st.warning("⚠️ 請至少選擇2個特徵進行聚類實驗。")

elif page == "🎲 高斯混合模型":
    st.markdown('<h1 class="main-header">🎲 高斯混合模型</h1>', unsafe_allow_html=True)
    
    st.markdown("## 🧮 算法原理")
    
    st.markdown("### 📐 高斯混合模型概念")
    st.markdown("""
    高斯混合模型(Gaussian Mixture Model, GMM)是概率聚類算法：
    
    1. **概率模型**：每個聚類都是一個高斯分布
    2. **軟聚類**：每個點屬於各個聚類的概率
    3. **EM算法**：期望最大化算法進行參數估計
    4. **混合權重**：每個高斯成分的權重
    """)
    
    st.latex(r'''
    p(x) = \sum_{k=1}^{K} \pi_k \mathcal{N}(x|\mu_k, \Sigma_k)
    ''')
    
    st.markdown("### 🔤 公式變數解釋")
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("""
        **主要變數含義**：
        - **p(x)**: 數據點x的概率密度
        - **K**: 高斯混合成分的數量
        - **πₖ**: 第k個成分的混合權重（0≤πₖ≤1，∑πₖ=1）
        - **𝒩(x|μₖ, Σₖ)**: 第k個高斯分布
        """)
    
    with col2:
        st.success("""
        **高斯分布參數**：
        - **μₖ**: 第k個高斯分布的均值向量（中心位置）
        - **Σₖ**: 第k個高斯分布的協方差矩陣（形狀和方向）
        - **∑**: 對所有K個成分求和
        - **意義**: 每個點由多個高斯分布的加權組合生成
        """)
    
    st.markdown("### ⚖️ 模型優缺點")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.success("### ✅ 優點")
        st.markdown("""
        - 🎲 **軟聚類**：提供概率歸屬度
        - 📊 **橢圓形聚類**：可以處理橢圓形聚類
        - 🔄 **協方差建模**：考慮特徵間相關性  
        - 📏 **概率解釋**：結果具有概率意義
        - 🎯 **生成模型**：可以生成新數據
        """)
    
    with col2:
        st.error("### ❌ 缺點")
        st.markdown("""
        - 🎛️ **需要預設k值**：需要事先知道聚類數
        - 🔄 **對初始化敏感**：可能收斂到局部最優
        - 📐 **假設高斯分布**：數據需要符合高斯假設
        - ⚡ **計算複雜**：比K-means慢
        """)
    
    st.markdown("## 🎛️ 互動式實驗")
    
    X, true_labels = get_current_data()
    
    # 參數設置
    col1, col2 = st.columns(2)
    
    with col1:
        n_components = st.slider("混合成分數量：", 2, 10, len(np.unique(true_labels)))
        covariance_type = st.selectbox("協方差類型：", ["full", "tied", "diag", "spherical"])
        max_iter = st.slider("最大迭代次數：", 50, 500, 100)
    
    with col2:
        init_params = st.selectbox("初始化方法：", ["kmeans", "random"])
        random_state = st.slider("隨機種子：", 1, 100, 42)
        
        # 特徵選擇
        if len(X.columns) > 2:
            selected_features = st.multiselect(
                "選擇用於聚類的特徵：",
                X.columns.tolist(),
                default=X.columns.tolist()[:2]
            )
        else:
            selected_features = X.columns.tolist()
    
    if len(selected_features) >= 2:
        X_selected = X[selected_features]
        
        # 標準化選項
        normalize = st.checkbox("是否進行特徵標準化？", value=True,
                               help="GMM假設高斯分布，特徵尺度會影響協方差估計和EM收斂，建議標準化")
        
        if normalize:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_selected)
        else:
            X_scaled = X_selected.values
        
        # 執行GMM聚類
        try:
            gmm = GaussianMixture(
                n_components=n_components,
                covariance_type=covariance_type or "full",
                max_iter=max_iter,
                init_params=init_params or "kmeans",
                random_state=random_state
            )
            
            cluster_labels = gmm.fit_predict(X_scaled)
            probabilities = gmm.predict_proba(X_scaled)
            
            # 計算評價指標
            silhouette_avg = silhouette_score(X_scaled, cluster_labels)
            ch_score = calinski_harabasz_score(X_scaled, cluster_labels)
            db_score = davies_bouldin_score(X_scaled, cluster_labels)
            
            # 外部指標
            ari_score = adjusted_rand_score(true_labels, cluster_labels)
            nmi_score = normalized_mutual_info_score(true_labels, cluster_labels)
            fmi_score = fowlkes_mallows_score(true_labels, cluster_labels)
            
            # 顯示結果
            st.markdown("### 📊 聚類結果")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("輪廓係數", f"{silhouette_avg:.4f}")
                st.metric("CH指數", f"{ch_score:.2f}")
                st.metric("DB指數", f"{db_score:.4f}")
            
            with col2:
                st.metric("混合成分數", f"{n_components}")
                st.metric("迭代次數", f"{gmm.n_iter_}")
                st.metric("收斂狀態", "✅ 收斂" if gmm.converged_ else "❌ 未收斂")
            
            with col3:
                st.metric("調整蘭德指數", f"{ari_score:.4f}")
                st.metric("標準化互信息", f"{nmi_score:.4f}")
                st.metric("對數似然", f"{gmm.score(X_scaled):.2f}")
            
            # 可視化結果
            st.markdown("### 📈 聚類結果可視化")
            
            if len(selected_features) >= 2:
                col1, col2 = st.columns(2)
                
                with col1:
                    # GMM聚類結果（硬聚類）- 使用plasma色彩映射
                    fig = px.scatter(
                        x=X_selected.iloc[:, 0], 
                        y=X_selected.iloc[:, 1],
                        color=cluster_labels,  # 使用數值，配合連續色彩映射
                        title=f"GMM聚類結果 ({n_components}個聚類, {n_components}個均值中心)",
                        labels={'color': '聚類標籤'},
                        color_continuous_scale='plasma'
                    )
                    fig.update_layout(height=500)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # 概率軟聚類（顯示最大概率）
                    max_prob = np.max(probabilities, axis=1)
                    fig = px.scatter(
                        x=X_selected.iloc[:, 0], 
                        y=X_selected.iloc[:, 1],
                        color=max_prob,
                        title="軟聚類（歸屬概率）",
                        labels={'color': '最大歸屬概率'},
                        color_continuous_scale='viridis'
                    )
                    fig.update_layout(height=500)
                    st.plotly_chart(fig, use_container_width=True)
            
            # 概率分析
            st.markdown("### 🎲 概率分析")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📊 聚類歸屬概率分布")
                
                # 計算平均歸屬概率
                avg_prob = np.mean(np.max(probabilities, axis=1))
                uncertain_points = np.sum(np.max(probabilities, axis=1) < 0.7)
                
                st.info(f"""
                - **平均最大概率**: {avg_prob:.3f}
                - **不確定點數量**: {uncertain_points} ({uncertain_points/len(X_scaled)*100:.1f}%)
                - **確定性評估**: {'高' if avg_prob > 0.8 else '中' if avg_prob > 0.6 else '低'}
                """)
                
                # 概率分布直方圖
                fig = px.histogram(
                    x=np.max(probabilities, axis=1),
                    nbins=20,
                    title="最大歸屬概率分布",
                    labels={'x': '最大歸屬概率', 'y': '樣本數量'}
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("#### 🔍 協方差類型解釋")
                
                if covariance_type == "full":
                    st.info("""
                    **Full協方差**：
                    - 每個成分有獨立的協方差矩陣
                    - 可以產生任意方向的橢圓
                    - 參數最多，最靈活
                    - 需要較多數據避免過擬合
                    """)
                elif covariance_type == "tied":
                    st.info("""
                    **Tied協方差**：
                    - 所有成分共享同一個協方差矩陣
                    - 所有聚類有相同的形狀和方向
                    - 參數適中
                    - 適合相似形狀的聚類
                    """)
                elif covariance_type == "diag":
                    st.info("""
                    **Diagonal協方差**：
                    - 協方差矩陣為對角矩陣
                    - 橢圓軸與座標軸平行
                    - 參數較少
                    - 假設特徵間獨立
                    """)
                else:  # spherical
                    st.info("""
                    **Spherical協方差**：
                    - 所有方向方差相等
                    - 產生圓形聚類
                    - 參數最少
                    - 類似K-means假設
                    """)
            
            # 混合權重分析
            st.markdown("### ⚖️ 混合權重分析")
            
            weights = gmm.weights_
            weight_df = pd.DataFrame({
                '成分': [f'成分 {i}' for i in range(n_components)],
                '權重': weights,
                '樣本比例': [np.sum(cluster_labels == i)/len(cluster_labels) for i in range(n_components)]
            })
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.bar(
                    weight_df, 
                    x='成分', 
                    y='權重',
                    title="混合權重分布"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.bar(
                    weight_df, 
                    x='成分', 
                    y='樣本比例',
                    title="實際樣本分布"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        except Exception as e:
            st.error(f"GMM聚類執行失敗：{str(e)}")
    
    else:
        st.warning("⚠️ 請至少選擇2個特徵進行聚類實驗。")

elif page == "🕸️ 譜聚類":
    st.markdown('<h1 class="main-header">🕸️ 譜聚類</h1>', unsafe_allow_html=True)
    
    st.markdown("## 🧮 算法原理")
    
    st.markdown("### 📐 譜聚類概念")
    st.markdown("""
    譜聚類(Spectral Clustering)基於圖論和線性代數：
    
    1. **圖構建**：將數據點視為圖中的節點，計算相似度矩陣
    2. **拉普拉斯矩陣**：構建圖的拉普拉斯矩陣
    3. **特徵分解**：對拉普拉斯矩陣進行特徵分解
    4. **K-means聚類**：在特徵空間中應用K-means
    """)
    
    st.latex(r'''
    L = D - W, \quad L_{norm} = D^{-1/2}LD^{-1/2}
    ''')
    
    st.markdown("### 🔤 公式變數解釋")
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("""
        **拉普拉斯矩陣變數**：
        - **L**: 拉普拉斯矩陣（Laplacian Matrix）
        - **D**: 度矩陣（Degree Matrix），對角矩陣
        - **W**: 相似度矩陣（Weight/Similarity Matrix）
        - **Lₙₒᵣₘ**: 標準化拉普拉斯矩陣
        """)
    
    with col2:
        st.success("""
        **矩陣含義解釋**：
        - **W[i,j]**: 點i和點j之間的相似度（如RBF核值）
        - **D[i,i]**: 點i的度數（所有相似度之和）
        - **D⁻¹/²**: 度矩陣的-1/2次方
        - **目標**: 通過特徵分解找到數據的低維嵌入
        """)
    
    st.markdown("### ⚖️ 模型優缺點")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.success("### ✅ 優點")
        st.markdown("""
        - 🌊 **非凸聚類**：能處理任意形狀的聚類
        - 🕸️ **圖論基礎**：基於數據點間的相似性
        - 📊 **理論保證**：有良好的數學理論支撐
        - 🎯 **適合複雜結構**：處理複雜的數據結構
        - 🔄 **降維效果**：同時實現降維和聚類
        """)
    
    with col2:
        st.error("### ❌ 缺點")
        st.markdown("""
        - 🎛️ **需要預設k值**：需要事先知道聚類數
        - ⚡ **計算開銷大**：特徵分解計算複雜
        - 📏 **參數敏感**：對相似度參數敏感
        - 💾 **內存需求高**：需要存儲相似度矩陣
        """)
    
    st.markdown("## 🎛️ 互動式實驗")
    
    X, true_labels = get_current_data()
    
    # 參數設置
    col1, col2 = st.columns(2)
    
    with col1:
        n_clusters = st.slider("聚類數量：", 2, 10, len(np.unique(true_labels)))
        gamma = st.slider("RBF核參數 (γ)：", 0.1, 10.0, 1.0, 0.1)
        
    with col2:
        affinity = st.selectbox("親和度類型：", ["rbf", "nearest_neighbors", "precomputed"])
        n_neighbors = st.slider("近鄰數量 (僅nearest_neighbors)：", 5, 20, 10)
        
        # 特徵選擇
        if len(X.columns) > 2:
            selected_features = st.multiselect(
                "選擇用於聚類的特徵：",
                X.columns.tolist(),
                default=X.columns.tolist()[:2]
            )
        else:
            selected_features = X.columns.tolist()
    
    if len(selected_features) >= 2:
        X_selected = X[selected_features]
        
        # 標準化選項
        normalize = st.checkbox("是否進行特徵標準化？", value=True,
                               help="譜聚類基於相似度矩陣，特徵尺度會影響RBF核函數計算，建議標準化")
        
        if normalize:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_selected)
        else:
            X_scaled = X_selected.values
        
        # 執行譜聚類
        try:
            if affinity == "rbf":
                spectral = SpectralClustering(
                    n_clusters=n_clusters,
                    gamma=gamma,
                    affinity="rbf",
                    random_state=42
                )
            elif affinity == "nearest_neighbors":
                spectral = SpectralClustering(
                    n_clusters=n_clusters,
                    affinity="nearest_neighbors",
                    n_neighbors=n_neighbors,
                    random_state=42
                )
            else:
                # 預計算親和度矩陣
                from sklearn.metrics.pairwise import rbf_kernel
                affinity_matrix = rbf_kernel(X_scaled, gamma=gamma)
                spectral = SpectralClustering(
                    n_clusters=n_clusters,
                    affinity="precomputed",
                    random_state=42
                )
                cluster_labels = spectral.fit_predict(affinity_matrix)
            
            if affinity != "precomputed":
                cluster_labels = spectral.fit_predict(X_scaled)
            
            # 計算評價指標
            silhouette_avg = silhouette_score(X_scaled, cluster_labels)
            ch_score = calinski_harabasz_score(X_scaled, cluster_labels)
            db_score = davies_bouldin_score(X_scaled, cluster_labels)
            
            # 外部指標
            ari_score = adjusted_rand_score(true_labels, cluster_labels)
            nmi_score = normalized_mutual_info_score(true_labels, cluster_labels)
            fmi_score = fowlkes_mallows_score(true_labels, cluster_labels)
            
            # 顯示結果
            st.markdown("### 📊 聚類結果")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("輪廓係數", f"{silhouette_avg:.4f}")
                st.metric("CH指數", f"{ch_score:.2f}")
            
            with col2:
                st.metric("DB指數", f"{db_score:.4f}")
                st.metric("聚類數量", f"{n_clusters}")
            
            with col3:
                st.metric("調整蘭德指數", f"{ari_score:.4f}")
                st.metric("標準化互信息", f"{nmi_score:.4f}")
            
            # 可視化結果
            st.markdown("### 📈 聚類結果可視化")
            
            if len(selected_features) >= 2:
                col1, col2 = st.columns(2)
                
                with col1:
                    # 譜聚類結果 - 使用plasma色彩映射
                    fig = px.scatter(
                        x=X_selected.iloc[:, 0], 
                        y=X_selected.iloc[:, 1],
                        color=cluster_labels,  # 使用數值，配合連續色彩映射
                        title=f"譜聚類結果 ({n_clusters}個聚類, {affinity})",
                        labels={'color': '聚類標籤'},
                        color_continuous_scale='plasma'
                    )
                    fig.update_layout(height=500)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # 真實標籤 - 使用更明顯的顏色
                    fig = px.scatter(
                        x=X_selected.iloc[:, 0], 
                        y=X_selected.iloc[:, 1],
                        color=[f"類別 {i}" for i in true_labels],  # 轉換為字符串，讓plotly識別為離散變量
                        title="真實標籤分布",
                        labels={'color': '真實標籤'},
                        color_discrete_sequence=['#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6', '#1ABC9C', '#34495E', '#E67E22']
                    )
                    fig.update_layout(height=500)
                    st.plotly_chart(fig, use_container_width=True)
            
            # 親和度類型說明
            st.markdown("### 🔗 親和度類型解釋")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if affinity == "rbf":
                    st.info(f"""
                    **RBF親和度**：
                    - 使用高斯核函數: $exp(-\\gamma ||x_i - x_j||^2)$
                    - γ參數控制核的寬度
                    - γ越大，相似度越局部化
                    - 當前γ值: {gamma}
                    """)
                elif affinity == "nearest_neighbors":
                    st.info(f"""
                    **近鄰親和度**：
                    - 基於k近鄰構建圖
                    - 只考慮最近的k個鄰居
                    - 產生稀疏的親和度矩陣
                    - 當前k值: {n_neighbors}
                    """)
                else:
                    st.info("""
                    **預計算親和度**：
                    - 使用預先計算的親和度矩陣
                    - 可以使用自定義相似度函數
                    - 提供最大的靈活性
                    - 需要更多內存
                    """)
            
            with col2:
                st.markdown("#### 📊 參數調優建議")
                
                if affinity == "rbf":
                    if gamma < 1:
                        st.info("💡 γ值較小：相似度較全局，可能產生大聚類")
                    elif gamma > 5:
                        st.warning("⚠️ γ值較大：相似度過於局部，可能過擬合")
                    else:
                        st.success("✅ γ值適中：平衡局部和全局相似度")
                
                elif affinity == "nearest_neighbors":
                    if n_neighbors < 8:
                        st.info("💡 k值較小：圖較稀疏，適合明顯分離的聚類")
                    elif n_neighbors > 15:
                        st.warning("⚠️ k值較大：圖較密集，可能連接不同聚類")
                    else:
                        st.success("✅ k值適中：平衡稀疏性和連通性")
            
            # 譜聚類特點分析
            st.markdown("### 🎯 譜聚類適用場景")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("#### ✅ 適合的數據")
                st.markdown("""
                - 非凸聚類（月牙形、環形）
                - 流形數據
                - 圖結構數據
                - 複雜幾何結構
                """)
            
            with col2:
                st.markdown("#### ❌ 不適合的數據")
                st.markdown("""
                - 高維稀疏數據
                - 大規模數據集
                - 簡單的球形聚類
                - 噪聲很多的數據
                """)
            
            with col3:
                st.markdown("#### 🔧 參數選擇技巧")
                st.markdown("""
                - RBF: γ ∝ 1/特徵維度
                - k-NN: k ≈ log(n)
                - 先嘗試不同參數
                - 觀察聚類可視化結果
                """)
        
        except Exception as e:
            st.error(f"譜聚類執行失敗：{str(e)}")
            st.info("可能原因：數據量太大或參數設置問題")
    
    else:
        st.warning("⚠️ 請至少選擇2個特徵進行聚類實驗。")

elif page == "🌿 BIRCH聚類":
    st.markdown('<h1 class="main-header">🌿 BIRCH聚類</h1>', unsafe_allow_html=True)
    
    st.markdown("## 🧮 算法原理")
    
    st.markdown("### 📐 BIRCH算法概念")
    st.markdown("""
    BIRCH (Balanced Iterative Reducing and Clustering using Hierarchies) 是專為大數據集設計的增量聚類算法：
    
    1. **CF樹(Clustering Feature Tree)**：緊湊的樹狀數據結構
    2. **增量處理**：逐一掃描數據點，動態更新CF樹
    3. **兩階段聚類**：先構建CF樹，再對葉節點聚類
    4. **內存友好**：O(n)時間複雜度，適合大數據集
    """)
    
    st.markdown("### 🌳 聚類特徵(CF)向量")
    st.latex(r'''
    CF_i = (N_i, LS_i, SS_i)
    ''')
    st.markdown("### 🔤 公式變數解釋")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("""
        **聚類特徵(CF)向量**：
        - **CFᵢ**: 第i個聚類的特徵向量
        - **Nᵢ**: 聚類中數據點的數量
        - **LSᵢ**: 線性和（Linear Sum）= ∑X_j
        - **SSᵢ**: 平方和（Sum of Squares）= ∑X_j²
        """)
    
    with col2:
        st.success("""
        **CF向量的優勢**：
        - **緊湊性**: 用3個值概括整個聚類
        - **可加性**: CF₁ + CF₂ = 合併後的CF
        - **距離計算**: 可直接用CF計算聚類間距離
        - **增量更新**: 新點加入時可快速更新CF
        """)
    
    st.markdown("### ⚖️ 模型優缺點")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.success("### ✅ 優點")
        st.markdown("""
        - 🚀 **增量學習**：可以處理流式數據
        - 💾 **內存效率**：只需一次掃描數據
        - ⚡ **計算快速**：O(n)時間複雜度
        - 📊 **可擴展性**：適合大規模數據集
        - 🎯 **穩定性**：對數據順序不敏感
        """)
    
    with col2:
        st.error("### ❌ 缺點")
        st.markdown("""
        - 📐 **假設球形聚類**：不適合任意形狀
        - 🎛️ **參數敏感**：threshold參數需要調優
        - 🔄 **需要數值特徵**：不能直接處理類別特徵
        - 📏 **對密度敏感**：密度差異大時效果差
        """)
    
    st.markdown("## 🎛️ 互動式實驗")
    
    X, true_labels = get_current_data()
    
    # 參數設置
    col1, col2 = st.columns(2)
    
    with col1:
        n_clusters = st.slider("聚類數量：", 2, 10, len(np.unique(true_labels)))
        threshold = st.slider("閾值 (threshold)：", 0.1, 2.0, 0.5, 0.1)
        branching_factor = st.slider("分支因子：", 10, 100, 50, 10)
    
    with col2:
        # 特徵選擇
        if len(X.columns) > 2:
            selected_features = st.multiselect(
                "選擇用於聚類的特徵：",
                X.columns.tolist(),
                default=X.columns.tolist()[:2]
            )
        else:
            selected_features = X.columns.tolist()
        
        compute_labels = st.checkbox("計算最終聚類標籤", value=True)
    
    if len(selected_features) >= 2:
        X_selected = X[selected_features]
        
        # 標準化選項
        normalize = st.checkbox("是否進行特徵標準化？", value=True,
                               help="BIRCH使用歐氏距離構建CF樹，特徵尺度差異會影響聚類特徵計算，強烈建議標準化")
        
        if normalize:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_selected)
        else:
            X_scaled = X_selected.values
        
        # 執行BIRCH聚類
        try:
            if compute_labels:
                birch = Birch(
                    n_clusters=n_clusters,
                    threshold=threshold,
                    branching_factor=branching_factor
                )
            else:
                birch = Birch(
                    threshold=threshold,
                    branching_factor=branching_factor
                )
            
            cluster_labels = birch.fit_predict(X_scaled)
            
            # 統計CF樹信息
            n_cf_nodes = len(birch.subcluster_centers_)
            
            # 如果計算了聚類標籤，計算評價指標
            if compute_labels and len(np.unique(cluster_labels)) > 1:
                silhouette_avg = silhouette_score(X_scaled, cluster_labels)
                ch_score = calinski_harabasz_score(X_scaled, cluster_labels)
                db_score = davies_bouldin_score(X_scaled, cluster_labels)
                
                # 外部指標
                ari_score = adjusted_rand_score(true_labels, cluster_labels)
                nmi_score = normalized_mutual_info_score(true_labels, cluster_labels)
                fmi_score = fowlkes_mallows_score(true_labels, cluster_labels)
            else:
                silhouette_avg = ch_score = db_score = np.nan
                ari_score = nmi_score = fmi_score = np.nan
            
            # 顯示結果
            st.markdown("### 📊 聚類結果")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("CF子聚類數", f"{n_cf_nodes}")
                st.metric("最終聚類數", f"{len(np.unique(cluster_labels))}")
                st.metric("閾值參數", f"{threshold}")
            
            with col2:
                if not np.isnan(silhouette_avg):
                    st.metric("輪廓係數", f"{silhouette_avg:.4f}")
                    st.metric("CH指數", f"{ch_score:.2f}")
                    st.metric("DB指數", f"{db_score:.4f}")
                else:
                    st.metric("輪廓係數", "N/A")
                    st.metric("CH指數", "N/A")
                    st.metric("DB指數", "N/A")
            
            with col3:
                if not np.isnan(ari_score):
                    st.metric("調整蘭德指數", f"{ari_score:.4f}")
                    st.metric("標準化互信息", f"{nmi_score:.4f}")
                    st.metric("FMI指數", f"{fmi_score:.4f}")
                else:
                    st.metric("ARI", "N/A")
                    st.metric("NMI", "N/A")
                    st.metric("FMI", "N/A")
            
            # 可視化結果
            st.markdown("### 📈 聚類結果可視化")
            
            if len(selected_features) >= 2:
                col1, col2 = st.columns(2)
                
                with col1:
                    # BIRCH聚類結果 - 使用更明顯的顏色
                    if compute_labels:
                        fig = px.scatter(
                            x=X_selected.iloc[:, 0], 
                            y=X_selected.iloc[:, 1],
                            color=cluster_labels,  # 使用數值，配合連續色彩映射
                            title=f"BIRCH聚類結果 ({len(np.unique(cluster_labels))}個聚類, {n_cf_nodes}個CF中心)",
                            labels={'color': '聚類標籤'},
                            color_continuous_scale='plasma'
                        )
                        
                        # 顯示子聚類中心
                        if normalize:
                            centers_original = scaler.inverse_transform(birch.subcluster_centers_)
                        else:
                            centers_original = birch.subcluster_centers_
                        
                        fig.add_scatter(
                            x=centers_original[:, 0],
                            y=centers_original[:, 1],
                            mode='markers',
                            marker=dict(symbol='diamond', size=12, color='black', line=dict(width=2, color='white')),
                            name='CF子聚類中心'
                        )
                    else:
                        fig = px.scatter(
                            x=X_selected.iloc[:, 0], 
                            y=X_selected.iloc[:, 1],
                            title="BIRCH CF樹構建（未聚類）",
                            color_discrete_sequence=['blue']
                        )
                    
                    fig.update_layout(height=500)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # 真實標籤 - 使用更明顯的顏色
                    fig = px.scatter(
                        x=X_selected.iloc[:, 0], 
                        y=X_selected.iloc[:, 1],
                        color=true_labels,  # 使用數值，配合連續色彩映射
                        title="真實標籤分布",
                        labels={'color': '真實標籤'},
                        color_continuous_scale='viridis'
                    )
                    fig.update_layout(height=500)
                    st.plotly_chart(fig, use_container_width=True)
            
            # 參數調優指南
            st.markdown("### 🎯 參數調優指南")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 🔧 Threshold調優")
                
                if n_cf_nodes > len(X_selected) * 0.8:
                    st.warning(f"⚠️ CF節點過多({n_cf_nodes})：threshold可能過小")
                    st.markdown("**建議**: 增加threshold值，減少CF節點數")
                elif n_cf_nodes < 10:
                    st.info(f"ℹ️ CF節點較少({n_cf_nodes})：threshold可能過大")
                    st.markdown("**建議**: 減少threshold值，增加子聚類解析度")
                else:
                    st.success(f"✅ CF節點數適中({n_cf_nodes})")
                
                st.markdown(f"""
                **當前設置評估**：
                - Threshold: {threshold}
                - CF節點數: {n_cf_nodes}
                - 數據點數: {len(X_selected)}
                - CF節點比例: {n_cf_nodes/len(X_selected)*100:.1f}%
                """)
            
            with col2:
                st.markdown("#### 🌳 分支因子優化")
                
                st.info(f"""
                **分支因子 = {branching_factor}**
                
                - **作用**: 控制CF樹每個內部節點的最大子節點數
                - **較小值**: 樹更深，內存使用更少，但可能增加搜索時間
                - **較大值**: 樹更寬，搜索更快，但內存使用更多
                - **建議**: 50-100適合大多數情況
                """)
                
                if branching_factor < 30:
                    st.warning("⚠️ 分支因子較小，可能導致樹過深")
                elif branching_factor > 80:
                    st.info("ℹ️ 分支因子較大，注意內存使用")
                else:
                    st.success("✅ 分支因子設置合理")
            
            # BIRCH特點分析
            st.markdown("### 🚀 BIRCH算法特點")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("#### ✅ 適用場景")
                st.markdown("""
                - 大規模數據集聚類
                - 流式數據處理
                - 內存受限環境
                - 需要快速聚類的場景
                - 球形或準球形聚類
                """)
            
            with col2:
                st.markdown("#### ❌ 不適用場景")
                st.markdown("""
                - 任意形狀的聚類
                - 密度變化很大的數據
                - 維度詛咒嚴重的高維數據
                - 需要精確聚類邊界的場景
                """)
            
            with col3:
                st.markdown("#### 🔧 使用技巧")
                st.markdown("""
                - 先用小樣本調優threshold
                - 分支因子設為50-100
                - 數據標準化很重要
                - 可與其他算法結合使用
                """)
        
        except Exception as e:
            st.error(f"BIRCH聚類執行失敗：{str(e)}")
            st.info("提示：確保數據格式正確，參數設置合理")
    
    else:
        st.warning("⚠️ 請至少選擇2個特徵進行聚類實驗。")

else:
    st.markdown(f"# {page}")
    st.info("此頁面正在開發中...") 