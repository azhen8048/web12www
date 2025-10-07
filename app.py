import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from matplotlib import font_manager
import xgboost
from xgboost import XGBClassifier

# 加载保存的随机森林模型
model = joblib.load('xgb_model.pkl')

# 特征范围定义（根据提供的特征范围和数据类型）
feature_ranges = {
    "Grade_of_splenomegaly": {
        "type": "categorical",
        "options": [1, 2, 3],
        "default": 1
    },
    "Splenic_thickness": {
        "type": "numerical",
        "min": 0.0,
        "max": 20.0,
        "default": 5.0,
        "unit": "cm"
    },
    "POD_D_dimer": {
        "type": "numerical",
        "min": 0.0,
        "max": 50.0,
        "default": 14.0,
        "unit": "mg/L"
    },
    "POD_PLT": {
        "type": "numerical",
        "min": 0.0,
        "max": 2000.,
        "default": 450.0,
        "unit": "×10⁹/L"
    },
    "PVD": {
        "type": "numerical",
        "min": 0.0,
        "max": 30.0,
        "default": 14.0,
        "unit": "mm"
    },
    "PVV": {
        "type": "numerical",
        "min": 0.0,
        "max": 100.0,
        "default": 20.0,
        "unit": "cm/s"
    },
    "PVF": {
        "type": "numerical",
        "min": 0.0,
        "max": 5000.0,
        "default": 1000.0,
        "unit": "ml/min"
    },
    "SVD": {
        "type": "numerical",
        "min": 0.0,
        "max": 30.0,
        "default": 10.0,
        "unit": "mm"
    },
}

# Streamlit 界面
st.title("Prediction Model")

# 动态生成输入项
st.header("Enter the following feature values:")
feature_values = []
for feature, properties in feature_ranges.items():
    display_name = feature.replace('_', ' ')
    display_name = display_name.replace('POD D dimer', 'POD D-dimer')  
    if properties["type"] == "numerical":
        value = st.number_input(
            label=f"{display_name} ({properties['unit']}, {properties['min']}–{properties['max']})",
            min_value=float(properties["min"]),
            max_value=float(properties["max"]),
            value=float(properties["default"]),
        )
    elif properties["type"] == "categorical":
        value = st.selectbox(
            label=f"{display_name} (Select a value)",
            options=properties["options"],
        )
    feature_values.append(value)

# 转换为模型输入格式
features = np.array([feature_values])

# 预测与 SHAP 可视化
if st.button("Predict"):
    # 模型预测
    predicted_class = model.predict(features)[0]
    predicted_proba = model.predict_proba(features)[0]

    # 提取预测的类别概率
    probability = predicted_proba[predicted_class] * 100

    # 显示预测结果，使用 Matplotlib 渲染指定字体
    text = f"Based on feature values, predicted possibility of PVST is {probability:.2f}%"
    fig, ax = plt.subplots(figsize=(8, 1))

    # 设置Times New Roman斜体加粗
    try:
        prop = font_manager.FontProperties(
            family='Times New Roman',
            style='italic',
            weight='bold',
            size=16
        )
        ax.text(
            0.5, 0.5, text,
            fontproperties=prop,
            ha='center', va='center',
            transform=ax.transAxes
        )
    except:
        ax.text(
            0.5, 0.5, text,
            fontsize=16,
            ha='center', va='center',
            style='italic',
            weight='bold',
            family='serif',
            transform=ax.transAxes
        )

    ax.axis('off')
    plt.savefig("prediction_text.png", bbox_inches='tight', dpi=300, transparent=True)
    st.image("prediction_text.png")

        # ===== 计算 SHAP 值 =====
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(
        pd.DataFrame([feature_values], columns=feature_ranges.keys())
    )

    # ---- 统一成 2-D ----
    if isinstance(shap_values, list):                 
        shap_values = np.array(shap_values[1])        
    if shap_values.ndim == 3 and shap_values.shape[2] == 1:
        shap_values = shap_values[:, :, 0]           

    baseline = float(explainer.expected_value)       
    sv = shap_values[0]                              

    # ---- SHAP 力图 ----
    shap_fig = shap.force_plot(
        baseline,
        sv,
        pd.DataFrame([feature_values], columns=feature_ranges.keys()),
        matplotlib=True,
    )
    plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)
    st.image("shap_force_plot.png")