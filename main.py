import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import odeint
import joblib

# 保存は cntrol+S
#実行はstreamlit run main.py
# control + C でstreamlit 終了

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_squared_error
from math import sqrt



# ファイルのアップロード
uploaded_file = st.file_uploader("CSVファイルをアップロードしてください", type=["csv"])

# ファイルがアップロードされた場合
if uploaded_file is not None:
    # CSVファイルを読み込む
    new_df = pd.read_csv(uploaded_file)
    new_df1 = new_df.iloc[:, 1:]


    # テストデータの説明変数を準備
    X_test_features_only = new_df1.iloc[:, :]

    # 保存したモデルを読み込む
    loaded_model1 = joblib.load('best_model1_new.pkl')
    loaded_model2 = joblib.load('best_model2_new.pkl')
    loaded_model3 = joblib.load('best_model3_new.pkl')
    loaded_model4 = joblib.load('best_model4_new.pkl')

    # モデルに説明変数のみのテストデータを入力して予測
    y_pred_test_model1 = loaded_model1.predict(X_test_features_only)
    y_pred_test_model2 = loaded_model2.predict(X_test_features_only)
    y_pred_test_model3 = loaded_model3.predict(X_test_features_only)
    y_pred_test_model4 = loaded_model4.predict(X_test_features_only)




st.title('血中濃度予測アプリ')
#st.write("Hello")


Dosage = st.text_input("投与量を入れてください" + (" (mg/kg)"))


# DosageをDoseに変換してモデルに適用
try:
    Dose = float(Dosage)  # Dosageを浮動小数点数に変換
except ValueError:
    st.error("有効な数値を入力してください")
    st.stop()  # エラーメッセージを表示してアプリを停止

# テストデータを作成
data1 = [[1, 'chemical_structure']]

# DataFrameに格納
df_1 = pd.DataFrame(data1, columns=['Name', 'SMILES'])

# DataFrameの表示
print(df_1)


# 定数を設定する（固定値）
Qh = 96.6
Vh = 1.5
Qr = 96.6
Vr = 0.28
BW = 70
# パラメータの入力'Doseの単位はµg/kg' k, CLint, CLr, FaFg は機械学習で予測(後で)
#k = y_pred_test_model2 # model2で算出した値
k = float(y_pred_test_model2)
Rb = 0.777
Kph = 0.697855
Kpr = Kph 
#CLint = y_pred_test_model4 # model4で算出した値
CLint =float(y_pred_test_model4)
CLr = 0.0528
fup = 0.829
#V1 = y_pred_test_model3 # model3で算出した値
V1= float(y_pred_test_model3)
#FaFg =y_pred_test_model1 # model1で算出した値
FaFg= float(y_pred_test_model1)

# 初期条件を設定する
X0 = Dose * BW * FaFg
Ch0 = 0
Cb0 = 0
Cr0 = 0
# 4変数の連立微分方程式を定義する 　　　腸管、肝臓、全身、腎臓の4コンパートメントモデルで作る
def model(y, t):
    X, Ch, Cb, Cr = y
    dXdt = -k*X
    dChdt = (k*X - (Qh*Ch*Rb)/Kph - CLint*Ch/Kph*fup + Qh*Cb)/Vh
    dCbdt = (-(Qh+Qr)*Cb+Qh*Ch*Rb/Kph+Qr*Cr*Rb/Kpr)/V1
    dCrdt = (Qr*Cb - Qr*Cr*Rb/Kpr - CLr*Cr/Kpr*fup)/Vr
    return [dXdt, dChdt, dCbdt, dCrdt]

# 時間ステップを設定する
t0 = 0  # 初期時刻
t_final = 8  # 最終時刻
delta_t = 0.01 # Δt
t = np.arange(t0, t_final, delta_t)

# 初期値を設定して微分方程式を解く
y0 = [X0, Ch0, Cb0, Cr0]
sol = odeint(model, y0, t)

# Plasma_AUC と Liver_AUC の算出
Ch = sol[:, 0]
Cp = sol[:, 1]
indices = np.where((t>=t0)&(t<=t_final))[0]
AUC_liver = np.trapz(Ch[indices], t[indices])
AUC_plasma = np.trapz(Cp[indices],t[indices])

# Cbを血漿中濃度に変換
sol[:, 1]=sol[:, 1]/Rb

# Plasma_AUC と Liver_AUC の算出
Ch = sol[:, 0]
Cp = sol[:, 1]
indices = np.where((t>=t0)&(t<=t_final))[0]
AUC_liver = np.trapz(Ch[indices], t[indices])
AUC_plasma = np.trapz(Cp[indices],t[indices])

# 各コンパートメントの血中濃度をグラフ化
plt.plot(t, sol[:, 1], label='Cb(t)')
plt.xlabel('Time (h)')
plt.ylabel(' Plasma concentration µg/L')
plt.legend()
plt.show()

# グラフを表示
st.pyplot(plt)
# 各コンパートメントの血中濃度をグラフ化
st.line_chart(sol[:, 1], use_container_width=True)