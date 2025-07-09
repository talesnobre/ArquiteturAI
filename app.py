import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import r2_score
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Validador de Modelos", layout="wide")
st.title("📊 Validador de Modelos com R² e Visualização Interativa")

# === 1. Escolha do modelo ===
modelo_opcao = st.selectbox("🔍 Escolha o modelo para validação:", 
                            ("Random Forest", "Decision Tree"))

modelo_path = {
    "Random Forest": "best_random_forest_model.pkl",
    "Decision Tree": "best_decision_tree_model.pkl"
}[modelo_opcao]

# === 2. Upload do CSV ===
arquivo = st.file_uploader("📁 Faça upload do CSV para validação", type=["csv", "xls", "xlsx"])

if arquivo is not None:
    try:
        df_val = pd.read_csv(arquivo, encoding="utf-8")
        st.success("✅ Arquivo carregado com sucesso!")

        st.write("📋 Pré-visualização dos dados:")
        st.dataframe(df_val.head())

        # === 3. Escolha da variável alvo (target) ===
        colunas = df_val.columns.tolist()
        coluna_alvo = st.selectbox("🎯 Selecione a coluna alvo (target):", colunas)

        if coluna_alvo:
            X_val = df_val.drop(columns=[coluna_alvo])
            y_val = df_val[coluna_alvo]

            try:
                modelo = joblib.load(modelo_path)
            except FileNotFoundError:
                st.error(f"❌ Modelo '{modelo_path}' não encontrado.")
            else:
                y_pred = modelo.predict(X_val)
                r2 = r2_score(y_val, y_pred)

                st.subheader("📈 Resultado da Validação:")
                st.metric("R² Score", f"{r2:.4f}")

                # === Tabela comparativa ===
                df_resultado = X_val.copy()
                df_resultado["Real"] = y_val.values
                df_resultado["Previsto"] = y_pred

                st.subheader("📊 Comparativo Real vs Previsto")
                st.dataframe(df_resultado)
                

                # fig = px.line(df_resultado, x='Index', y=['Real', 'Previsto'], color='variable', symbol='variable', title='Regressor', labels={ 'value': 'Valor','Index':'ID'})
                # st.plotly_chart(fig, use_container_width=True)
                
                # === Gráfico com Plotly ===
                st.subheader("📉 Gráfico de Dispersão (Plotly)")
                fig = px.scatter(
                    df_resultado,
                    x="Real",
                    y="Previsto",
                    title="Dispersão: Valor Real vs Previsto",
                    labels={"Real": "Valor Real", "Previsto": "Valor Previsto"},
                    opacity=0.7
                )
                fig.update_layout(width=800, height=500)
                st.plotly_chart(fig, use_container_width=True)

                # === Download do comparativo ===
                csv_resultado = df_resultado.to_csv(index=False).encode("utf-8")
                st.download_button("📥 Baixar comparativo em CSV", csv_resultado, "resultado_comparativo.csv", "text/csv")
    except Exception as e:
        st.error(f"⚠️ Erro ao processar o arquivo: {e}")
