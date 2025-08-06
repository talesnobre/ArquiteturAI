import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import r2_score
import plotly.express as px

st.set_page_config(page_title="Validador de Modelos", layout="wide")
st.title("📊 Validador de Modelos com R² e Visualização Interativa")

# === 1. Upload do CSV ===
arquivo = st.file_uploader("📁 Faça upload do CSV para validação", type=["csv", "xls", "xlsx"])

if arquivo is not None:
    try:
        df_val = pd.read_csv(arquivo, encoding="utf-8")
        st.success("✅ Arquivo carregado com sucesso!")

        st.write("📋 Pré-visualização dos dados:")
        st.dataframe(df_val.head())

        # === 2. Escolha do modelo e target ===
        modelo_opcao = st.selectbox("🔍 Escolha o tipo de modelo:", 
                                    ("Random Forest", "Decision Tree"))

        target_opcao = st.selectbox("🎯 Escolha o target (coluna alvo):",
                                    ("UDI ", "UDI_more", "UDI_less"))

        # Montar caminho do modelo conforme escolhas
        prefixo = "best_random_forest_model" if modelo_opcao == "Random Forest" else "best_decision_tree_model"
        modelo_path = f"{prefixo}.pkl" if target_opcao == "UDI " else f"{prefixo}_{target_opcao}.pkl"

        if target_opcao not in df_val.columns:
            st.error(f"❌ A coluna '{target_opcao}' não foi encontrada no arquivo.")
        else:
            # Remover colunas de target que não estão sendo previstas
            targets = ["UDI ", "UDI_more", "UDI_less"]
            targets_para_remover = [col for col in targets if col != target_opcao]
            X_val = df_val.drop(columns=targets_para_remover + [target_opcao])
            y_val = df_val[target_opcao]

            try:
                modelo = joblib.load(modelo_path)
            except FileNotFoundError:
                st.error(f"❌ Modelo '{modelo_path}' não encontrado.")
            else:
                y_pred = modelo.predict(X_val)
                r2 = r2_score(y_val, y_pred)

                st.subheader("📈 Resultado da Validação:")
                st.metric("R² Score", f"{r2:.4f}")

                df_resultado = X_val.copy()
                df_resultado["Real"] = y_val.values
                df_resultado["Previsto"] = y_pred

                st.subheader("📊 Comparativo Real vs Previsto")
                st.dataframe(df_resultado)

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

                csv_resultado = df_resultado.to_csv(index=False).encode("utf-8")
                st.download_button("📥 Baixar comparativo em CSV", csv_resultado, "resultado_comparativo.csv", "text/csv")

    except Exception as e:
        st.error(f"⚠️ Erro ao processar o arquivo: {e}")
