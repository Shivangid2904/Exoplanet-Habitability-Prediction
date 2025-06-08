import shap
import matplotlib.pyplot as plt
import streamlit as st
import io
import pandas as pd
import numpy as np

def explain_prediction(input_data, model):
    # Ensure input is a DataFrame
    if not isinstance(input_data, pd.DataFrame):
        input_df = pd.DataFrame([input_data])
    else:
        input_df = input_data

    # Initialize SHAP explainer (for tree-based models)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_df)

    # Predict class
    predicted_class = model.predict(input_df)[0]

    # Handle SHAP values depending on their format
    if isinstance(shap_values, list):
        # For binary or multi-class classification
        shap_for_pred_class = shap_values[predicted_class][0]  # 1D array
        base_value = explainer.expected_value[predicted_class]
    elif isinstance(shap_values, np.ndarray):
        # This can happen in regression or binary classification (shape: (1, n_features, 2) or similar)
        if shap_values.ndim == 3:
            shap_for_pred_class = shap_values[0, :, predicted_class]
            base_value = explainer.expected_value[predicted_class]
        elif shap_values.ndim == 2:
            shap_for_pred_class = shap_values[0]
            base_value = explainer.expected_value
        else:
            raise ValueError("Unexpected shape of shap_values: " + str(shap_values.shape))
    else:
        raise ValueError("shap_values type not supported: " + str(type(shap_values)))
    print("SHAP values shape:", np.shape(shap_values))
    print("SHAP slice shape:", np.shape(shap_for_pred_class))

    # Build Explanation object
    explanation = shap.Explanation(
        values=shap_for_pred_class,
        base_values=base_value,
        data=input_df.iloc[0],
        feature_names=input_df.columns
    )

    # Create waterfall plot
    shap.plots.waterfall(explanation, show=False)
    fig = plt.gcf()

    # Save plot to buffer
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
    buf.seek(0)
    st.image(buf)

    # Clean up memory
    plt.close(fig)
