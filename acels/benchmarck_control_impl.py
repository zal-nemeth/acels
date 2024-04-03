from acels.benchmark_model_impl import compare_datasets

# -------------------------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------------------------
if __name__ == "__main__":
    model_id = "141"
    data_exists = False

    original_csv_path = f"acels/data/{model_id}_test_coordinates.csv"

    # Non-quantized results
    model_type_non_quant = "non_quant_impl"
    non_quant_pred = f"acels\\predictions\\{model_id}_non_quantized_predictions.csv"
    non_quant_impl_pred = f"acels\\full_design_output_{model_id}.csv"

    metrics_non_quant_pred_impl = compare_datasets(
        model_id,
        model_type_non_quant,
        original_csv_path,
        non_quant_impl_pred,
        data_exists,
    )
