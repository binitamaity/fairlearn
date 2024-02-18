from fairlearn.metrics import demographic_parity_difference



print(demographic_parity_difference(y_true,
                                    y_pred,
                                    sensitive_features=sf_data))