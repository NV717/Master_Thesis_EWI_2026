import pandas as pd

def decode_static(pp, cat, cont, boolean):
    cat_df = pd.DataFrame(pp.cat_encoder.inverse_transform((cat.cpu().numpy() - 1)),
                          columns=pp.static_cat_cols)
    cont_df = pd.DataFrame(pp.cont_scaler.inverse_transform(cont.cpu().numpy()),
                           columns=pp.static_cont_cols)
    bool_df = pd.DataFrame(boolean.cpu().numpy() > 0.5, columns=pp.static_bool_cols)
    return pd.concat([cat_df, cont_df, bool_df], axis=1)
