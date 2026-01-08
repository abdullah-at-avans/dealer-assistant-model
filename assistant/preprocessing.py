from .printer import log_info
from .timer import timed

def preprocess_messages(messages_df):
    messages_df = messages_df[messages_df['remark_type'] == 'customer_contact']

    drop_columns = [
        'crm_partner_id',
        'to_crm_partner_id',
        'read',
        'remark_status',
        'postpone_datetime',
        'crm_partner_completed_id',
        'completed_datetime',
        'crm_partner_in_progress_id',
        'in_progress_datetime',
        'source_id',
        'created',
        'modified',
        'department'
    ]

    messages_df = messages_df.drop(
        # Alleen kollommen die nu bestaan in de dataframe verwijderen
        columns=[col for col in drop_columns if col in messages_df.columns.tolist()]
    )

    # klant berichten (1821 entries) ⊆ totale berichten (848127 entries); -99.79%
    messages_df.head()
    return messages_df

def preprocess_works(works_df):
    return works_df.drop_duplicates(subset="description", keep="last")

def preprocess(datasets: dict):
    datasets['messages'] = timed('Pre processing messages', preprocess_messages, datasets['messages'])
    datasets['works'] = timed('Pre processing works', preprocess_works, datasets['works'])
    # ... de rest van de datasets zijn op vroege fase (extraction) al gepreprocessed

    return datasets
