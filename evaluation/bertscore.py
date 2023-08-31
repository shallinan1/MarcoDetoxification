from datasets import load_metric

def get_bert_scores(evaluation_dataset, lang = 'en', id = None):
    bert_metric = load_metric('bertscore', experiment_id = id)
    for model_input, gold_references in evaluation_dataset:
        bert_metric.add_batch(predictions=[model_input], references=[gold_references])

    final_score = bert_metric.compute(lang = lang)
    return final_score