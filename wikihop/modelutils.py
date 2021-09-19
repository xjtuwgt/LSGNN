from tqdm import tqdm
import torch

def wikihop_model_evaluation(args, model, dataloader):
    model.eval()
    logs = []
    pred_dict = {}
    for batch in tqdm(dataloader):
        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        for key, value in batch.items():
            if key not in {'id'}:
                batch[key] = value.to(args.device)
        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        with torch.no_grad():
            scores = model(batch)
            scores = scores.squeeze(dim=-1)
            batch_size, cand_ans_num = scores.shape
            cand_mask = batch['cand_mask']
            sigmoid_scores = torch.sigmoid(scores)
            sigmoid_scores[cand_mask==0] = -1
            labels = batch['label']
            label_idxes = batch['label_id'].squeeze(dim=-1).tolist()

            pred_labels = torch.argmax(sigmoid_scores, dim=-1).tolist()
            true_labels = torch.argmax(labels, dim=-1).tolist()

            for idx in range(batch_size):
                example_id = batch['id'][idx]
                pred_dict[example_id] = pred_labels[idx]
                assert true_labels[idx] == label_idxes[idx]
                logs.append({'accuracy': 1.0 if pred_labels[idx] == true_labels[idx] else 0.0})
            del batch
    metrics = {}
    for metric in logs[0].keys():
        metrics[metric] = sum([log[metric] for log in logs]) / len(logs)
    return metrics, pred_dict


def model_parameter_summary(m: torch.nn.Module, only_trainable: bool = False):
    """
    returns the total number of parameters used by `m` (only counting
    shared parameters once); if `only_trainable` is True, then only
    includes parameters with `requires_grad = True`
    """
    parameters = list(m.parameters())
    if only_trainable:
        parameters = [p for p in parameters if p.requires_grad]
    unique = {p.data_ptr(): p for p in parameters}.values()
    model_para_number = sum(p.numel() for p in unique)
    print('Number of parameters of model = {}'.format(model_para_number))
    return model_para_number