import numpy as np

from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
import torch
import argparse
from transformers import RobertaModel,RobertaTokenizer
from framework import RLADmodel
from dataloader import get_data
from torch.utils.data import DataLoader
from tqdm import tqdm
from train import setup_distributed

def main(args):
    rank, world_size = setup_distributed()
    if args.device is None:
        device = f'cuda:{rank}' if torch.cuda.is_available() else 'cpu'
    print('rank:', rank, 'world_size:', world_size, 'device:', device)
    print("Fetching data...")
    test_data_path = args.data_dir + args.test_data_file
    test_dataset = get_data( test_data_path)
    tokenizer = RobertaTokenizer.from_pretrained(args.pretrained_model)
    pre_model = RobertaModel.from_pretrained(args.pretrained_model)
    pre_model.requires_grad_(False)

    model = RLADmodel(tokenizer,pre_model,device,'')
    model_path=args.model+".pth"
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    test_loader=DataLoader(test_dataset,batch_size=1,shuffle=True)

    for epoch in range(args.epoch):
        model.eval()
        all_predictions = []
        all_targets = []
        all_outputs=[]

        with tqdm(test_loader, desc=f'Epoch {epoch + 1}/{args.epoch}') as loop:
            for sample_batch in loop:
                inputs = sample_batch["text"]
                targets = sample_batch["label"]
                outputs= model.predict(inputs)
                predictions = [0 if outputs[i][0] > outputs[i][1] else 1 for i in
                           range(len(outputs))]

                # Collect predictions and targets for metrics calculation
                all_predictions.extend(np.array(predictions))
                all_targets.extend(targets.cpu().numpy())
                all_outputs.append(outputs.cpu().detach().numpy())

                batch_accuracy = accuracy_score(all_targets, all_predictions)
                loop.set_postfix(acc=batch_accuracy)

        accuracy = accuracy_score(all_targets, all_predictions)
        auroc = roc_auc_score(all_targets, all_predictions)
        precision = precision_score(all_targets, all_predictions)
        recall = recall_score(all_targets, all_predictions)
        f1 = f1_score(all_targets, all_predictions)

        print(
            f"Epoch [{epoch + 1}/{args.epoch}] Accuracy: {accuracy:.4f}, AUROC: {auroc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument('--data_dir',type=str,default='M4/')
    parser.add_argument('--test_data_file', type=str, default='reddit_chatGPT_test.jsonl')
    parser.add_argument('--optimizer',type=str,default='Adam')
    parser.add_argument('--lr',type=float,default=0.0005)
    parser.add_argument('--epoch',type=int,default=1)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--device',type=str,default=None)
    parser.add_argument('--model',type=str,default="model/model_g_3,1")
    parser.add_argument('--pretrained_model', type=str, default="plm/roberta-base")
    args, unparsed = parser.parse_known_args()
    main(args)