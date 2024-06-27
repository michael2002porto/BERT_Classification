import random
import sys

import torch
import torch.nn as nn

import pytorch_lightning as pl

from transformers import BertModel
from sklearn.metrics import classification_report

from torchmetrics import Accuracy, F1Score, PrecisionRecallCurve

class MultiClassModel(pl.LightningModule):
    def __init__(self,
                 dropout,
                 n_out,
                 lr) -> None:
        super(MultiClassModel, self).__init__()

        # seed untuk weight
        torch.manual_seed(1) # Untuk GPU
        random.seed(1) # Untuk CPU

        # to make use of all the outputs from each training_step()
        self.training_step_outputs = []

        # to make use of all the outputs from each predict_step()
        self.predict_step_outputs = []

        # inisialisasi bert
        # sudah di training terhadap dataset tertentu oleh orang di wikipedia
        self.bert = BertModel.from_pretrained('indolem/indobert-base-uncased')

        # hasil dimasukkan ke linear function
        # pre_classifier = agar weight tidak hilang ketika epoch selanjutnya. Agar weight dapat digunakan kembali
        self.pre_classifier = nn.Linear(768, 768)

        self.dropout = nn.Dropout(dropout)

        # n_out = jumlah label
        self.num_classes = n_out
        # jumlah label = 5
        # classifier untuk merubah menjadi label
        self.classifier = nn.Linear(768, n_out)

        self.lr = lr

        # menghitung loss function
        self.criterion = nn.BCEWithLogitsLoss()

        self.accuracy = Accuracy(task="multiclass", num_classes = self.num_classes)
        self.f1 = F1Score(task = "multiclass", 
                          average = "micro", 
                          multidim_average = "global",
                          num_classes = self.num_classes)
        self.precission_recall = PrecisionRecallCurve(task = "multiclass", num_classes = self.num_classes)

    # mengambil input dari bert, pre_classifier
    def forward(self, input_ids, attention_mask, token_type_ids):
        bert_out = self.bert(input_ids = input_ids,
                             attention_mask = attention_mask,
                             token_type_ids = token_type_ids)

        hidden_state = bert_out[0]
        pooler = hidden_state[:, 0]
        # Output size (batch size = 20 baris, sequence length = 100 kata / token, hidden_size = 768 tensor jumlah vektor representation dari)

        # pre classifier untuk mentransfer wight output ke epch selanjuntya
        pooler = self.pre_classifier(pooler)
        # kontrol hasil pooler min -1 max 1
        pooler = torch.nn.Tanh()(pooler)

        pooler = self.dropout(pooler)
        # classifier untuk memprojeksikan hasil pooler (768) ke jumlah label (5)
        output = self.classifier(pooler)

        return output

    def configure_optimizers(self):
        # di dalam parameter adam, parameters untuk mengambil kesuluruhan input yg di atas

        # Fungsi adam 
        # Tranfer epoch 1 ke epoch 2
        # Mengontrol (efisiensi) loss
        # Proses training lebih cepat
        # Tidak memakan memori berlebih
        optimizer = torch.optim.Adam(self.parameters(), lr = self.lr)
        return optimizer

        #Learning rate semakin tinggi maka hasil itunya semakin besar
    
    def training_step(self, batch, batch_idx):
        x_input_ids, x_token_type_ids, x_attention_mask, y = batch

        out = self(input_ids = x_input_ids,
                   attention_mask = x_attention_mask,
                   token_type_ids = x_token_type_ids)
        # Ke tiga parameter di input dan di olah oleh method / function forward

        loss = self.criterion(out, target = y.float())

        pred = out.argmax(1).cpu()
        true = y.argmax(1).cpu()

        self.accuracy(out, y)
        report = classification_report(true, pred, output_dict = True, zero_division = 0)

        self.log("accuracy", report["accuracy"], prog_bar = True)
        self.log("loss", loss)

        outputs = {"loss": loss, "predictions": out, "labels": y}
        self.training_step_outputs.append(outputs)

        return outputs

    def validation_step(self, batch, batch_idx):
        # Tidak transfer weight
        x_input_ids, x_token_type_ids, x_attention_mask, y = batch

        out = self(input_ids = x_input_ids,
                   attention_mask = x_attention_mask,
                   token_type_ids = x_token_type_ids)
        # Ke tiga parameter di input dan di olah oleh method / function forward

        loss = self.criterion(out, target = y.float())

        pred = out.argmax(1).cpu()
        true = y.argmax(1).cpu()

        report = classification_report(true, pred, output_dict = True, zero_division = 0)
        self.accuracy(out, y)

        self.log("accuracy", report["accuracy"], prog_bar = True)
        self.log("loss", loss)

        return loss

    def predict_step(self, batch, batch_idx):
        # Tidak ada transfer weight
        x_input_ids, x_token_type_ids, x_attention_mask, y = batch

        out = self(input_ids = x_input_ids,
                   attention_mask = x_attention_mask,
                   token_type_ids = x_token_type_ids)
        # Ke tiga parameter di input dan di olah oleh method / function forward

        pred = out.argmax(1).cpu()
        true = y.argmax(1).cpu()

        outputs = {"predictions": out, "labels": y}
        self.predict_step_outputs.append(outputs)

        # return [pred, true]
        return outputs

    def on_train_epoch_end(self):
        labels = []
        predictions = []

        for output in self.training_step_outputs:
            for out_lbl in output["labels"].detach().cpu():
                labels.append(out_lbl)
            for out_pred in output["predictions"].detach().cpu():
                predictions.append(out_pred)

        # argmax(dim=1) = convert one-hot encoded labels to class indices
        labels = torch.stack(labels).int().argmax(dim=1)
        predictions = torch.stack(predictions).argmax(dim=1)

        print("\n")
        print("labels = ", labels)
        print("predictions = ", predictions)
        print("num_classes = ", self.num_classes)

        # Hitung akurasi
        accuracy = Accuracy(task = "multiclass", num_classes = self.num_classes)
        acc = accuracy(predictions, labels)

        # Print Akurasinya
        print("Overall Training Accuracy : ", acc)
        print("\n")
        # sys.exit()

        # free memory
        self.training_step_outputs.clear()

    def on_predict_epoch_end(self):
        labels = []
        predictions = []

        for output in self.predict_step_outputs:
            # print(output[0]["predictions"][0])
            # print(len(output))
            # break
            for out_lbl in output["labels"].detach().cpu():
                print(out_lbl)
                sys.exit()
                labels.append(out_lbl)
            for out_pred in output["predictions"].detach().cpu():
                predictions.append(out_pred)

        # argmax(dim=1) = convert one-hot encoded labels to class indices
        labels = torch.stack(labels).int().argmax(dim=1)
        predictions = torch.stack(predictions).argmax(dim=1)

        print("\n")
        print("labels = ", labels)
        print("predictions = ", predictions)
        print("num_classes = ", self.num_classes)

        accuracy = Accuracy(task = "multiclass", num_classes = self.num_classes)
        acc = accuracy(predictions, labels)
        print("Overall Testing Accuracy : ", acc)
        print("\n")
        # sys.exit()

        # free memory
        self.predict_step_outputs.clear()