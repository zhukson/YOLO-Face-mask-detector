import torch
from torch.utils.data import DataLoader
from Mask_data_set import mask_data_set
from model import Model
from loss_function import Loss_function
from loss_v2 import Loss_yolov1

if __name__ == '__main__':
    epoch = 40
    batchsize = 5
    lr = 0.01
    dataset = mask_data_set()
    train_dataloader = DataLoader(dataset, batch_size=batchsize, shuffle=True)
    model = Model().cuda()
    for layer in model.children():
        layer.requires_grad = False
        break
    criterion = Loss_yolov1()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
    for e in range(epoch):
        model.train()
        yl = torch.Tensor([0])
        for i, (inputs, labels) in enumerate(train_dataloader):
            inputs = inputs.float().cuda()
            labels = labels.float().cuda()
            pred = model(inputs)
            loss = criterion(pred, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print("Epoch %d/%d| Step %d/%d| Loss: %.2f" % (e, epoch, i, len(dataset) // batchsize, loss))
            yl = yl + loss
        if (e + 1) % 10 == 0:
            torch.save(model, "./saved_models/YOLOv1_epoch" + str(e + 1) + ".pkl")
            # compute_val_map(model)
