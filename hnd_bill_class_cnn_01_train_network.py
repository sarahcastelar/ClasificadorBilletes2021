from Lempira import LempiraNet
from billetes_dataset import BilletesDataset
from sys import argv
from json import load
import torch
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler


def model_summary(model):
    print("model_summary")
    print()
    print("Layer_name"+"\t"*7+"Number of Parameters")
    print("="*100)
    model_parameters = [
        layer for layer in model.parameters() if layer.requires_grad]
    layer_name = [child for child in model.children()]
    j = 0
    total_params = 0
    print("\t"*10)
    for i in layer_name:
        print()
        param = 0
        try:
            bias = (i.bias is not None)
        except:
            bias = False
        if not bias:
            param = model_parameters[j].numel()+model_parameters[j+1].numel()
            j = j+2
        else:
            param = model_parameters[j].numel()
            j = j+1
        print(str(i)+"\t"*3+str(param))
        total_params += param
    print("="*100)
    print(f"Total Params:{total_params}")


def main():
    if len(argv) < 3:
        print("Uso:")
        print(f"\tpython {argv[0]} in_tags in_dir")
        return
    in_etiquetas = argv[1]
    in_dir = argv[2]
    etiquetas = load(open(in_etiquetas, 'r'))

    # constants for training
    num_workers = 0
    fails = 0
    batch_size = 36
    valid_size = 0.02
    epochs = 25

    trainset = BilletesDataset(
        root_dir=in_dir, etiquetas=etiquetas, size=(320, 128))
    # Finding indices for validation set
    num_train = len(trainset)
    indices = list(range(num_train))

    # Randomize indices
    np.random.shuffle(indices)
    split = int(np.floor(num_train*valid_size))

    # Making samplers for training and validation batches
    train_index, test_index = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_index)
    valid_sampler = SubsetRandomSampler(test_index)

    # creando los  data loaders
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers)

    valid_loader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, sampler=valid_sampler, num_workers=num_workers)

    # test_loader = torch.utils.data.DataLoader(
    #     test_data, batch_size=batch_size, num_workers=num_workers)

    net = LempiraNet(ratio_width=5, ratio_height=2)
    model_summary(net)

    # funcion de perdida (cross entropy loss)
    criterion = torch.nn.CrossEntropyLoss()
    # optimizador
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01)

    # para llevar la perdida del set de validación
    minimum_validation_loss = np.inf

    for epoch in range(1, epochs+1):
        train_loss = 0
        valid_loss = 0

        # training steps
        net.train()
        for batch_index, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = net(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()*data.size(0)

        # validation steps
        net.eval()
        for batch_index, (data, target) in enumerate(valid_loader):
            output = net(data)
            loss = criterion(output, target)
            valid_loss += loss.item()*data.size(0)

        print(
            f'Epoch {epoch}\t Training Loss: {train_loss/len(train_loader)}\t Validation Loss:{valid_loss/len(valid_loader)}')
        # Guardando el modelo cada vez que la perdida de validación decrementa.
        if valid_loss <= minimum_validation_loss:
            fails = 0 
            print(
                f'Validation loss decreased from {round(minimum_validation_loss, 6)} to {round(valid_loss, 6)}')
            torch.save(net.state_dict(), 'trained_model.pt')
            minimum_validation_loss = valid_loss
            print('Saving New Model')
        else:
            # si las fallas llega a 10, se cierra el programa y se guarda el modelo
            fails += 1
            if fails >= 10:
                print('Loss haven\'t decrease in a time! Saving Last Model')
                torch.save(net.state_dict(), 'trained_model.pt')
                minimum_validation_loss = valid_loss
                exit(0)


if __name__ == '__main__':
    main()
