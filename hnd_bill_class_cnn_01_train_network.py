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
    if len(argv) < 5:
        print("Uso:")
        print(
            f"\tpython {argv[0]} in_tags in_dir cuda model_out use_resnet trained_model")
        return
    in_etiquetas = argv[1]
    in_dir = argv[2]
    use_cuda = argv[3] in ["True", "true", "enable", "yes", "Yes"]
    device = 'cuda' if use_cuda and torch.cuda.is_available() else 'cpu'
    model_out = argv[4]
    use_resnet = argv[5] in ["True", "true", "enable", "yes", "Yes"]
    pretrain = len(argv) == 7
    model_name = argv[5] if pretrain else None

    etiquetas = load(open(in_etiquetas, 'r'))

    # constants for training
    num_workers = 0
    fails = 0
    batch_size = 36
    valid_size = 0.02
    epochs = 2500
    preprocessed = True

    # preprocesado
    if preprocessed:
        target_w, target_h = 320, 128
        ratio_w, ratio_h = 5, 2
    else:
        target_w, target_h = 256, 256
        ratio_w, ratio_h = 4, 4

    trainset = BilletesDataset(
        root_dir=in_dir, etiquetas=etiquetas, size=(target_w, target_h))
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

    if use_resnet:
        print("Model utilizando resnet")
        torch.hub._validate_not_a_forked_repo=lambda a,b,c: True
        net = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
    else:
        net = LempiraNet(ratio_width=ratio_w, ratio_height=ratio_h, out=18)

    if pretrain and not use_resnet:
        print("Model pre entrenado se botara la ultima layer")
        net.load_state_dict(torch.load(
            model_name, map_location=torch.device(device)))
        net.drop_last_layer(len(trainset.classes))
    elif pretrain and use_resnet:
        new_linear = torch.nn.Linear(net.fc.in_features, 18)
        net.fc = new_linear

    if use_cuda:
        net = net.to(device)
        print('Modelo enviado a CUDA')

    # funcion de perdida (cross entropy loss)
    criterion = torch.nn.CrossEntropyLoss()
    # optimizador
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001)

    # para llevar la perdida del set de validación
    minimum_validation_loss = np.inf

    for epoch in range(1, epochs+1):
        train_loss_sum = 0
        valid_loss_sum = 0

        # training steps
        net.train()
        for batch_index, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            if use_cuda:
                data = data.cuda()
                target = target.cuda()

            output = net(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item()*data.size(0)

        # validation steps
        net.eval()
        for batch_index, (data, target) in enumerate(valid_loader):
            if use_cuda:
                data = data.cuda()
                target = target.cuda()

            output = net(data)
            loss = criterion(output, target)
            valid_loss_sum += loss.item()*data.size(0)

        valid_loss = valid_loss_sum/(len(valid_loader)*batch_size)
        train_loss = train_loss_sum/(len(train_loader)*batch_size)
        print(
            f'Epoch {epoch}\t Training Loss: {train_loss}\t Validation Loss:{valid_loss}')
        # Guardando el modelo cada vez que la perdida de validación decrementa.
        if valid_loss <= minimum_validation_loss:
            fails = 0
            print(
                f'Validation loss decreased from {round(minimum_validation_loss, 6)} to {round(valid_loss, 6)}')
            torch.save(net.state_dict(), model_out)
            minimum_validation_loss = valid_loss
            print('Saving New Model')
        else:
            # si las fallas llega a 10, se cierra el programa y se guarda el modelo
            fails += 1
            if fails >= 100:
                print('Loss haven\'t decrease in a time! Saving Last Model')
                torch.save(net.state_dict(), model_out)
                minimum_validation_loss = valid_loss
                exit(0)


if __name__ == '__main__':
    main()
