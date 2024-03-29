from Lempira import LempiraNet
from indoor_dataset import IndoorDataset
from sys import argv
import torch
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler


def main():
    if len(argv) < 4:
        print("Uso:")
        print(f"\tpython {argv[0]} imgs_file in_dir model_out cuda reload")
        return
    imgs_file = argv[1]
    in_dir = argv[2]
    model_out = argv[3]
    use_cuda = argv[4] in ["True", "true", "enable", "yes", "Yes"]
    device = 'cuda' if use_cuda and torch.cuda.is_available() else 'cpu'
    should_reload = False
    if len(argv) == 6:
        should_reload = True
    with open(imgs_file, 'r') as img_list:
        lines = img_list.readlines()
    clean_lines = list(map(lambda s: s.strip(), lines))

    # constants for training
    num_workers = 0
    fails = 0
    batch_size = 32
    valid_size = 0.1
    epochs = 5000
    preprocessed = True
    max_padding = 0.10

    flip_chance = 0.0
    color_change_chance = 0.1

    if preprocessed:
        img_size = (320, 128)
        ratio_w, ratio_h = 5, 2
    else:
        img_size = (256, 256)
        ratio_w, ratio_h = 4, 4

    trainset = IndoorDataset(
        root_dir=in_dir, img_files=clean_lines, training=True,
        size=img_size, max_padding=max_padding,
        flip_chance=flip_chance, color_change_chance=color_change_chance)
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

    net = LempiraNet(ratio_width=ratio_w, ratio_height=ratio_h, out=67)
    if should_reload:
        net.load_state_dict(torch.load(
            model_out, map_location=torch.device(device)))
            
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
