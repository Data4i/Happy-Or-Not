import torch

def fit(net, epochs, trainloader, testloader, optimizer, criterion, acc_fn, display_range = 1, device = 'cpu',):
    epochs_count = list()
    train_loss_count = list()
    test_loss_count = list()
    train_acc_count = list()
    test_acc_count = list()
    
    # loop over the dataset multiple times
    for epoch in range(1, epochs):
        
        training_loss = .0
        training_acc = .0
        net.train()
        for i, batch in enumerate(trainloader, 0):
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)
    
            # zero the parameter gradients
            optimizer.zero_grad()
    
            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs.squeeze(), labels)
            train_acc = acc_fn(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()
    
            training_loss += loss.item()
            training_acc += train_acc.item()
            
        training_loss /= len(trainloader)
        train_loss_count.append(training_loss)
        training_acc /= len(trainloader)
        train_acc_count.append(training_acc)
        
        
        testing_loss, testing_acc = .0, .0
        net.eval()
        with torch.inference_mode():
            for test_batch in testloader:
                test_inputs, test_labels = test_batch
                test_inputs, test_labels = test_inputs.to(device), test_labels.to(device)
                
                test_outputs = net(test_inputs)
                loss = criterion(test_outputs.squeeze(), test_labels)
                test_acc = acc_fn(outputs.squeeze(), labels)
                
                testing_loss += loss
                testing_acc += test_acc.item()
                
            testing_loss /= len(testloader)
            test_loss_count.append(testing_loss)
            testing_acc /= len(testloader)
            test_acc_count.append(testing_acc)
            
        epochs_count.append(epoch)
        
        if epoch % display_range == 0:    
            print(f'Epoch: {epoch}')
            print(f'Train Loss: {training_loss}, Training Accuracy : {training_acc}')
            print(f'Test Loss: {testing_loss}, Test Accuracy : {testing_acc}')
            print('-------------------------------------------------------')
            print('-------------------------------------------------------')
    
    print('Finished Training')
    
    return {
        'epochs': epochs_count,
        'train_loss': train_loss_count,
        'test_loss': test_loss_count,
        'train_acc': train_acc_count,
        'test_acc': test_acc_count
    }