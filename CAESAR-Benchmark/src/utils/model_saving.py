import torch


def load_model(model,
               filepath,
               optimizer=None,
               checkpoint_attribs=None,
               show_checkpoint_info=False,
               strict_load=True):

    model_checkpoint = torch.load(filepath, map_location=torch.device('cpu'))
    model.load_state_dict(model_checkpoint['state_dict'], strict=strict_load)
    
    if (optimizer is not None):
        optimizer.load_state_dict(model_checkpoint['optimizer_state'])
    
    if(checkpoint_attribs is not None):
        attrib_dict = {}
        for attrib in checkpoint_attribs:
            attrib_dict[attrib] = model_checkpoint[attrib]

        if (show_checkpoint_info):
            print('======Saved Model Info======')
            for attrib in checkpoint_attribs:
                print(attrib, model_checkpoint[attrib])
            print('=======Model======')
            print(model)
            print('#######Model#######')
            if (optimizer != None):
                print('=======Model======')
                print(optimizer)
                print('#######Model#######')

    print(f'loaded the model and optimizer successfully from {filepath}')

    return model, optimizer
