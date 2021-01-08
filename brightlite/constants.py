class Training:

    class Defaults:
        DEVICE = 'cpu'
        CUDA = 'cuda:0'
        EPOCHS = 1
        TRAIN_STATE = 'dead'
        VALID_STATE = 'dead'
        USE_MULTIPLE_GPUS = False

    class WarmUp:
        TRAIN_STATE = 'warming up'
        VALID_STATE = 'warming up'

    class Active:
        TRAIN_STATE = 'alive'
        VALID_STATE = 'alive'

    class Pause:
        TRAIN_STATE = 'on pause'
        VALID_STATE = 'on pause'


class ErrorMsgs:

    class Training:
        TRAIN_LOADER_NONE = 'train loader cannot be none'
        EPOCHS_FAIL = 'epochs assertion failed'
