class Training:

    class Defaults:
        DEVICE = 'cpu'
        GPU = 'cuda:0'
        EPOCHS = 1
        TRAIN_STATE = 'dead'
        TEST_STATE = 'dead'

    class WarmUp:
        TRAIN_STATE = 'warming up'
        TEST_STATE = 'warming up'

    class Active:
        TRAIN_STATE = 'alive'
        TEST_STATE = 'alive'


class ErrorMsgs:

    class Training:
        TRAIN_LOADER_NONE = 'train loader cannot be none'
        EPOCHS_FAIL = 'epochs assertion failed'
