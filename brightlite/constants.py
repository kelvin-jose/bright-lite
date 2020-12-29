class Training:

    class Defaults:
        DEVICE = 'cpu'
        EPOCHS = 1
        TRAIN_STATE = 'dead'
        TEST_STATE = 'dead'

    class WarmUp:
        TRAIN_STATE = 'warming up'
        TEST_STATE = 'warming up'

    class Active:
        TRAIN_STATE = 'alive'
        TEST_STATE = 'alive'
