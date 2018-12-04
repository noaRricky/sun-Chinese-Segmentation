from allennlp.common.testing import ModelTestCase


class SimpleTaggerTest(ModelTestCase):
    def setUp(self):
        super(SimpleTaggerTest, self).setUp()
        self.set_up_model('tests/fixture/simple.json',
                          'tests/fixture/peoples.txt')

    def test_model_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)
