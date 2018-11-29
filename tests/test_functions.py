import image

from torch.utils.data import DataLoader


class TestModule(object):
    def test_one(self):
        assert 'h' in 'help'

    def test_load(self):
        loaders, sizes = image.load_and_process_data()

        assert isinstance(loaders['train'], DataLoader) and isinstance(
            loaders['valid'], DataLoader), "DataLoader not instantiated correctly"

        assert sizes['train'] == 6552 and sizes['valid'] == 818

    def test_cat_name(self):
        assert image.get_cat_to_name() is not None, "File cat_to_name.json not found"

