from utils import create_data_lists

if __name__ == '__main__':
    create_data_lists(train_folders=['./Images/train'],
                      valid_folders=['./Images/val'],
                      test_folders=['./Images/test'],
                      min_size=100,
                      output_folder='./')
