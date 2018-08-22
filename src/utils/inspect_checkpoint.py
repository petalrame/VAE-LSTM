from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file #pylint: disable=E0611

if __name__ == '__main__':
    print_tensors_in_checkpoint_file(file_name='/home/tldr/Projects/models/current/VAE-LSTM/results/test/model.ckpt-600', tensor_name='', all_tensors=True)