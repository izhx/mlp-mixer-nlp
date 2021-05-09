
local embedding_dim = 300;
local mixer_tsh = {
  type: "mixer_in_timestep_hidden",
  hidden_size: embedding_dim,
  num_layers: 6,
  num_tokens: 128,
  expansion_factor: 1.0,
  dropout: 0.5
};
local lstm = {
  "type": "lstm",
  "input_size": embedding_dim,
  "hidden_size": 512,
  "num_layers": 2
};

local encoder = mixer_tsh;

{
  "dataset_reader": {
    "type": "sst_tokens",
    "use_subtrees": false,
    "granularity": "2-class"
  },
  "validation_dataset_reader": {
    "type": "sst_tokens",
    "use_subtrees": false,
    "granularity": "2-class"
  },
  "train_data_path": "https://allennlp.s3.amazonaws.com/datasets/sst/train.txt",
  "validation_data_path": "https://allennlp.s3.amazonaws.com/datasets/sst/dev.txt",
  "test_data_path": "https://allennlp.s3.amazonaws.com/datasets/sst/test.txt",
  evaluate_on_test: true,
  "model": {
    "type": "basic_classifier",
    "text_field_embedder": {
      "token_embedders": {
        "tokens": {
          "type": "embedding",
          "embedding_dim": embedding_dim,
          "pretrained_file": "/home/data/embedding/glove.840B.300d.txt",
          "trainable": false
        }
      }
    },
    "seq2vec_encoder": encoder
  },
  "data_loader": {
    "batch_sampler": {
      "type": "bucket",
      "batch_size" : 32,
      sorting_keys: ["tokens"]
    }
  },
  "trainer": {
    "num_epochs": 50,
    "patience": 5,
    "grad_norm": 5.0,
    "validation_metric": "+accuracy",
    "optimizer": {
      "type": "adam",
      "lr": 0.001
    }
  }
}
