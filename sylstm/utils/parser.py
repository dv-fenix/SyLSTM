import configargparse as cfargparse


class ArgumentParser(cfargparse.ArgumentParser):
    def __init__(
        self,
        config_file_parser_class=cfargparse.YAMLConfigFileParser,
        formatter_class=cfargparse.ArgumentDefaultsHelpFormatter,
        **kwargs
    ):
        super(ArgumentParser, self).__init__(
            config_file_parser_class=config_file_parser_class,
            formatter_class=formatter_class,
            **kwargs
        )

    @classmethod
    def defaults(cls, *args):
        """Get default arguments added to a parser by all ``*args``."""
        dummy_parser = cls()
        for callback in args:
            callback(dummy_parser)
        defaults = dummy_parser.parse_known_args([])[0]
        return defaults

    @classmethod
    def validate_train_opts(cls, train_opt):
        if train_opt.resume and train_opt.model_path is None:
            raise AssertionError(
                "Please specify the path to the saved model checkpoint."
            )

        if train_opt.use_glove and train_opt.embedding_dim != 200:
            raise AssertionError(
                "The embedding dimensionlaity for the pretrained Glove Twitter Embeddings is 200."
            )
