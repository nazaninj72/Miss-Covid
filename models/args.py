
import argparse
parser = argparse.ArgumentParser()
parser.add_argument(
        '--max_length',
        type=int,
        default=160,
        help='Maximum sequence length'
    )
parser.add_argument(
        '--embedding-size',
        type=int,
        default=768,
        choices=[768, 1024],
        help='Embedding size to be used. If the chosen model is bert, embedding-size should be 768 for ct-bert 1024 should be chosen.'
    )

# answer = args.square**2
# if args.verbosity == 2:
#     print("the square of {} equals {}".format(args.square, answer))
# elif args.verbosity == 1:
#     print("{}^2 == {}".format(args.square, answer))
# else:
#     print(answer)