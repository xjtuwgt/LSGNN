from wikihop.wikihopdataset import WikiHopSpanDrop, BetaWikiHopSpanDrop
import random
span_drop_transforms = [WikiHopSpanDrop(drop_ratio=0.1),
                       WikiHopSpanDrop(drop_ratio=0.2)
                        ]
beta_drop_transforms = [BetaWikiHopSpanDrop(drop_ratio=0.1),
                       BetaWikiHopSpanDrop(drop_ratio=0.2)]

class SpanDropFunc:
    def __init__(self, span_drop_funcs, beta_span_drop_funcs):
        self.drop_funcs = span_drop_funcs
        self.beta_drop_funcs = beta_span_drop_funcs

    def __call__(self, x):
        if random.random() > 0.5:
            func_1 = random.choice(self.drop_funcs)
            func_2 = random.choice(self.beta_drop_funcs)
        else:
            func_1 = random.choice(self.beta_drop_funcs)
            func_2 = random.choice(self.drop_funcs)
        return func_1(x), func_2(x)

class TwoAugTransform:
    """Take two random crops of one sequence as the query and key."""
    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q, k = self.base_transform(x)
        return [q, k]

def are_eq(a, b):
    return set(a) == set(b) and len(a) == len(b)

if __name__ == '__main__':
    import itertools

    x = [list(range(10)), list(range(10, 20))]
    # print(x)
    x_len = len(list(itertools.chain(*x)))
    # func = BetaWikiHopSpanDrop(drop_ratio=0.45)
    # print(func(x))

    func = SpanDropFunc(span_drop_transforms, beta_drop_transforms)
    two_aug = TwoAugTransform(base_transform=func)



    # y1, y2 = two_aug(x)
    #
    # print(y1)
    # print(y2)


    count = 0
    N = 20000
    for i in range(N):
        y1, y2 = two_aug(x)

        y1 = list(itertools.chain(*y1))
        y2 = list(itertools.chain(*y2))

        y3, y4 = two_aug(x)
        y3 = list(itertools.chain(*y3))
        y4 = list(itertools.chain(*y4))

        y5, y6 = two_aug(x)
        y5 = list(itertools.chain(*y5))
        y6 = list(itertools.chain(*y6))

        y7, y8 = two_aug(x)
        y7 = list(itertools.chain(*y7))
        y8 = list(itertools.chain(*y8))
        if are_eq(set(y1), set(y2)) and are_eq(set(y3), set(y4)) and are_eq(set(y5), set(y6)) and are_eq(set(y7), set(y8)):
            print(y1)
            print(y2)
            count = count + 1

        # print(y1)
        # print(y2)
        # print()
        # # print(y)
        # # print(len(list(itertools.chain(*y))))
        # if len(list(itertools.chain(*y1))) == x_len:
        #     count = count + 1
    print(count/N)