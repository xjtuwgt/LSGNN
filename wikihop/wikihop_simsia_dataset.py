from wikihop.wikihopdataset import WikiHopSpanDrop, BetaWikiHopSpanDrop
import random
span_drop_transforms = [WikiHopSpanDrop(drop_ratio=0.1),
                       WikiHopSpanDrop(drop_ratio=0.2),
                        WikiHopSpanDrop(drop_ratio=0.3),
                       # BetaWikiHopSpanDrop(drop_ratio=0.1),
                       BetaWikiHopSpanDrop(drop_ratio=0.2)
                        ]

class SpanDropFunc:
    def __init__(self, span_drop_funcs):
        self.drop_funcs = span_drop_funcs

    def __call__(self, x):
        func = random.choice(self.drop_funcs)
        return func(x)

class TwoAugTransform:
    """Take two random crops of one sequence as the query and key."""
    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]

def are_eq(a, b):
    return set(a) == set(b) and len(a) == len(b)

if __name__ == '__main__':
    import itertools

    x = [list(range(10)), list(range(10, 20)), list(range(20, 30))]
    # print(x)
    x_len = len(list(itertools.chain(*x)))
    # func = BetaWikiHopSpanDrop(drop_ratio=0.45)
    # print(func(x))

    func = SpanDropFunc(span_drop_transforms)
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
        if are_eq(set(y1), set(y2)):
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